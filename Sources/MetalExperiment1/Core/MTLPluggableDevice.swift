//
//  MTLPluggableDevice.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

/// An object for customizing initialization of a new Metal-backed pluggable device object.
public class MTLPluggableDeviceDescriptor: NSObject {
  /// Whether to fetch the pluggable device from an internal cache.
  ///
  /// The default value is `true`. Creating a pluggable device is costly, so this automatically
  /// fetches the pluggable device object from an internal cache. Disabling the cache mechanism
  /// means two different `MTLPluggableDevice` instances could wrap the same `MTLDevice`. With
  /// caching disabled, a machine with only one GPU can simulate a multi-GPU system.
  public var usesDeviceCache: Bool = true
  
  /// Whether the pluggable device stores tensors in CPU-accessible memory.
  ///
  /// The default value is `nil`. If not set, Metal automatically determines the fastest storage
  /// mode based on the `MTLDevice` that creates the pluggable device. The value defaults to `true`
  /// on devices with unified memory and `false` otherwise.
  public var prefersSharedMemory: Bool?
  
  // TODO: Mechanism to make something the default pluggable device if not set. This reduces
  // CPU-side overhead of accessing a device and lets the user intercept the frontend's
  // lazy initialization. They can choose what device gets the fast-path.
}

extension MTLDevice {
  /// Creates an object from a descriptor to execute arbitrary operations in a GPU-accelerated Swift
  /// framework.
  ///
  /// - Parameter descriptor: A description of the pluggable device to create.
  /// - Returns: A pluggable device object.
  public func makePluggableDevice(descriptor: MTLPluggableDeviceDescriptor) -> MTLPluggableDevice? {
    MTLPluggableDevice.deviceCacheMutex.sync {
      withUnsafeAddress(of: self) { address in
        let key = MTLPluggableDeviceCacheKey(
          mtlDeviceAddress: address,
          prefersSharedMemory: descriptor.prefersSharedMemory ?? self.hasUnifiedMemory)
        if descriptor.usesDeviceCache {
          if let cachedDevice = MTLPluggableDevice.deviceCache[key] {
            return cachedDevice
          }
        }
        
        var pluggableDevice: MTLPluggableDevice
        let defaultPluggableDevice = MTLPluggableDevice.default
        if descriptor.usesDeviceCache,
           self === defaultPluggableDevice.mtlDevice,
           key.prefersSharedMemory == defaultPluggableDevice.prefersSharedMemory {
          pluggableDevice = defaultPluggableDevice
        } else {
          pluggableDevice = MTLPluggableDevice(mtlDevice: self, isDefault: false)
        }
        
        // If caching is turned off, do not insert the device into the cache.
        if descriptor.usesDeviceCache {
          precondition(
            MTLPluggableDevice.deviceCache[key] == nil,
            "Pluggable device was already in the cache.")
          MTLPluggableDevice.deviceCache[key] = pluggableDevice
        }
        return pluggableDevice
      }
    }
  }
}

struct MTLPluggableDeviceCacheKey: Hashable {
  var mtlDeviceAddress: UnsafeMutableRawPointer
  var prefersSharedMemory: Bool
}

/// A wrapper around `MTLDevice` that eagerly executes arbitrary operations, using just-in-time
/// graph compilation to reduce overhead.
public class MTLPluggableDevice {
  static var profilingEncoding = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_COMMAND_STREAM")
  
  /// A pluggable device that encapsulates the system default `MTLDevice`. To access this in the
  /// public API, call `MTLDevice.makePluggableDevice` with default initialization parameters.
  static let `default`: MTLPluggableDevice = MTLPluggableDevice(
    mtlDevice: MTLCreateSystemDefaultDevice()!, isDefault: true)
  
  static var deviceCache: [MTLPluggableDeviceCacheKey: MTLPluggableDevice] = [:]

  static var deviceCacheMutex: Mutex = Mutex()
  
  /// The `MTLDevice` that executes operations.
  public private(set) var mtlDevice: MTLDevice
  
  /// Whether the pluggable device stores tensors in CPU-accessible memory.
  public private(set) var prefersSharedMemory: Bool
  
  var isDefault: Bool
  var commandQueue: MTLCommandQueue
  var commandBufferDictionary: [Int: MTLCommandBuffer] = [:]
  var synchronizationFence: MTLFence
  var synchronizationEvent: MTLEvent
  var synchronizationCounter: UInt64 = 0
  
  // `maxCommandsPerBatch` would be mutable if an AI changed parameters of JIT compilation.
  var maxCommandsPerBatch = 128
  var maxCommandsPerSmallBatch = 16
  var numCommittedBatches: UnsafeAtomic<Int> = .create(0)
  var numScheduledBatches: UnsafeAtomic<Int> = .create(0)
  var numCompletedBatches: UnsafeAtomic<Int> = .create(0)
  var eagerOperations: [EagerOperation] = []
  
  // Read the atomic from the same thread that's modifying it.
  func _fastLoadCommittedBatches() -> Int {
    let ptr = unsafeBitCast(numCommittedBatches, to: UnsafePointer<Int>.self)
    return ptr.pointee
  }
  
  var justFinishedBarrier = true
  var waitingOnTimer = false
  var schedulingLatency = MovingAverage<UInt64>(repeating: 50_000, count: 16)
  var schedulingQueue = DispatchQueue(
    label: "com.s4tf.metal.Context.schedulingQueue", qos: .userInteractive)
  
  var nextAllocationID: UInt64 = 0
  var numDeinitializedAllocations: UInt64 = 0
  var permitExceedingSystemRAM = false
  
  var heapAllocator: HeapAllocator
  var shaderCache: ShaderCache
  // TODO: Cache for MPSKernel objects (if needed)
  // TODO: Cache for MPSGraph objects
  
  // Using mutex locks instead of GCD for fast synchronization across threads. This directly calls C
  // functions instead of delegating to the `Mutex` wrapper, which would cause ARC overhead.
  #if os(Windows)
  private var _mutex: SRWLOCK
  #else
  private var _mutex: pthread_mutex_t
  #endif
  
  internal init(mtlDevice: MTLDevice, isDefault: Bool = false) {
    self.mtlDevice = mtlDevice
    self.isDefault = isDefault
    
    // Initialize shader cache, MPSGraph cache, etc first. They create asynchronous tasks that run
    // on background threads.
    self.shaderCache = ShaderCache(mtlDevice: mtlDevice)
    self.heapAllocator = HeapAllocator(mtlDevice: mtlDevice)
    
    self.commandQueue = mtlDevice.makeCommandQueue(maxCommandBufferCount: 3)!
    self.prefersSharedMemory = mtlDevice.hasUnifiedMemory
    self.synchronizationFence = mtlDevice.makeFence()!
    self.synchronizationEvent = mtlDevice.makeEvent()!
    
    self._mutex = .init()
    #if os(Windows)
    InitializeSRWLock(&_mutex)
    #else
    pthread_mutex_init(&_mutex, nil)
    #endif
  }
  
  deinit {
    numScheduledBatches.destroy()
    numCompletedBatches.destroy()
    
    #if os(Windows)
    // SRWLOCKs do not need explicit destruction
    #else
    pthread_mutex_destroy(&_mutex)
    #endif
  }
}

extension MTLPluggableDevice {
  @inline(__always)
  func acquireMutex() {
    #if os(Windows)
    AcquireSRWLockExclusive(&_mutex)
    #else
    let code = pthread_mutex_lock(&_mutex)
    
    // Only perform this check in debug mode because it appears frequently and in a hotpath.
    assert(code == 0, "Attempt to acquire mutex returned '\(code)'.")
    #endif
  }
  
  @inline(__always)
  func releaseMutex() {
    #if os(Windows)
    ReleaseSRWLockExclusive(&_mutex)
    #else
    let code = pthread_mutex_unlock(&_mutex)
    
    // Only perform this check in debug mode because it appears frequently and in a hotpath.
    precondition(code == 0, "Attempt to release mutex returned '\(code)'.")
    #endif
  }
  
  // Borrowed from `_ExecutionContext` in https://github.com/s4tf/s4tf
  @inline(__always)
  func sync<Result>(execute body: () throws -> Result) rethrows -> Result {
    acquireMutex()
    defer {
      releaseMutex()
    }
    return try body()
  }
}
