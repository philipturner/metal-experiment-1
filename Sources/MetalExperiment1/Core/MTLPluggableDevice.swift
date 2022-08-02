//
//  MTLPluggableDevice.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

/// A wrapper around `MTLDevice` that eagerly executes arbitrary operations.
public class MTLPluggableDevice {
  static var profilingEncoding = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_COMMAND_STREAM")
  
  /// A pluggable device that encapsulates the system default `MTLDevice`.
  public static let `default`: MTLPluggableDevice = {
    let device = MTLPluggableDevice(mtlDevice: MTLCreateSystemDefaultDevice()!, isDefault: true)
    deviceCacheMutex.sync {
      withUnsafeAddress(of: device.mtlDevice) { address in
        deviceCache[address] = device
      }
    }
    return device
  }()
  
  /// A pluggable device that encapsulates the specified `MTLDevice`.
  ///
  /// Creating a pluggable device is costly, so this fetches the return value from an internal cache
  /// by default. Disabling the cache mechanism means two different `MTLPluggableDevice` instances
  /// could wrap the same `MTLDevice`. With caching disabled, a machine with only one GPU can
  /// simulate a multi-GPU system.
  ///
  /// - Parameters:
  ///   - mtlDevice: The `MTLDevice` to execute operations on.
  ///   - fromCache: Whether to fetch the return value from an internal cache.
  /// - Returns: The pluggable device.
  public static func custom(mtlDevice: MTLDevice, fromCache: Bool = true) -> MTLPluggableDevice {
    withUnsafeAddress(of: mtlDevice) { address in
      if fromCache {
        let cachedDevice = deviceCacheMutex.sync {
          deviceCache[address]
        }
        if let cachedDevice = cachedDevice {
          return cachedDevice
        }
      }
      
      let newDevice = MTLPluggableDevice(mtlDevice: mtlDevice, isDefault: false)
      
      // If caching is turned off, do not insert the device into the cache.
      if fromCache {
        deviceCacheMutex.sync {
          precondition(deviceCache[address] == nil, "This should never happen.")
          deviceCache[address] = newDevice
        }
      }
      return newDevice
    }
  }
  
  private static var deviceCache: [UnsafeMutableRawPointer: MTLPluggableDevice] = [:]
  
  private static var deviceCacheMutex: Mutex = Mutex()
  
  /// The `MTLDevice` that executes operations.
  public private(set) var mtlDevice: MTLDevice
  
  /// Whether to store tensors in CPU-accessible memory. This defaults to `true` on systems with
  /// unified memory, and `false` otherwise.
  public var prefersSharedMemory: Bool {
    get {
      // Don't need to sync because booleans are inherently atomic. However, everything else is
      // synchronized with a mutex lock.
      self.sync {
        self._prefersSharedMemory
      }
    }
    set {
      // Disable changing shared memory preference while encoding instructions.
      self.sync {
        self._prefersSharedMemory = newValue
      }
    }
  }
  
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
  var _prefersSharedMemory: Bool
  
  var heapAllocator: HeapAllocator
  var shaderCache: ShaderCache
  // TODO: Cache for MPSKernel objects (if needed)
  // TODO: Cache for MPSGraph objects
  
  // Using mutex locks instead of GCD for fast synchronization across processes.
  #if os(Windows)
  private var _mutex: SRWLOCK
  #else
  private var _mutex: pthread_mutex_t
  #endif
  
  private init(mtlDevice: MTLDevice, isDefault: Bool = false) {
    self.mtlDevice = mtlDevice
    self.isDefault = isDefault
    
    // Initialize shader cache, MPSGraph cache, etc first. They create asynchronous tasks that run
    // on background threads.
    self.shaderCache = ShaderCache(mtlDevice: mtlDevice)
    self.heapAllocator = HeapAllocator(mtlDevice: mtlDevice)
    
    self.commandQueue = mtlDevice.makeCommandQueue(maxCommandBufferCount: 3)!
    self._prefersSharedMemory = mtlDevice.hasUnifiedMemory
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
