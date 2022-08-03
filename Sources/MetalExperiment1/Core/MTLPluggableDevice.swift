//
//  MTLPluggableDevice.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

/// The options you use to specify the pluggable device storage mode.
public enum MTLPluggableDeviceStorageMode: UInt {
  /// Metal chooses the fastest storage mode based on ``MTLPluggableDevice/mtlDevice``.
  case automatic = 0
  
  /// Store tensors in memory accessible to both the CPU and GPU.
  case shared = 1
  
  /// Store tensors in memory accessible only to the GPU.
  case `private` = 2
}

/// An object for customizing initialization of a new Metal-backed pluggable device object.
public class MTLPluggableDeviceDescriptor: NSObject {
  /// Whether to fetch the pluggable device from an internal cache.
  ///
  /// The default value is `true`. Creating a pluggable device is costly, so this automatically
  /// fetches the pluggable device object from an internal cache. Disabling the cache mechanism
  /// means two different `MTLPluggableDevice` instances could wrap the same `MTLDevice`. With
  /// caching disabled, a machine with only one GPU can simulate a multi-GPU system.
  public var usesDeviceCache: Bool = true
  
  /// The storage mode for tensors used to execute operations.
  ///
  /// The default value is `.automatic`.
  public var storageMode: MTLPluggableDeviceStorageMode = .automatic
  
  /// Whether to override the system default pluggable device.
  ///
  /// The default value is `.default`.
  ///
  /// To minimize synchronization overhead, Metal declares one pluggable device as the "default"
  /// device. Running operations on this device incurs lower CPU overhead than other devices. The
  /// "default" pluggable device is lazily initialized to an instance with the following properties.
  /// Once it is defined, it cannot be overriden until the app or script finishes.
  ///
  /// - ``MTLPluggableDevice/mtlDevice`` equals the value returned by
  ///   `MTLCreateSystemDefaultDevice()`.
  /// - ``MTLPluggableDevice/storageMode`` matches whether `mtlDevice` has unified memory.
  /// - The descriptor has `usesDeviceCache` set to `true`.
  ///
  /// In top-level scripting environments such as Jupyter, one can execute arbitrary code before the
  /// "default" pluggable device materializes. This provides an opportunity to specify which
  /// `MTLPluggableDevice` receives the synchronization fast-path. Otherwise, the first function
  /// call into a GPU-accelerated framework typically defines the default pluggable device.
  public var overrideMode: MTLPluggableDeviceOverrideMode = .default
}

/// The options you use to specify the pluggable device override mode.
public enum MTLPluggableDeviceOverrideMode: UInt {
  /// Do not set this instance as the default pluggable device.
  case never = 0
  
  /// Metal sets this as the default pluggable device if it satisfies certain conditions. See
  /// ``MTLPluggableDeviceDescriptor/overrideMode`` for a description of these conditions.
  case `default` = 1
  
  /// Override the default pluggable device if it has not been defined.
  case whenPossible = 2
  
  /// Invoke a runtime crash if the default pluggable device has already been defined.
  case always = 3
}

extension MTLDevice {
  /// Creates an object from a descriptor to execute arbitrary operations in a GPU-accelerated Swift
  /// framework.
  ///
  /// - Parameter descriptor: A description of the pluggable device to create.
  /// - Returns: A pluggable device object.
  public func makePluggableDevice(descriptor: MTLPluggableDeviceDescriptor) -> MTLPluggableDevice? {
    // TODO: Allow for initializing multiple Metal pluggable devices concurrently.
    MTLPluggableDevice.deviceCacheMutex.sync {
      withUnsafeAddress(of: self) { address in
        var storageMode: MTLStorageMode
        switch descriptor.storageMode {
        case .automatic:
          storageMode = self.hasUnifiedMemory ? .shared : .private
        case .shared:
          storageMode = .shared
        case .private:
          storageMode = .private
        }
        let key = MTLPluggableDeviceCacheKey(
          mtlDeviceAddress: address,
          storageMode: storageMode)
        if descriptor.usesDeviceCache {
          if let cachedDevice = MTLPluggableDevice.deviceCache[key] {
            return cachedDevice
          }
        }
        
        var pluggableDevice: MTLPluggableDevice
        let defaultPluggableDevice = MTLPluggableDevice.default
        if descriptor.usesDeviceCache,
           self === defaultPluggableDevice.mtlDevice,
           key.storageMode == defaultPluggableDevice.storageMode {
          pluggableDevice = defaultPluggableDevice
        } else {
          pluggableDevice = MTLPluggableDevice(mtlDevice: self, isDefault: false)
          pluggableDevice.prefersSharedMemory = key.storageMode == .shared
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
  var storageMode: MTLStorageMode
}

/// A wrapper around `MTLDevice` that eagerly executes arbitrary operations, using just-in-time
/// graph compilation to reduce overhead.
public class MTLPluggableDevice: NSObject {
  static var profilingEncoding = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_COMMAND_STREAM")
  
  /// A pluggable device that encapsulates the system default `MTLDevice`. To access this in the
  /// public API, call `MTLDevice.makePluggableDevice` with default initialization parameters.
  static let `default`: MTLPluggableDevice = MTLPluggableDevice(
    mtlDevice: MTLCreateSystemDefaultDevice()!, isDefault: true)
  
  static var deviceCache: [MTLPluggableDeviceCacheKey: MTLPluggableDevice] = [:]

  static var deviceCacheMutex: Mutex = Mutex()
  
  /// The `MTLDevice` that executes operations.
  public private(set) var mtlDevice: MTLDevice
  
  /// The resource options for tensors used to execute operations.
  ///
  /// Possible values are `.storageModeShared` and `.storageModePrivate`.
  public var resourceOptions: MTLResourceOptions {
    prefersSharedMemory ? .storageModeShared : .storageModePrivate
  }
  
  /// The storage mode for tensors used to execute operations.
  ///
  /// Possible values are `.shared` and `.private`.
  public var storageMode: MTLStorageMode {
    prefersSharedMemory ? .shared : .private
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
  var prefersSharedMemory: Bool
  
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
    
    super.init()
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
