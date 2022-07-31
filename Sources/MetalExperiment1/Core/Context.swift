//
//  Context.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

public class Context {
  static var profilingEncoding = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_COMMAND_STREAM")
  
  static let global = Context()
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var commandBufferDictionary: [Int: MTLCommandBuffer] = [:]
  var synchronizationFence: MTLFence
  var synchronizationEvent: MTLEvent
  var synchronizationCounter: UInt64 = 0
  
  // For leak checking in tests, lower overhead than S4TF's `Context.local.globalTensorCount`.
  var globalTensorCount: Int = 0
  
  static let maxBatchesInFlight = 3
  static var maxCommandsPerBatch = 128
  static var maxCommandsPerSmallBatch = 16
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
  var preferSharedStorage: Bool
  
  // Using mutex locks instead of GCD for fast synchronization across processes.
  #if os(Windows)
  private var _mutex: SRWLOCK
  #else
  private var _mutex: pthread_mutex_t
  #endif
  
  init() {
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxBatchesInFlight)!
    self.preferSharedStorage = device.hasUnifiedMemory
    self.synchronizationFence = device.makeFence()!
    self.synchronizationEvent = device.makeEvent()!
    
    self._mutex = .init()
    #if os(Windows)
    InitializeSRWLock(&_mutex)
    #else
    pthread_mutex_init(&_mutex, nil)
    #endif
    
    ShaderCache.load(device: device)
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

extension Context {
  @inline(__always)
  func acquireMutex() {
    #if os(Windows)
    AcquireSRWLockExclusive(&_mutex)
    #else
    let code = pthread_mutex_lock(&_mutex)
    precondition(code == 0, "Attempt to acquire mutex returned '\(code)'.")
    #endif
  }
  
  @inline(__always)
  func releaseMutex() {
    #if os(Windows)
    ReleaseSRWLockExclusive(&_mutex)
    #else
    let code = pthread_mutex_unlock(&_mutex)
    precondition(code == 0, "Attempt to release mutex returned '\(code)'.")
    #endif
  }
  
  // Borrowed from https://github.com/s4tf/s4tf
  @inline(__always)
  func sync<Result>(execute body: () throws -> Result) rethrows -> Result {
    acquireMutex()
    defer {
      releaseMutex()
    }
    return try body()
  }
}
