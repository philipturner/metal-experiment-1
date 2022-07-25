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
  var justFinishedBarrier = true
  // If just finished barrier, spin off a timer that force-flushes the stream. If a command buffer
  // was committed since then, abort the timer.
  
  static let maxBatchesInFlight = 10
  static var maxCommandsPerBatch = 128
  static var maxCommandsPerSmallBatch = 16
  var numCommittedBatches: UnsafeAtomic<Int> = .create(0)
  var numScheduledBatches: UnsafeAtomic<Int> = .create(0)
  var numCompletedBatches: UnsafeAtomic<Int> = .create(0)
  var eagerOperations: [EagerOperation] = []
  
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
    
    self._mutex = .init()
    #if os(Windows)
    InitializeSRWLock(&_mutex)
    #else
    pthread_mutex_init(&_mutex, nil)
    #endif
    
    // Loads all commonly used shaders. Operations that reference these SHOULD NOT call
    // `enqueue(_:)` on the shader cache, because that's redundant and wastes clock cycles. I don't
    // know why anything would call `enqueue(_:)`, but it's there for if something needs to.
    ShaderCache.load(device: device)
  }
  
  deinit {
    numCommittedBatches.destroy()
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
