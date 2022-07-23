//
//  Context.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

public class Context {
  static let global = Context()
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var commandBufferDictionary: [Int: MTLCommandBuffer] = [:]
  static let maxBatchesInFlight = 10
  
  static var profilingEncoding = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_COMMAND_STREAM")
  
  static var maxCommandsPerBatch = 128
  var numCommittedBatches: UnsafeAtomic<Int>
  var numScheduledBatches: UnsafeAtomic<Int>
  var numCompletedBatches: UnsafeAtomic<Int>
  var eagerOperations: [EagerOperation] = []
  
  var nextAllocationID: UInt64 = 0
  var numDeinitializedAllocations: UInt64 = 0
  var permitExceedingSystemRAM = false
  var preferSharedStorage: Bool
  
  init() {
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxBatchesInFlight)!
    self.preferSharedStorage = device.hasUnifiedMemory
    
    numCommittedBatches = .create(0)
    numScheduledBatches = .create(0)
    numCompletedBatches = .create(0)
    
    // Loads all commonly used shaders. Operations that reference these SHOULD NOT call
    // `enqueue(_:)` on the shader cache, because that's redundant and wastes clock cycles. I don't
    // know why anything would call `enqueue(_:)`, but it's there for if something needs to.
    ShaderCache.load(device: device)
  }
  
  deinit {
    numCommittedBatches.destroy()
    numScheduledBatches.destroy()
    numCompletedBatches.destroy()
  }
}
