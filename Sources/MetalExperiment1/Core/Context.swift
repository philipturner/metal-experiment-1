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
  var unaryComputePipeline: MTLComputePipelineState
  var commandBufferDictionary: [Int: MTLCommandBuffer] = [:]
  static let maxBatchesInFlight = 10
  
  static var profilingEncoding = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_COMMAND_STREAM")
  
  static var maxCommandsPerBatch = 128
  var numCommittedBatches: ManagedAtomic<Int> = .init(0)
  var numScheduledBatches: ManagedAtomic<Int> = .init(0)
  var numCompletedBatches: ManagedAtomic<Int> = .init(0)
  var eagerOperations: [EagerOperation] = []
  
  var allocations: [UInt64: Allocation] = [:]
  var nextAllocationID: UInt64 = 0
  var permitExceedingSystemRAM = false
  var preferSharedStorage: Bool
  
  init() {
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxBatchesInFlight)!
    self.preferSharedStorage = device.hasUnifiedMemory
    
    let bundleURL = Bundle.module.resourceURL!
    let shadersURL = bundleURL.appendingPathComponent("Shaders", isDirectory: true)
    let unaryURL = shadersURL.appendingPathComponent("Unary.metal", isDirectory: false)
    let unaryData = FileManager.default.contents(atPath: unaryURL.path)!
    let unaryString = String(data: unaryData, encoding: .utf8)!
    
    let unaryLibrary = try! device.makeLibrary(source: unaryString, options: nil)
    let unaryFunction = unaryLibrary.makeFunction(name: "unaryOperation")!
    self.unaryComputePipeline = try! device.makeComputePipelineState(function: unaryFunction)
  }
}
