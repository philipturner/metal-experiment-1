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
  
  // TODO: Remove these properties and place them in a test of theoretical maximum throughput.
  static let numBufferElements = 1000
  lazy var allocation1: UInt64 = generateID() // Next operation's input, current state of execution.
  lazy var allocation2: UInt64 = generateID() // Next operation's output.
  var operationCount = 0 // Current value of elements in `buffer1`.
  
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
  
  init() {
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxBatchesInFlight)!
    
    let bundleURL = Bundle.module.resourceURL!
    let shadersURL = bundleURL.appendingPathComponent("Shaders", isDirectory: true)
    let unaryURL = shadersURL.appendingPathComponent("Unary.metal", isDirectory: false)
    let unaryData = FileManager.default.contents(atPath: unaryURL.path)!
    let unaryString = String(data: unaryData, encoding: .utf8)!
    
    let unaryLibrary = try! device.makeLibrary(source: unaryString, options: nil)
    let unaryFunction = unaryLibrary.makeFunction(name: "unaryOperation")!
    self.unaryComputePipeline = try! device.makeComputePipelineState(function: unaryFunction)
  }
  
  private func generateID() -> UInt64 {
    let bufferSize = Context.numBufferElements * MemoryLayout<Float>.stride
    let id = Context.generateID(allocationSize: bufferSize)
    try! Context.initialize(id: id) { bufferPointer in
      let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
      ptr.initialize(repeating: 0.0)
    }
    return id
  }
}
