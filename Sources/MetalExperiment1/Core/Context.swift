//
//  Context.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Atomics
import Metal

class Context {
  static var global = Context()
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var computePipeline: MTLComputePipelineState
  var lastCommandBuffer: MTLCommandBuffer?
  static let maxBatchesInFlight = 10
  
  static let numBufferElements = 1000
  var buffer1: MTLBuffer // Input for next operation, current state of execution.
  var buffer2: MTLBuffer // Output for next operation.
  var operationCount = 0 // Current value of elements in `buffer1`.
  
  static var profilingEncoding = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_COMMAND_STREAM")
  
  static var maxCommandsPerBatch = 100
  var numCommittedBatches: ManagedAtomic<Int> = .init(0)
  var numScheduledBatches: ManagedAtomic<Int> = .init(0)
  var numCompletedBatches: ManagedAtomic<Int> = .init(0)
  var bufferedOperations: [Operation] = []
  
  // Function to write to the raw CPU-side memory (zero overhead)
  // Function to automatically de-allocate
  // Does not automatically allocate
  // Lazy allocation so that placeholders in fused ops (or graph mode) can happen
  //
  // Could deallocate already be called by the user, showing which tensors disappear and
  // automatically fusing unary ops?
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
    self.computePipeline = try! device.makeComputePipelineState(function: unaryFunction)
    
    let bufferSize = Context.numBufferElements * MemoryLayout<Float>.stride
    self.buffer1 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    self.buffer2 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
  }
}
