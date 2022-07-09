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
  static let maxCommandBuffers = 5
  
  static let bufferNumElements = 10
  var buffer1: MTLBuffer // Input for next operation, current state of execution.
  var buffer2: MTLBuffer // Output for next operation.
  var operationCount = 0 // Current value of elements in `buffer1`.
  
  static var maxCommandsPerCmdbuf = 100
  var currentCommandBuffer: MTLCommandBuffer?
  var currentComputeEncoder: MTLComputeCommandEncoder?
  var numEncodedCommands = 0
  var committedCmdbufCount = 0
  var completedCmdbufCount: ManagedAtomic<Int> = .init(0)
  
  init() {
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxCommandBuffers / 1)!
    
    let bundleURL = Bundle.module.resourceURL!
    let shadersURL = bundleURL.appendingPathComponent("Shaders", isDirectory: true)
    let unaryURL = shadersURL.appendingPathComponent("Unary.metal", isDirectory: false)
    let unaryData = FileManager.default.contents(atPath: unaryURL.path)!
    let unaryString = String(data: unaryData, encoding: .utf8)!
    
    let unaryLibrary = try! device.makeLibrary(source: unaryString, options: nil)
    let unaryFunction = unaryLibrary.makeFunction(name: "unaryOperation")!
    self.computePipeline = try! device.makeComputePipelineState(function: unaryFunction)
    
    let bufferSize = Context.bufferNumElements * MemoryLayout<Float>.stride
    self.buffer1 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    self.buffer2 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    
  }
}
