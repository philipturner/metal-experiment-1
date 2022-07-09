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
  static let maxCommandBuffers = 10
  
  static let bufferNumElements = 1000
  var buffer1: MTLBuffer // Input for next operation, current state of execution.
  var buffer2: MTLBuffer // Output for next operation.
  var operationCount = 0 // Current value of elements in `buffer1`.
  
  var atomics: [ManagedAtomic<Int>]
  var atomicIndex = 0
  func cycleAtomics() -> ManagedAtomic<Int> {
    precondition(atomics.count == Context.maxCommandBuffers)
    let output = atomics[atomicIndex]
    atomicIndex += 1
    if atomicIndex >= atomics.count {
      atomicIndex = 0
    }
    return output
  }
  
  var events: [MTLSharedEvent]
  var eventIndex = 0
  func cycleEvents() -> MTLSharedEvent {
    precondition(events.count == Context.maxCommandBuffers)
    let output = events[eventIndex]
    eventIndex += 1
    if eventIndex >= events.count {
      eventIndex = 0
    }
    return output
  }
  
  init() {
    Profiler.checkpoint()
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: Context.maxCommandBuffers)!
    Profiler.log("Initialize Metal runtime")
    
    Profiler.checkpoint()
    let bundleURL = Bundle.module.resourceURL!
    let shadersURL = bundleURL.appendingPathComponent("Shaders", isDirectory: true)
    let unaryURL = shadersURL.appendingPathComponent("Unary.metal", isDirectory: false)
    let unaryData = FileManager.default.contents(atPath: unaryURL.path)!
    let unaryString = String(data: unaryData, encoding: .utf8)!
    Profiler.log("Load shader from disk")
    
    Profiler.checkpoint()
    let unaryLibrary = try! device.makeLibrary(source: unaryString, options: nil)
    let unaryFunction = unaryLibrary.makeFunction(name: "unaryOperation")!
    self.computePipeline = try! device.makeComputePipelineState(function: unaryFunction)
    Profiler.log("Create shader object")
    
    Profiler.checkpoint()
    let bufferSize = Context.bufferNumElements * MemoryLayout<Float>.stride
    self.buffer1 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    self.buffer2 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    Profiler.log("Create buffers")
    
    Profiler.checkpoint()
    _ = ManagedAtomic<Int>.self
    Profiler.log("Load Atomics framework")
    
    Profiler.checkpoint()
    self.atomics = []
    for _ in 0..<Context.maxCommandBuffers {
      atomics.append(ManagedAtomic<Int>(0))
    }
    
    self.events = []
    for _ in 0..<Context.maxCommandBuffers {
      events.append(device.makeSharedEvent()!)
    }
    Profiler.log("Create synchronization objects")
  }
  
  func commitEmptyCommandBuffer() {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeEncoder.endEncoding()
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
  }
  
  func barrier(showingStats: Bool = false) {
    var kernelTimeMessage: String
    var gpuTimeMessage: String
    if let commandBuffer = lastCommandBuffer {
      commandBuffer.waitUntilCompleted()
      let kernelTime = commandBuffer.kernelEndTime - commandBuffer.kernelStartTime
      let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
      kernelTimeMessage = "\(Int(kernelTime * 1e6))"
      gpuTimeMessage = "\(Int(gpuTime * 1e6))"
    } else {
      kernelTimeMessage = "n/a"
      gpuTimeMessage = "n/a"
    }
    if showingStats {
      print("Kernel time: \(kernelTimeMessage) \(Profiler.timeUnit)")
      print("GPU time: \(gpuTimeMessage) \(Profiler.timeUnit)")
    }
  }
}
