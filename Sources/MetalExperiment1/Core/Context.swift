//
//  Context.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Metal

class Context {
  static var global = Context()
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var computePipeline: MTLComputePipelineState
  var lastCommandBuffer: MTLCommandBuffer?
  
  init() {
    Profiler.checkpoint()
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue()!
    Profiler.log("initialize Metal runtime")
    
    Profiler.checkpoint()
    let bundleURL = Bundle.module.resourceURL!
    let shadersURL = bundleURL.appendingPathComponent("Shaders", isDirectory: true)
    let unaryURL = shadersURL.appendingPathComponent("Unary.metal", isDirectory: false)
    let unaryData = FileManager.default.contents(atPath: unaryURL.path)!
    let unaryString = String(data: unaryData, encoding: .utf8)!
    Profiler.log("load shader from disk")
    
    Profiler.checkpoint()
    let unaryLibrary = try! device.makeLibrary(source: unaryString, options: nil)
    let unaryFunction = unaryLibrary.makeFunction(name: "unaryOperation")!
    self.computePipeline = try! device.makeComputePipelineState(function: unaryFunction)
    Profiler.log("create shader object")
  }
  
  func commitEmptyCommandBuffer() {
    let commandBuffer = commandQueue.makeCommandBuffer()!
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
      print("Kernel time: \(kernelTimeMessage)")
      print("GPU time: \(gpuTimeMessage)")
    }
  }
}
