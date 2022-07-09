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
  
  init() {
    Profiler.checkpoint()
    self.device = MTLCreateSystemDefaultDevice()!
    self.commandQueue = device.makeCommandQueue()!
    Profiler.log("device and command queue")
    
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
    
  
//    print(bundlePath)
    
//    let library = try! device.makeDefaultLibrary(bundle: .module)
//    let function = library.makeFunction(name: "addition")!
//    computePipeline = try! device.makeComputePipelineState(function: function)
  }
}
