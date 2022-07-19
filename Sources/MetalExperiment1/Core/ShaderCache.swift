//
//  ShaderCache.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import Metal

enum ShaderCache {
  static let dispatchQueue = DispatchQueue(label: "com.s4tf.metal.ShaderCache.dispatchQueue")
  static var semaphores: [StringWrapper: DispatchSemaphore] = [:]
  static var pipelines: [StringWrapper: MTLComputePipelineState] = [:]
  
  private static var device: MTLDevice!
  private static var defaultLibrary: MTLLibrary?
  private static var shaderSourceDirectory: URL = Bundle.module.resourceURL!
  private static var binaryArchiveDirectory: URL = shaderSourceDirectory
    .appendingPathComponent("Archives", isDirectory: true)
  
  // Called during `Context.init`. Since the global context is currently initializing, it can't
  // access the device via `Context.global.device`.
  static func load(device: MTLDevice) {
    Self.device = device
    Self.defaultLibrary = try? device.makeDefaultLibrary(bundle: .module)
    
    // In SwiftPM builds, the bundle does not include a Metal library. You have to compile shaders
    // at runtime, although this only incurs a measurable cost once. On the second run through
    // SwiftPM tests, it loads just as fast as Xcode with the pre-compiled Metal library.
    if defaultLibrary == nil {
      try? FileManager.default.createDirectory(
        at: binaryArchiveDirectory, withIntermediateDirectories: false)
    }
    
    // Load frequently used shaders immediately. Rarely used ones finish loading asynchronously. It
    // takes longer to compile shaders on the first load if you use multiple threads. I guess the
    // system has a global lock on its Metal compiler, which degrades performance.
    enqueue(name: "unary_f32_i32", asynchronous: false)
    enqueue(name: "binary", asynchronous: false)
    DispatchQueue.global().async {
      enqueue(name: "unary_u32_i64_u64", asynchronous: false)
      enqueue(name: "tertiary", asynchronous: false)
    }
  }
  
  @inline(__always)
  static func enqueue(name: StringWrapper, asynchronous: Bool = true) {
    if semaphores[name] != nil {
      // Already enqueued.
    } else {
      actuallyEnqueue(name: name, asynchronous: asynchronous)
    }
  }
  
  @inline(never)
  static func actuallyEnqueue(name: StringWrapper, asynchronous: Bool) {
    let semaphore = DispatchSemaphore(value: 0)
    semaphores[name] = semaphore
    let closure = {
      let nameString = name.makeString()
      var library: MTLLibrary
      if let defaultLibrary = defaultLibrary {
        library = defaultLibrary
      } else {
        let shaderSourceURL = shaderSourceDirectory
          .appendingPathComponent(nameString + ".metal", isDirectory: false)
        guard let shaderSource = try? String(contentsOf: shaderSourceURL, encoding: .utf8) else {
          fatalError("Could not find shader source for pipeline '\(nameString)'.")
        }
        
        // Using a Metal binary archive doesn't affect SwiftPM build performance. The fastest and
        // simplest approach is compiling the shader from source. The system has an internal cache
        // that skips compilation if the exact same source code was JIT-compiled before.
        library = try! device.makeLibrary(source: shaderSource, options: nil)
      }
      
      let function = library.makeFunction(name: nameString)!
      let pipeline = try! device.makeComputePipelineState(function: function)
      dispatchQueue.sync {
        pipelines[name] = pipeline
      }
      semaphore.signal()
    }
    if asynchronous {
      DispatchQueue.global().async(execute: closure)
    } else {
      closure()
    }
  }
  
  static func wait(name: StringWrapper) -> MTLComputePipelineState {
    // Catch shader loading bugs.
    guard let semaphore = semaphores[name] else {
      fatalError("Waited on pipeline '\(name.makeString())' before it was enqueued.")
    }
    semaphore.wait()
    return dispatchQueue.sync {
      guard let pipeline = pipelines[name] else {
        fatalError("Could not find pipeline '\(name.makeString())'.")
      }
      // Remove pipeline from dictionary to catch shader loading bugs.
      pipelines[name] = nil
      return pipeline
    }
  }
}

extension ShaderCache {
  static let unary_f32_i32 = Self.wait(name: "unary_f32_i32")
  static let unary_u32_i64_u64 = Self.wait(name: "unary_u32_i64_u64")
  static let binary = Self.wait(name: "binary")
  static let tertiary = Self.wait(name: "tertiary")
}
