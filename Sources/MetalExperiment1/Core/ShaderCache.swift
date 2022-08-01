//
//  ShaderCache.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import Metal

class ShaderCache {
  var dispatchQueue = DispatchQueue(label: "com.s4tf.metal.ShaderCache.dispatchQueue")
  var semaphores: [StringWrapper: DispatchSemaphore] = [:]
  var pipelines: [StringWrapper: MTLComputePipelineState] = [:]
  
  private var device: MTLDevice
  private var defaultLibrary: MTLLibrary?
  private var shaderSourceDirectory: URL
  
  lazy var elementwise_f32_i32 = wait(name: "elementwise_f32_i32")
  lazy var elementwise_u32_i64_u64 = wait(name: "elementwise_u32_i64_u64")
  
  // Called during `Context.init`. Since the encapsulating context is currently initializing, it
  // can't access the device via `Context.global.device`.
  init(mtlDevice: MTLDevice) {
    self.device = mtlDevice
    self.defaultLibrary = try? device.makeDefaultLibrary(bundle: .module)
    self.shaderSourceDirectory = Bundle.module.resourceURL!
    
    // In SwiftPM builds, the bundle does not include a Metal library. You have to compile shaders
    // at runtime, although this only incurs a measurable cost once. On the second run through
    // SwiftPM tests, it loads just as fast as Xcode with the pre-compiled Metal library.
    
    // Load frequently used shaders immediately. Rarely used ones finish loading asynchronously. It
    // takes longer to compile shaders on the first load if you use multiple threads. I guess the
    // system has a global lock on its Metal compiler, which degrades performance.
    enqueue(name: "elementwise_f32_i32", asynchronous: false)
    enqueue(name: "elementwise_u32_i64_u64", asynchronous: true)
  }
  
  @inline(__always)
  func enqueue(name: StringWrapper, asynchronous: Bool = true) {
    if semaphores[name] != nil {
      // Already enqueued.
    } else {
      actuallyEnqueue(name: name, asynchronous: asynchronous)
    }
  }
  
  @inline(never)
  func actuallyEnqueue(name: StringWrapper, asynchronous: Bool) {
    let semaphore = DispatchSemaphore(value: 0)
    semaphores[name] = semaphore
    let closure = { [self] in
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
  
  func wait(name: StringWrapper) -> MTLComputePipelineState {
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


