//
//  ShaderCache.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import Metal

// Operates in two modes. The first is "automatic", where it queries the system shader cache. This
// works if you're in Xcode or on iOS. The second is a fallback mode, used for command-line builds
// with SwiftPM.
enum ShaderCache {
  static let dispatchQueue = DispatchQueue(label: "com.s4tf.metal.ShaderCache.dispatchQueue")
  static var semaphores: [StringWrapper: DispatchSemaphore] = [:]
  static var pipelines: [StringWrapper: MTLComputePipelineState] = [:]
  
  private static var device: MTLDevice!
  private static var defaultLibrary: MTLLibrary? // Uses the system shader cache.
  
  // Called during `Context.init`. Since the global context is currently initializing, it can't
  // access the device via `Context.global.device`.
  static func load(device: MTLDevice) {
    Self.device = device
    Self.defaultLibrary = try? device.makeDefaultLibrary(bundle: .module)
//    precondition(defaultLibrary != nil)
    
    // Loading custom shaders may be multithreaded. Only pre-load the commonly used ones; rarely
    // used ones go to a background queue as soon as an eager operation is encoded.
  }
  
  @inline(__always)
  static func enqueue(name: StringWrapper) {
    if semaphores[name] != nil {
      // Already enqueued.
    } else {
      actuallyEnqueue(name: name)
    }
  }
  
  @inline(never)
  static func actuallyEnqueue(name: StringWrapper) {
    let semaphore = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
      // ???
      semaphore.signal()
    }
    semaphores[name] = semaphore
  }
  
  static func wait(name: StringWrapper) -> MTLComputePipelineState {
    // Catch shader loading bugs.
    guard let semaphore = semaphores[name] else {
      fatalError("Waited on operation \(name.makeString()) before it was enqueued.")
    }
    semaphore.wait()
    return ShaderCache.dispatchQueue.sync {
      pipelines[name]!
    }
  }
}

extension ShaderCache {
  static let unary_f32_i32 = Self.wait(name: "unary_f32_i32")
  static let unary_u32_i64_u64 = Self.wait(name: "unary_u32_i64_u64")
  static let binary = Self.wait(name: "binary")
  static let tertiary = Self.wait(name: "tertiary")
}
