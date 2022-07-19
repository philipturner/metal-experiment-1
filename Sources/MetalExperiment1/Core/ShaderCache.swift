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
//
// TODO: Chunk groups of pipelines into Metal binary archives, reducing I/O overhead. The current
// implementation has one pipeline per archive. This optimization is low-priority because in most
// cases, you use Xcode to compile, which uses the system shader cache. If there are noticeable
// bottlenecks in SwiftPM builds, this optimization might be worthwhile.
enum ShaderCache {
  static let dispatchQueue = DispatchQueue(label: "com.s4tf.metal.ShaderCache.dispatchQueue")
  static var semaphores: [StringWrapper: DispatchSemaphore] = [:]
  static var pipelines: [StringWrapper: MTLComputePipelineState] = [:]
  
  private static var device: MTLDevice!
  private static var defaultLibrary: MTLLibrary? // Uses the system shader cache.
  private static var shaderSourceDirectory: URL = Bundle.module.resourceURL!
  private static var binaryArchiveDirectory: URL = shaderSourceDirectory
    .appendingPathComponent("Archives", isDirectory: true)
  
  // Called during `Context.init`. Since the global context is currently initializing, it can't
  // access the device via `Context.global.device`.
  static func load(device: MTLDevice) {
    Self.device = device
//    Self.defaultLibrary = try? device.makeDefaultLibrary(bundle: .module)
    
    if defaultLibrary == nil {
      try! FileManager.default.createDirectory(
        at: binaryArchiveDirectory, withIntermediateDirectories: false)
    }
    
    // Loading custom shaders may be multithreaded (profile to determine whether this is faster).
    // Only pre-load the commonly used ones; rarely used ones go to a background queue as soon as an
    // eager operation is encoded.
    Profiler.checkpoint()
    enqueue(name: "unary_f32_i32")
    enqueue(name: "unary_u32_i64_u64")
    enqueue(name: "binary")
    enqueue(name: "tertiary")
    _ = (unary_f32_i32)
    _ = (unary_u32_i64_u64)
    _ = (binary)
    _ = (tertiary)
    Profiler.log("Create pipelines")
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
    semaphores[name] = semaphore
    DispatchQueue.global().async {
      let nameString = name.makeString()
      var pipeline: MTLComputePipelineState
      func makePipeline(library: MTLLibrary) -> MTLComputePipelineState {
        let function = library.makeFunction(name: nameString)!
        return try! device.makeComputePipelineState(function: function)
      }
      
      if let defaultLibrary = defaultLibrary {
        pipeline = makePipeline(library: defaultLibrary)
      } else {
        let fileName = nameString + ".metal"
        let shaderSourcePath = shaderSourceDirectory
          .appendingPathComponent(fileName, isDirectory: false).path
        
        // `keyPath` is not the `KeyPath<Root, Value>` used for introspection!
        let keyPath = binaryArchiveDirectory
          .appendingPathComponent(fileName, isDirectory: false).path
        let fileManager = FileManager.default
        guard let shaderSourceData = fileManager.contents(atPath: shaderSourcePath) else {
          fatalError("Could not find shader source for pipeline '\(name.makeString())'.")
        }
        
        let archiveName = nameString + ".metallib"
        let archiveURL = binaryArchiveDirectory
          .appendingPathComponent(archiveName, isDirectory: false)
        let keyData: Data? = fileManager.contents(atPath: keyPath)
        if shaderSourceData != keyData {
          // The `unicode` encoding lets symbols like "µ" appear in the Metal shader. It takes
          // longer to process than homogeneous UTF-8, but removes the chance of creating bugs in
          // the future. Swift source files are encoded in `unicode`, so a grapheme cluster might
          // leak if someone transfers comments from Swift to Metal files. This would be especially
          // problematic since it only appears in the SwiftPM build, but goes under the radar when
          // Xcode (the dominant use case) is used to compile it.
          let shaderSource = String(data: shaderSourceData, encoding: .unicode)!
          let library = try! device.makeLibrary(source: shaderSource, options: nil)
          pipeline = makePipeline(library: library)
          
          // Atomically swap the binary archive. Remove its key (the shader source) to invalidate
          // it, then replace the ".metallib", then write the new key.
          try? fileManager.removeItem(atPath: keyPath)
        } else {
          let library = try! device.makeLibrary(URL: archiveURL)
        }
        
        //
        
        //
        _ = device.makeLibrary(URL:)
        fatalError("SwiftPM shader loading not yet implemented.")
      }
      
      dispatchQueue.sync {
        pipelines[name] = pipeline
      }
      semaphore.signal()
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
