//
//  Interface.swift
//  
//
//  Created by Philip Turner on 7/15/22.
//

import Darwin

extension Context {
  public static func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<UInt64>
  ) {
    _dispatchQueue.sync {
      Context.global._executeOperation(name, attributes, inputs, outputs)
    }
  }
  
  // Accepted attribute keys: UnsafePointer<Int8>
  // Accepted attribute values: <= Int64, <= Double, UnsafePointer<Int8>
  // Encode the raw 8 bytes of everything in the buffer, the backend will decide its type based on
  // the string key. All pointers are padded to 8 bytes.
  //
  // Serialized into a buffer like OpenCL's properties arrays. Instead of being terminated by a zero
  // key, it uses the "count" property of a Swift buffer pointer. The frontend should use
  // `StaticString.utf8Start` for strings, but doesn't have to.
  
  // Atoms of data are padded to 16 bytes. For strings, encode the count followed by the raw C
  // pointer. For arrays, encode the count followed by the pointer to its underlying data. This
  // rule applies recursively. After the second level of recursion, store elements in their native
  // layout size.
  
  // Atoms of data are padded to 8 bytes. For strings, encode the raw C pointer. Its length will be
  // decoded using `strlen`. For arrays, encode a pointer to a second area of memory, which contains
  // first the count and then the data pointer. For arrays of strings, the elements must be tuples
  // of (count, C-style pointer).
  //
  // The rules above apply recursively, with one caveat. Upon entering the second level of
  // recursion, the only data types are counts and pointers. Store these in the native integer
  // width, instead of 64-bit. This convention improves performance and ease of implementation.
  //
  // This encoding allows for near zero-cost encoding on the frontend. It can be accomplished with
  // generics and `withUnsafeTemporaryBuffer`. Using 8 bytes instead of 16
  
  // The reason for this encoding is --- so it can be synthesized with Swift generics and a Swift
  // temporary buffer.
  private func _executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<UInt64>
  ) {
    precondition(StringWrapper(wrapping: name) == StringWrapper("increment"))
  }
}

// MARK: - Operator Dispatch Table

struct OperatorRegistry {
  typealias FunctionSignature = @convention(c) (
    OpaquePointer, Int, OpaquePointer, Int, OpaquePointer, Int) -> Void
  
  struct Arguments {
    var attributes: UnsafeRawBufferPointer
    var inputs: UnsafeBufferPointer<UInt64>
    var outputs: UnsafeMutableBufferPointer<UInt64>
    
    @inline(__always)
    init(
      _ attributesPtr: OpaquePointer,
      _ attributesCount: Int,
      _ inputsPtr: OpaquePointer,
      _ inputsCount: Int,
      _ outputsPtr: OpaquePointer,
      _ outputsCount: Int
    ) {
      attributes = .init(start: .init(attributesPtr), count: attributesCount)
      inputs = .init(start: .init(inputsPtr), count: inputsCount)
      outputs = .init(start: .init(outputsPtr), count: outputsCount)
    }
  }
  
  struct Function {
    var body: FunctionSignature
    
    @inline(__always)
    init(_ body: @escaping FunctionSignature) {
      self.body = body
    }
    
    @inline(__always)
    func call(
      _ attributes: UnsafeRawBufferPointer,
      _ inputs: UnsafeBufferPointer<UInt64>,
      _ outputs: UnsafeMutableBufferPointer<UInt64>
    ) {
      body(
        .init(attributes.baseAddress.unsafelyUnwrapped), attributes.count,
        .init(inputs.baseAddress.unsafelyUnwrapped), inputs.count,
        .init(outputs.baseAddress.unsafelyUnwrapped), outputs.count)
    }
    
    @inline(__always)
    static func advanceScalar(_ ptr: inout UnsafeRawBufferPointer) {
      let baseAddress = ptr.baseAddress.unsafelyUnwrapped
      let start = baseAddress.advanced(by: 8)
      let count = ptr.count - 8
      ptr = UnsafeRawBufferPointer(start: start, count: count)
    }
    
    @inline(__always)
    static func advanceString(_ ptr: inout UnsafeRawBufferPointer) {
      let baseAddress = ptr.baseAddress.unsafelyUnwrapped
      let start = baseAddress.advanced(by: 16)
      let count = ptr.count - 16
      ptr = UnsafeRawBufferPointer(start: start, count: count)
    }
    
    @inline(__always)
    static func advanceUInt64(_ ptr: inout UnsafeBufferPointer<UInt64>) {
      let baseAddress = ptr.baseAddress.unsafelyUnwrapped
      let start = baseAddress.advanced(by: 1)
      let count = ptr.count - 1
      ptr = UnsafeBufferPointer<UInt64>(start: start, count: count)
    }
    
    @inline(__always)
    static func advanceUInt64(_ ptr: inout UnsafeMutableBufferPointer<UInt64>) {
      let baseAddress = ptr.baseAddress.unsafelyUnwrapped
      let start = baseAddress.advanced(by: 1)
      let count = ptr.count - 1
      ptr = UnsafeMutableBufferPointer<UInt64>(start: start, count: count)
    }
    
    @inline(__always)
    static func acceptScalar<T: SIMDScalar>(_ ptr: inout UnsafeRawBufferPointer) -> T {
      let value = ptr.assumingMemoryBound(to: T.self)[0]
      advanceScalar(&ptr)
      return value
    }
    
    @inline(__always)
    static func acceptString(_ ptr: inout UnsafeRawBufferPointer) -> StringWrapper {
      let cString = ptr.assumingMemoryBound(to: UnsafeRawPointer.self)[0]
      
      // Performance note: function call to `strlen`.
      let count = strlen(cString.assumingMemoryBound(to: CChar.self))
      let value = StringWrapper(wrapping: .init(start: cString, count: count))
      advanceRaw(&ptr)
      return value
    }
    
    @inline(__always)
    static func acceptInput(_ ptr: inout UnsafeBufferPointer<UInt64>) -> UInt64 {
      let value = ptr[0]
      advanceUInt64(&ptr)
      return value
    }
    
    @inline(__always)
    static func returnOutput(_ ptr: inout UnsafeMutableBufferPointer<UInt64>, _ output: UInt64) {
      ptr[0] = output
      advanceUInt64(&ptr)
    }
  }
}

extension OperatorRegistry {
  static let registry: [StringWrapper: Function] = [
    "increment": increment
  ]
}

// MARK: - Operator Functions

extension OperatorRegistry {
  static let increment = Function {
    let args = Arguments($0, $1, $2, $3, $4 ,$5)
    // Encode the eager operator.
    
    precondition(inputs.count == 1)
    precondition(outputs.count == 1)
    
    // TODO: Optimize this by only fetching from the dictionary once. Use the retrieved allocation
    // for both extracting size and incrementing its reference count. Also, make a force-inlined
    // internal function for generating IDs, which returns the newly generated `Allocation` without
    // requiring that it be fetched from a dictionary later on.
    let input1 = inputs[0]
    let allocationSize = _compilerFetchAllocation(id: input1).size
    let output1 = _compilerGenerateID(allocationSize: allocationSize)
    let size = allocationSize / MemoryLayout<Float>.stride
//    commitIncrement(inputID: input1, outputID: output1, size: size)
    
    outputs[0] = output1 // return
  }
}
