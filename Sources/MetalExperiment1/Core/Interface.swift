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
  
  // Rule for encoding attributes:
  //
  // Atoms of data are padded to 16 bytes. For strings, encode the count, then the raw C pointer.
  // For arrays, encode the count, then the pointer to its underlying data. This rule applies
  // recursively. After the first level of recursion, store elements in their native layout size.
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
    
    // Pointer advancing
    
    @inline(__always)
    static func advanceAtom(_ ptr: inout UnsafeRawBufferPointer) {
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
    
    // Data decoding
    
    @inline(__always)
    static func acceptScalar<T: SIMDScalar>(_ ptr: inout UnsafeRawBufferPointer) -> T {
      let value = ptr.assumingMemoryBound(to: T.self)[0]
      advanceAtom(&ptr)
      return value
    }
    
    @inline(__always)
    static func acceptString(_ ptr: inout UnsafeRawBufferPointer) -> StringWrapper {
      let count = ptr.assumingMemoryBound(to: Int.self)[0]
      advanceAtom(&ptr)
      let start = ptr.assumingMemoryBound(to: UnsafeRawPointer.self)[0]
      advanceAtom(&ptr)
      return StringWrapper(wrapping: UnsafeRawBufferPointer(start: start, count: count))
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
