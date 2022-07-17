//
//  Interface.swift
//  
//
//  Created by Philip Turner on 7/15/22.
//

import Darwin

// MARK: - Operation Execution

extension Context {
  // The output is a buffer of interleaved (ID, rank). This eliminates the need to send a virtual
  // function call afterwards, just to ask "what was this output's rank?" For the shape or size in
  // bytes you need one virtual function call to materialize them on the frontend.
  //
  // Rules for encoding attributes:
  //
  // Atoms of data are padded to 16 bytes. For strings and arrays, encode an `UnsafeBufferPointer`
  // to their data. This rule applies recursively with arrays of strings, arrays of arrays, etc.
  // After the first level of recursion, store elements in their native layout stride.
  public static func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<(UInt64, Int)>
  ) {
    withDispatchQueue {
      Context.global._executeOperation(name, attributes, inputs, outputs)
    }
  }
  
  @inline(__always)
  private func _executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<(UInt64, Int)>
  ) {
    let string = StringWrapper(wrapping: name)
    guard let function = OperatorRegistry.registry[string] else {
      fatalError("Could not find operator '\(name)'")
    }
    function.call(attributes, inputs, outputs)
    self.maybeFlushStream()
  }
}

// MARK: - Operator Dispatch Table

struct OperatorRegistry {
  typealias FunctionSignature = @convention(c) (
    OpaquePointer?, Int, OpaquePointer?, Int, OpaquePointer?, Int) -> Void
  
  struct Arguments {
    var attributes: UnsafeRawBufferPointer
    var inputs: UnsafeBufferPointer<UInt64>
    var outputs: UnsafeMutableBufferPointer<(UInt64, Int)>
    
    @inline(__always)
    init(
      _ attributesPtr: OpaquePointer?,
      _ attributesCount: Int,
      _ inputsPtr: OpaquePointer?,
      _ inputsCount: Int,
      _ outputsPtr: OpaquePointer?,
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
      _ outputs: UnsafeMutableBufferPointer<(UInt64, Int)>
    ) {
      body(
        .init(attributes.baseAddress), attributes.count,
        .init(inputs.baseAddress), inputs.count,
        .init(outputs.baseAddress), outputs.count)
    }
  }
  
  // Pointer mutation
  
  @inline(__always)
  static func advanceAtom(_ ptr: inout UnsafeRawBufferPointer) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 16)
    let count = ptr.count - 16
    ptr = UnsafeRawBufferPointer(start: start, count: count)
  }
  
  @inline(__always)
  static func advanceInput(_ ptr: inout UnsafeBufferPointer<UInt64>) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 1)
    let count = ptr.count - 1
    ptr = UnsafeBufferPointer(start: start, count: count)
  }
  
  @inline(__always)
  static func advanceOutput(_ ptr: inout UnsafeMutableBufferPointer<(UInt64, Int)>) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 1)
    let count = ptr.count - 1
    ptr = UnsafeMutableBufferPointer(start: start, count: count)
  }
  
  // Data decoding
  
  @inline(__always)
  static func decodeScalar<T: SIMDScalar>(_ ptr: inout UnsafeRawBufferPointer) -> T {
    let value = ptr.assumingMemoryBound(to: T.self)[0]
    advanceAtom(&ptr)
    return value
  }
  
  @inline(__always)
  static func decodeString(_ ptr: inout UnsafeRawBufferPointer) -> StringWrapper {
    let wrappedValue = ptr.assumingMemoryBound(to: UnsafeRawBufferPointer.self)[0]
    advanceAtom(&ptr)
    return StringWrapper(wrapping: wrappedValue)
  }
  
  @inline(__always)
  static func decodeInput(_ ptr: inout UnsafeBufferPointer<UInt64>) -> UInt64 {
    let value = ptr[0]
    advanceInput(&ptr)
    return value
  }
  
  @inline(__always)
  static func encodeOutput(
    _ ptr: inout UnsafeMutableBufferPointer<(UInt64, Int)>,
    _ output: (UInt64, Int)) {
    ptr[0] = output
    advanceOutput(&ptr)
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
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    let ctx = Context.global
    precondition(args.inputs.count == 1)
    precondition(args.outputs.count == 1)
    
    // Fetch inputs
    let input1_id = decodeInput(&args.inputs)
    let input1_alloc = ctx._internalFetch(input1_id)
    precondition(input1_alloc.metadata.dataType == .float32)
    ctx._internalRetain(input1_alloc)
    
    // Generate outputs
    let (output1_id, output1_alloc) = ctx._internalAllocate(input1_alloc.metadata)
    ctx._internalRetain(output1_alloc)
    
    // Append operation
    let size = input1_alloc.metadata.byteCount / MemoryLayout<Float>.stride
    let operation = EagerOperation.Unary(
      type: .increment, input: input1_id, output: output1_id, size: size)
    ctx.eagerOperations.append(.unary(operation))
    
    // Return
    encodeOutput(&args.outputs, (output1_id, output1_alloc.metadata.rank))
  }
}
