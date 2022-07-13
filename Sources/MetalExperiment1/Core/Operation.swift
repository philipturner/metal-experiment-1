//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import MetalPerformanceShadersGraph

// Two levels of IR. One is higher-level, eagerly submitted. This other is optimized, with IDs
// filled in with allocations ready to be materialized.

enum UnaryOperationType: UInt16, CaseIterable {
  case increment
}

enum EagerOperation {
  struct Unary {
    var type: UnaryOperationType
    var input: UInt64
    var output: UInt64
    var size: Int
  }
  
  case unary(Unary)
}

// Instead of keeping references to the individual buffers, this keeps references to the
// compiled operations until finishing. That's a simpler way to manage the memory.
enum CompiledOperation {
  struct MultiUnary {
    // Change this to a possible SIMD vector to prevent allocating an array.
    var types: [UnaryOperationType]
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  
  case multiUnary(MultiUnary)
}

extension Context {
//  func encodeEagerOperation(
//    _ operation: EagerOperation,
//    into encoder: MTLComputeCommandEncoder
//  ) {
//    switch operation {
//    case .unary(let unary):
//      encodeSingleUnary(unary, into: encoder)
//    }
//  }
  
  func encodeCompiledOperation(
    _ operation: CompiledOperation,
    into encoder: MTLComputeCommandEncoder
  ) throws {
    switch operation {
    case .multiUnary(let multiUnary):
      try encodeMultiUnary(multiUnary, into: encoder)
    }
  }
}

// Making these private forces them to inline into their caller.
private extension Context {
  func encodeSingleUnary(
    _ operation: EagerOperation.Unary,
    into encoder: MTLComputeCommandEncoder
  ) {
    func getBuffer(id: UInt64) -> MTLBuffer {
      let allocation = try! _unsafeFetchAllocation(id: id)!
      try! allocation.materialize()
      return allocation.mtlBuffer!
    }
    let buffer1 = getBuffer(id: operation.input)
    let buffer2 = getBuffer(id: operation.output)
    encoder.setComputePipelineState(unaryComputePipeline)
    encoder.setBuffer(buffer1, offset: 0, index: 0)
    encoder.setBuffer(buffer2, offset: 0, index: 1)
    var bytes: Float = 1
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
  
  func encodeMultiUnary(
    _ operation: CompiledOperation.MultiUnary,
    into encoder: MTLComputeCommandEncoder
  ) throws {
    try operation.input.materialize()
    try operation.output.materialize()
    encoder.setComputePipelineState(unaryComputePipeline)
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 1)
    
    var bytes = Float(operation.types.count)
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
}
