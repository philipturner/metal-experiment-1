//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import MetalPerformanceShadersGraph

// Two levels of IR. One is higher-level, eagerly submitted. This other is optimized, with IDs
// filled in with allocations ready to be materialized.

enum EagerOperation {
  struct Unary {
    enum OpType: UInt16, CaseIterable {
      case increment
    }
    
    var type: OpType
    var input: MTLBuffer
    var output: MTLBuffer
    var size: Int
  }
  
  case unary(Unary)
}

enum CompiledOperation {
  struct MultiUnary {
    enum OpType: UInt16, CaseIterable {
      case increment
    }
    
    var types: [OpType]
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  
  case multiUnary(MultiUnary)
}

extension Context {
  func encodeEagerOperation(
    _ operation: EagerOperation,
    into encoder: MTLComputeCommandEncoder
  ) {
    switch operation {
    case .unary(let unary):
      encodeSingleUnary(unary, into: encoder)
    }
  }
  
  func encodeSingleUnary(
    _ operation: EagerOperation.Unary,
    into encoder: MTLComputeCommandEncoder
  ) {
    encoder.setComputePipelineState(unaryComputePipeline)
    encoder.setBuffer(operation.input, offset: 0, index: 0)
    encoder.setBuffer(operation.output, offset: 0, index: 1)
    var bytes: Float = 1
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
  
  func encodeMultiUnary(
    _ operation: CompiledOperation.MultiUnary,
    into encoder: MTLComputeCommandEncoder
  ) {
    encoder.setComputePipelineState(unaryComputePipeline)
  }
}
