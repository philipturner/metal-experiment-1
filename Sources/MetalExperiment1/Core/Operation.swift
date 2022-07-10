//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

enum Operation {
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

extension Context {
  func encodeSingleOperation(_ operation: Operation, into encoder: MTLComputeCommandEncoder) {
    switch operation {
    case .unary(let unary):
      encodeSingleUnary(unary, into: encoder)
    }
  }
  
  func encodeSingleUnary(_ operation: Operation.Unary, into encoder: MTLComputeCommandEncoder) {
    encoder.setComputePipelineState(computePipeline)
    encoder.setBuffer(operation.input, offset: 0, index: 0)
    encoder.setBuffer(operation.output, offset: 0, index: 1)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
}
