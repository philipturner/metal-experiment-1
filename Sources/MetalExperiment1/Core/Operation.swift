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

// Instead of keeping references to the individual buffers, this keeps references to the compiled
// operations until finishing. That's a simpler way to manage the memory.
enum CompiledOperation {
  struct MultiUnary {
    // TODO: Change this to a possible SIMD vector to prevent allocating an array.
    var types: [UnaryOperationType]
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  
  case multiUnary(MultiUnary)
}

// MARK: - Metal Compute Command Encoding

struct EncodingContext {
  let commandBuffer: MTLCommandBuffer
  private var encoder: MTLComputeCommandEncoder?
  private unowned var state: MTLComputePipelineState?
  
  init(commandBuffer: MTLCommandBuffer) {
    self.commandBuffer = commandBuffer
  }
  
  @inline(__always)
  mutating func makeEncoder() -> MTLComputeCommandEncoder {
    if let encoder = encoder {
      return encoder
    } else {
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      self.encoder = encoder
      return encoder
    }
  }
  
  @inline(__always)
  mutating func finishEncoder() {
    if let encoder = encoder {
      encoder.endEncoding()
      self.encoder = nil
    }
  }
  
  @inline(__always)
  mutating func setComputePipelineState(_ state: MTLComputePipelineState) {
    if self.state === state {
      // Skip function call.
    } else {
      self.state = state
      encoder!.setComputePipelineState(state)
    }
  }
}

extension Context {
  func encodeCompiledOperation(
    _ operation: CompiledOperation,
    into ectx: inout EncodingContext
  ) throws {
    switch operation {
    case .multiUnary(let multiUnary):
      try encodeMultiUnary(multiUnary, into: &ectx)
    }
  }
}

private extension Context {
  func encodeMultiUnary(
    _ operation: CompiledOperation.MultiUnary,
    into ectx: inout EncodingContext
  ) throws {
    try operation.input.materialize()
    try operation.output.materialize()
    
    let encoder = ectx.makeEncoder()
    ectx.setComputePipelineState(unaryComputePipeline)
    
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(operation.output.mtlBuffer!, offset: 0, index: 1)
    
    var bytes = Float(operation.types.count)
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
}
