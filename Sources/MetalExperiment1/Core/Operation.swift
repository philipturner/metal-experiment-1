//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

enum UnaryOperationType: UInt8, CaseIterable {
  case increment
}

enum EagerOperation {
  struct ExplicitCopy {
    var input: UInt64
    var output: UInt64
  }
  case explicitCopy(ExplicitCopy)
  
  struct Unary {
    var type: UnaryOperationType
    var input: UInt64
    var output: UInt64
  }
  case unary(Unary)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct ExplicitCopy {
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  case explicitCopy(ExplicitCopy)
  
  struct MultiUnary {
    // `dataTypes` has half the vector capacity of `types`. It doesn't need as much storage because
    // it's serialized efficiently. A new type is only recorded after each cast operation. When
    // encoding Metal commands, both lists expand to 2 bytes/element, mapping one-to-one with shader
    // loop iterations.
    var types: OperationTypeList16<UnaryOperationType>
    var dataTypes: OperationTypeList4<DataType>
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  case multiUnary(MultiUnary)
}

// MARK: - Metal Compute Command Encoding

struct EncodingContext {
  let commandBuffer: MTLCommandBuffer
  let commandBufferID: Int
  private var encoder: MTLComputeCommandEncoder?
  private unowned var state: MTLComputePipelineState?
  
  init(commandBuffer: MTLCommandBuffer, commandBufferID: Int) {
    self.commandBuffer = commandBuffer
    self.commandBufferID = commandBufferID
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
    case .explicitCopy(let explicitCopy):
      // TODO: A debug print after making the explicit copy operation, proving the function is being
      // called.
      fatalError("Explicit copy not yet implemented.")
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
    operation.output.lastModifiedCommandBufferID = ectx.commandBufferID
    
    let encoder = ectx.makeEncoder()
    ectx.setComputePipelineState(unaryComputePipeline)
    
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(operation.output.mtlBuffer!, offset: 0, index: 1)
    
    var bytes = Float(operation.types.count)
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
}
