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
  struct Unary {
    var type: UnaryOperationType
    var input: UInt64
    var output: UInt64
    var size: Int
  }
  
  case unary(Unary)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct MultiUnary {
    var types: OperationTypeList16<UnaryOperationType>
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
      
      let device = Context.global.device
      if let fence = Context.global.fence {
        encoder.waitForFence(fence)
      }
      Context.global.fence = device.makeFence()!
      
      return encoder
    }
  }
  
  @inline(__always)
  mutating func finishEncoder() {
    if let encoder = encoder {
      encoder.updateFence(Context.global.fence!)
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
//    operation.output.lastModifiedCommandBufferID = ectx.commandBufferID
    
    let encoder = ectx.makeEncoder()
    ectx.setComputePipelineState(unaryComputePipeline)
    
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(operation.output.mtlBuffer!, offset: 0, index: 1)
    
    var bytes: Float = 1
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
}
