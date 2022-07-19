//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import MetalPerformanceShaders

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
      self.state = nil
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
    case .explicitCopy(let explicitCopy):
      try encodeExplicitCopy(explicitCopy, into: &ectx)
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
    ectx.setComputePipelineState(ShaderCache.unary_f32_i32)
    
    encoder.setBuffer(operation.input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(operation.output.mtlBuffer!, offset: 0, index: 1)
    
    var bytes = Float(operation.types.count)
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(operation.size), threadsPerThreadgroup: 1)
  }
  
  func encodeExplicitCopy(
    _ operation: CompiledOperation.ExplicitCopy,
    into ectx: inout EncodingContext
  ) throws {
    try operation.input.materialize()
    try operation.output.materialize()
    operation.output.lastModifiedCommandBufferID = ectx.commandBufferID
    
    // Use a blit encoder because this command's dominant use case is marshalling data over PCIe. I
    // don't know the performance characteristics of PCIe, so it's best to delegate that to Apple's
    // optimized blit commands. If it were just copying between GPU memory regions, a specialized
    // compute shader would be just as fast and have less CPU-side overhead.
    //
    // I'm also not (yet) making optimized encoding utilities like `makeBlitEncoder` and
    // `finishBlitEncoder` just for this command. There is a very low chance that two explicit copy
    // operations appear next to each other. This is a viable optimization that I could pursue down
    // the road, alongside other enhancements to reading data from the GPU.
    ectx.finishEncoder()
    let blitEncoder = ectx.commandBuffer.makeBlitCommandEncoder()!
    defer {
      blitEncoder.endEncoding()
    }
    
    blitEncoder.copy(
      from: operation.input.mtlBuffer!, sourceOffset: 0, to: operation.output.mtlBuffer!,
      destinationOffset: 0, size: operation.byteCount)
  }
}
