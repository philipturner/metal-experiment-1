//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  public static func commitIncrement(inputID: UInt64, outputID: UInt64) {
    withDispatchQueue {
      Context.global.commitIncrement(inputID: inputID, outputID: outputID)
    }
  }
}

// Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
// allocation happens afterwards, during `flushStream`.
private extension Context {
  func commitIncrement(inputID: UInt64, outputID: UInt64) {
    _compilerRetain(id: inputID)
    _compilerRetain(id: outputID)
    let unary = EagerOperation.Unary(
      type: .increment, input: inputID, output: outputID, size: Context.numBufferElements)
    
    var compiledOperation: CompiledOperation
    let input = _compilerFetchAllocation(id: unary.input)
    let output = _compilerFetchAllocation(id: unary.output)
    _compilerRelease(input)
    _compilerRelease(output)
    
//    let multiUnary = CompiledOperation.MultiUnary(
//      input: input, output: output, size: unary.size)
//    compiledOperation = .multiUnary(multiUnary)

    let cmdbuf = commandQueue.makeCommandBuffer()!
    let encoder = cmdbuf.makeComputeCommandEncoder()!
    
    try! input.materialize()
    try! output.materialize()
    
    encoder.setComputePipelineState(unaryComputePipeline)
    encoder.setBuffer(input.mtlBuffer!, offset: 0, index: 0)
    encoder.setBuffer(output.mtlBuffer!, offset: 0, index: 1)
    
    var bytes: Float = 1
    encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
    encoder.dispatchThreadgroups(.init(1), threadsPerThreadgroup: 1)
    encoder.endEncoding()
    
    let retainClosure = {
      _ = input
      _ = output
    }

    cmdbuf.addCompletedHandler { selfRef in
      precondition(selfRef.status == .completed)
      Context._dispatchQueue.async(execute: retainClosure)
    }
    cmdbuf.commit()
  }
}
