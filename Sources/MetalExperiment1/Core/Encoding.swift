//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  static func validate() {
    withDispatchQueue {
      let ctx = Context.global
      ctx._compilerBarrier()
      try! Context.read(id: ctx.allocation1) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        precondition(ptr[0] == Float(ctx.operationCount))
      }
    }
  }
  
  static func commitStreamedCommand() {
    withDispatchQueue {
      Context.global.commitStreamedCommand()
    }
  }
  
  public static func commitIncrement(inputID: UInt64, outputID: UInt64) {
    withDispatchQueue {
      Context.global.commitIncrement(inputID: inputID, outputID: outputID)
    }
  }
  
  static func barrier() {
    withDispatchQueue {
      Context.global._compilerBarrier()
    }
  }
  
  @inline(__always)
  internal func _compilerBarrier(commandBufferID chosenID: Int? = nil) {
    flushStream()
    barrier(commandBufferID: chosenID)
  }
}

private extension Context {
  @inline(__always)
  func queryQueueBackPressure() -> Int {
    let numCommitted = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    let numScheduled = numScheduledBatches.load(ordering: .sequentiallyConsistent)
    return numCommitted - numScheduled
  }
  
  @inline(__always)
  func barrier(commandBufferID chosenID: Int? = nil) {
  }
}

// Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
// allocation happens afterwards, during `flushStream`.
private extension Context {
  func commitStreamedCommand() {
    let inputID = allocation1
    let outputID = allocation2
    operationCount += 1
    swap(&allocation1, &allocation2)
    commitIncrement(inputID: inputID, outputID: outputID)
  }
  
  func commitIncrement(inputID: UInt64, outputID: UInt64) {
    _compilerRetain(id: inputID)
    _compilerRetain(id: outputID)
    let operation = EagerOperation.Unary(
      type: .increment, input: inputID, output: outputID, size: Context.numBufferElements)
    eagerOperations.append(.unary(operation))
    
    let backPressure = queryQueueBackPressure()
    flushStream(precomputedBackPressure: backPressure)
  }
  
  func flushStream(precomputedBackPressure: Int? = nil) {
    let numEagerOperations = eagerOperations.count
    guard numEagerOperations > 0 else {
      return
    }
    

    let compiledOperations = compileEagerOperations()

    
    // Using relaxed memory ordering because it doesn't need to be atomic in the first place. It is
    // never modified by threads other than the encapsulating dispatch queue.
    var commandBufferID = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    var encodingContext = EncodingContext(
      commandBuffer: commandQueue.makeCommandBuffer()!, commandBufferID: commandBufferID)
    
    precondition(compiledOperations.count == 1)
    try! encodeCompiledOperation(compiledOperations[0], into: &encodingContext)
    encodingContext.finishEncoder()
    
    // Force the memory allocations to stay alive until the command buffer finishes.
    let retainClosure = {
      _ = compiledOperations
    }
    
    // Instead of `wrappingIncrement(ordering:)`, this code section uses `store(_:ordering:)`. It
    // always executes on the same thread, so it's okay to increment in a way that isn't perfectly
    // atomic.
    commandBufferID += 1
    numCommittedBatches.store(commandBufferID, ordering: .sequentiallyConsistent)
    
    encodingContext.commandBuffer.addScheduledHandler { selfRef in
      precondition(selfRef.status == .scheduled)
      self.numScheduledBatches.wrappingIncrement(ordering: .sequentiallyConsistent)
    }
    
    encodingContext.commandBuffer.addCompletedHandler { selfRef in
      precondition(selfRef.status == .completed)
      let numCommitted = self.numCommittedBatches.load(ordering: .sequentiallyConsistent)
      let numCompleted = self.numCompletedBatches.wrappingIncrementThenLoad(
        ordering: .sequentiallyConsistent)
      
      Context._dispatchQueue.async(execute: retainClosure)
    }
    encodingContext.commandBuffer.commit()
  }
}
