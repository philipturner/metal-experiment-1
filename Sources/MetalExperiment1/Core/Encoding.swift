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
      ctx.barrier()
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
  
  static func commitIncrement(inputID: UInt64, outputID: UInt64) {
    withDispatchQueue {
      Context.global.commitIncrement(inputID: inputID, outputID: outputID)
    }
  }
  
  static func barrier() {
    withDispatchQueue {
      Context.global.barrier()
    }
  }
  
  @inline(__always)
  internal func _compilerBarrier() {
    let ctx = Context.global
    ctx.flushStream()
    if let commandBuffer = ctx.lastCommandBuffer {
      commandBuffer.waitUntilCompleted()
    }
  }
}

private extension Context {
  func barrier() {
    flushStream()
    if let commandBuffer = lastCommandBuffer {
      commandBuffer.waitUntilCompleted()
    }
  }
  
  func queryQueueBackPressure() -> Int {
    let numCommitted = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    let numScheduled = numScheduledBatches.load(ordering: .sequentiallyConsistent)
    return numCommitted - numScheduled
  }
}

private extension Context {
  // Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
  // allocation happens afterwards, during `flushStream`.
  
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
    if eagerOperations.count < Context.maxCommandsPerBatch,
       backPressure >= 1 {
      return
    }
    
    // TODO: Add a heuristic that waits a few instructions before submitting. It gets off to a very
    // slow start right after reading a buffer's contents, being unable to fuse unary operators and
    // creating a no-op pass through the compiler.
    //
    // Idea: After a "read" instruction, you have a certain window of time to delay the next command
    // buffer. This should compound with the CPU-side "constant folding". To prevent this from
    // harming GPU-executed performance in the future, it wears off after a fixed number of µs. For
    // example, the timer could be 1/2 the round-trip latency of a command buffer (1/2 * 200 µs, but
    // can vary between platforms). I could track the minimum command buffer latency at runtime to
    // get a better estimate of its value across all GPUs.
    flushStream(precomputedBackPressure: backPressure)
  }
  
  func flushStream(precomputedBackPressure: Int? = nil) {
    let numEagerOperations = eagerOperations.count
    guard numEagerOperations > 0 else {
      return
    }
    var compileStartTime: UInt64 = 0
    var encodeStartTime: UInt64 = 0
    if Context.profilingEncoding {
      compileStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    let compiledOperations = compileEagerOperations()
   
    if Context.profilingEncoding {
      encodeStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    let previousBackPressure = precomputedBackPressure ?? queryQueueBackPressure()
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    
    // TODO: Divide the array of compiled operations when there is an out of memory error. Try to
    // eliminate a function call by doing this in a loop.
    for operation in compiledOperations {
      try! encodeCompiledOperation(operation, into: encoder)
    }
    encoder.endEncoding()
    
    numCommittedBatches.wrappingIncrement(ordering: .sequentiallyConsistent)
    commandBuffer.addScheduledHandler { _ in
      self.numScheduledBatches.wrappingIncrement(ordering: .sequentiallyConsistent)
      // TODO: Look back into the latency here. If the CPU does a control flow operator depending
      // on flushing the command stream, checking numCommitted == numCompleted here could halve
      // total latency. I originally settled on using the completion handler because in the
      // scheduled handler, it was so unreliable and often harmed performance or did nothing. I
      // should revisit this once I confirm the delay for a total stop of the pipeline is 400 μs.
      // If done right, I could sometimes reduce it to 200 μs here.
      //
      // "constant folding" on the CPU should reduce the overhead of scalar-wise operators after the
      // read to near-zero, so maybe we don't need to wait for two command buffers to come through
      // (2 x 200 μs). In that case, flushing the command stream in this scheduled handler would be
      // pointless. The delay is 200 μs in every case.
      //
      // This comment relates to the comment in `commentIncrement(inputID:outputID:)` above the call
      // to `flushStream(precomputedBackpressure:)`.
    }
    
    // Avoid retaining the reference to `compiledOperations` just to query its count later on.
    let numCompiledOperations = compiledOperations.count
    // Retain compiled operations in this closure.
    let retainCompiledOperations = { _ = compiledOperations }
    
    commandBuffer.addCompletedHandler { _ in
      let numCommitted = self.numCommittedBatches.load(ordering: .sequentiallyConsistent)
      let numCompleted = self.numCompletedBatches.wrappingIncrementThenLoad(
        ordering: .sequentiallyConsistent)
      
      // For when the CPU does something I/O blocking, yet the GPU has commands to execute. The
      // frontend never calls into the backend, leaving the GPU starved of work.
      if numCommitted == numCompleted {
        Context._dispatchQueue.async {
          retainCompiledOperations()
          Context.global.flushStream()
        }
      } else {
        Context._dispatchQueue.async(execute: retainCompiledOperations)
      }
    }
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
    
    if Context.profilingEncoding {
      let encodeEndTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
      // Try to vectorize the division by 1000.
      let compileDuration = Int(encodeStartTime - compileStartTime) / 1000
      let encodeDuration = Int(encodeEndTime - encodeStartTime) / 1000
      print("""
        Compile time: \(compileDuration) \(Profiler.timeUnit), \
        Encode time: \(encodeDuration) \(Profiler.timeUnit), \
        Batches in flight: \(previousBackPressure), \
        #Commands: \(numEagerOperations) -> \(numCompiledOperations)
        """)
    }
  }
}
