//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  // TODO: A special `with` function the caller can use to make everything inside happen on the
  // dispatch queue. This eliminates the extra 0.3 µs overhead from calling into the dispatch queue
  // on each function call. The actual dispatch queue will still be opaque. This should be used
  // widely in the _Raw namespace.
  static let dispatchQueue = DispatchQueue(label: "com.s4tf.metal.Context.dispatchQueue")
  
  // Internally, use an inline-always synchronization function that checks a global variable.
  // static func withSynchronization(_ body: ???)
  
  // Also, once you enter synchronization in any situation, this wrapper could free deadlocks. But
  // wait to implement that feature so you can find thread synchronization bugs in the early stages
  // of development.
  
  static func validate() {
    dispatchQueue.sync {
      Context.global.validate()
    }
  }
  
  static func commitStreamedCommand() {
    dispatchQueue.sync {
      Context.global.commitStreamedCommand()
    }
  }
  
  static func commitIncrement(inputID: UInt64, outputID: UInt64) {
    dispatchQueue.sync {
      Context.global.commitIncrement(inputID: inputID, outputID: outputID)
    }
  }
  
  static func barrier() {
    dispatchQueue.sync {
      Context.global.barrier()
    }
  }
  
  static func _unsafeBarrier() {
    Context.global.barrier()
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
  
  func validate() {
    barrier()
    let allocation = try! _unsafeFetchAllocation(id: allocation1)!
    let lastOutputBuffer = allocation.mtlBuffer!
    let ptr = lastOutputBuffer.contents().assumingMemoryBound(to: Float.self)
    precondition(ptr[0] == Float(operationCount))
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
    try! _unsafeRetain(id: inputID)
    try! _unsafeRetain(id: outputID)
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
    // example, the round-trip latency of a command buffer (150 µs, but can vary between platforms).
    flushStream(precomputedBackPressure: backPressure)
  }
  
  func flushStream(precomputedBackPressure: Int? = nil) {
    let numEagerOperations = eagerOperations.count
    guard numEagerOperations > 0 else {
      return
    }
    let previousBackPressure = precomputedBackPressure ?? queryQueueBackPressure()
    
    var compileStartTime: UInt64 = 0
    if Context.profilingEncoding {
      compileStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    let compiledOperations = compileEagerOperations()
    // Avoid retaining the reference to `compiledOperations` just to query its count later on.
    let numCompiledOperations = compiledOperations.count
    
    var encodeStartTime: UInt64 = 0
    if Context.profilingEncoding {
      encodeStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
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
    }
    commandBuffer.addCompletedHandler { [compiledOperations] _ in
      // Retain compiled operations in this closure.
      _ = compiledOperations
      let numCommitted = self.numCommittedBatches.load(ordering: .sequentiallyConsistent)
      let numCompleted = self.numCompletedBatches.wrappingIncrementThenLoad(
        ordering: .sequentiallyConsistent)
      
      // For when the CPU does something I/O blocking, yet the GPU has commands to execute. The
      // frontend never calls into the backend, leaving the GPU starved of work.
      if numCommitted == numCompleted {
        Context.dispatchQueue.async {
          Context.global.flushStream()
        }
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
