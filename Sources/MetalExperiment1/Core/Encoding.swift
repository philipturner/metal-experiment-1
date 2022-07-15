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
    // Using relaxed memory ordering because it doesn't need to be atomic in the first place. It is
    // never modified by threads other than the encapsulating dispatch queue.
    let commandBufferID = chosenID ?? (numCommittedBatches.load(ordering: .sequentiallyConsistent) - 1)
    if let lastCommandBuffer = commandBufferDictionary[commandBufferID] {
      lastCommandBuffer.waitUntilCompleted()
    }
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
    
    // This part of the function needs to be semantically separated from the part that processes
    // unique instructions. If they are physically separated by a lack of inlining, that is fine.
    // Especially if the second body tracks the real-time minimum command buffer latency, which
    // produces a lot of assembly instructions.
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
    //
    // Perhaps to query a reasonable minimum for command buffer latency, I can send an empty one at
    // program startup. It should take longer than most because the system isn't fired up, but it
    // will at least be smaller than a supermassive batch with 128 unique commands in it.
    flushStream(precomputedBackPressure: backPressure)
  }
  
  func flushStream(precomputedBackPressure: Int? = nil) {
    let numEagerOperations = eagerOperations.count
    guard numEagerOperations > 0 else {
      return
    }
    
    // Start profiling compilation.
    var compileStartTime: UInt64 = 0
    if Context.profilingEncoding {
      compileStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    let compiledOperations = compileEagerOperations()
    
    // Start profiling encoding.
    var encodeStartTime: UInt64 = 0
    if Context.profilingEncoding {
      encodeStartTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    let previousBackPressure = precomputedBackPressure ?? queryQueueBackPressure()
    defer {
      if Context.profilingEncoding {
        let encodeEndTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Try to vectorize the division by 1000.
        let compileDuration = Int(encodeStartTime - compileStartTime) / 1000
        let encodeDuration = Int(encodeEndTime - encodeStartTime) / 1000
        print("""
          Compile time: \(compileDuration) \(Profiler.timeUnit), \
          Encode time: \(encodeDuration) \(Profiler.timeUnit), \
          Batches in flight: \(previousBackPressure), \
          #Commands: \(numEagerOperations) -> \(compiledOperations.count)
          """)
      }
    }
    
    // If the compiler removes all eager operations (by constant folding or eliding no-ops), avoid
    // the overhead of creating a command buffer.
    if compiledOperations.count == 0 {
      return
    }
    
    // Using relaxed memory ordering because it doesn't need to be atomic in the first place. It is
    // never modified by threads other than the encapsulating dispatch queue.
    var commandBufferID = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    var encodingContext = EncodingContext(
      commandBuffer: commandQueue.makeCommandBuffer()!, commandBufferID: commandBufferID)
    commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
    
    // Only called in one location, the loop that iterates over each operation.
    func submitBatch() {
      
    }
    
    precondition(compiledOperations.count == 1)
    var i = 0
    var rangeStart = 0
    repeat {
      let operation = compiledOperations[i]
      try! encodeCompiledOperation(operation, into: &encodingContext)
      
      let nextIterator = i + 1
      let isLoopEnd = nextIterator == compiledOperations.count
      if isLoopEnd {
//        submitBatch(range: rangeStart..<nextIterator)
      }
      i = nextIterator
    }
    while i < compiledOperations.count
            
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
      
      // For when the CPU does something I/O blocking, yet the GPU has commands to execute. The
      // frontend never calls into the backend, leaving the GPU starved of work.
      if numCommitted == numCompleted {
        Context._dispatchQueue.async {
          retainClosure()
          Context.global.flushStream()
        }
      } else {
        Context._dispatchQueue.async(execute: retainClosure)
      }
    }
    encodingContext.commandBuffer.commit()
  }
}
