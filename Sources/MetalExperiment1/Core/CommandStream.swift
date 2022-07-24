//
//  CommandStream.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  static func barrier() {
    Context.global.sync {
      let ctx = Context.global
      ctx._internalFlushStream()
      ctx._internalBarrier()
    }
  }
  
  @inline(__always)
  func _internalFlushStream() {
    flushStream()
  }
  
  // Flush the command stream before calling this function.
  @inline(__always)
  func _internalBarrier(commandBufferID chosenID: Int? = nil) {
    // Using relaxed memory ordering because it doesn't need to be atomic in the first place. It is
    // never modified by threads other than the encapsulating dispatch queue.
    let commandBufferID = chosenID ?? (numCommittedBatches.load(ordering: .relaxed) - 1)
    if let lastCommandBuffer = commandBufferDictionary[commandBufferID] {
      lastCommandBuffer.waitUntilCompleted()
    }
  }
  
  // Only called in one location, but force-inlining anyway.
  @inline(__always)
  func maybeFlushStream() {
    let backPressure = queryQueueBackPressure()
    if eagerOperations.count < Context.maxCommandsPerBatch,
       backPressure >= 1 {
      return
    }
    
    // TODO: Add a heuristic that waits a few instructions before submitting. It gets off to a very
    // slow start right after reading a buffer's contents, being unable to fuse unary operations and
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
    
    // TODO: Heuristic that holds off on encoding 128 operations at least one has fully completed.
    // The GPU might be running its first convolution op, then you make 128 equally long convolution
    // ops that stall the GPU because the command buffer is huge. The maximum batch size could
    // exponentially grow up to 128, which fits with the performance pattern of the first
    // instructions taking longer. They have to wait on `MPSGraph`s and initial heap expansion
    // anyway.
    //
    // This does not mean you halt the acceptance of ops. This needs to wait as long as possible to
    // accept more MPSGraph submissions. This gradual expansion of batch size happens while flushing
    // the stream. It compiles all 128 operations to fish out elementwise ones that can be fused.
    // Between fusion boundaries, there is little cost to making new command buffers.
    //
    // A more refined idea: dynamically expand or contract command buffer size based on how long
    // operations are taking. This can use short-term memory, which is damped and doesn't respond
    // well to sudden changes in operation distribution. It can also use an elaborate long-term
    // memory mechanism, which hashes complex operations and profiles their execution time with
    // respect to several parameters. Perhaps it could be an opt-in plugin. This is sounding like a
    // real-time AI making quick decisions, running on the Apple AMX.
    //
    // It can be a costly neural network because it fires when dealing with excessively long
    // operations. For short ones like elementwise ops, it doesn't fire and isn't needed. This AI
    // could be generalizable for not just ML ops, but any GPGPU domain including linear algebra.
    // Even more awesome - I can train it using Swift for TensorFlow!
    flushStream(precomputedBackPressure: backPressure)
  }
}

// Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
// allocation happens afterwards, during `flushStream`.
private extension Context {
  @inline(__always)
  func queryQueueBackPressure() -> Int {
    let numCommitted = numCommittedBatches.load(ordering: .relaxed)
    let numScheduled = numScheduledBatches.load(ordering: .relaxed)
    return numCommitted - numScheduled
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
    let instructions = compileEagerOperations()
    
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
          #Commands: \(numEagerOperations) -> \(instructions.count)
          """)
      }
    }
    
    // If the compiler removes all eager operations (by constant folding or eliding no-ops), avoid
    // the overhead of creating a command buffer.
    if instructions.count == 0 {
      return
    }
    
    var commandBufferID = numCommittedBatches.load(ordering: .relaxed)
    var encodingContext = EncodingContext(
      commandBuffer: commandQueue.makeCommandBuffer()!, commandBufferID: commandBufferID)
    commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
    
    // Only called in one location, the loop that iterates over each operation.
    func submitBatch(range: Range<Int>) {
      encodingContext.finishEncoder()
      
      // Force the memory allocations to stay alive until the command buffer finishes.
      class Retainer {
        var instructions: [Instruction]
        init(retaining instructions: [Instruction]) {
          self.instructions = instructions
        }
      }
      var retainer: Unmanaged<Retainer>
      if range == instructions.indices {
        retainer = .passRetained(Retainer(retaining: instructions))
      } else {
        // Capture only the range of operations encoded in this batch.
        retainer = .passRetained(Retainer(retaining: Array(instructions[range])))
      }
      
      // Instead of `wrappingIncrement(ordering:)`, this code section uses `store(_:ordering:)`. It
      // always executes on the same thread, so it's okay to increment in a way that isn't perfectly
      // atomic.
      let currentCommandBufferID = commandBufferID
      commandBufferID += 1
      numCommittedBatches.store(commandBufferID, ordering: .relaxed)
      
      // TODO: Look back into the latency here. If the CPU does a control flow operation depending
      // on flushing the command stream, checking numCommitted == numCompleted here could halve
      // total latency. I originally settled on using the completion handler because in the
      // scheduled handler, it was so unreliable and often harmed performance or did nothing. I
      // should revisit this once I confirm the delay for a total stop of the pipeline is 400 μs.
      // If done right, I could sometimes reduce it to 200 μs here.
      //
      // "constant folding" on the CPU should reduce the overhead of scalar-wise operations after
      // the read to near-zero, so maybe we don't need to wait for two command buffers to come
      // through (2 x 200 μs). In that case, flushing the command stream in this scheduled handler
      // would be pointless. The delay is 200 μs in every case.
      //
      // This comment relates to the comment in `maybeFlushStream` above the call to
      // `flushStream(precomputedBackpressure:)`.
      encodingContext.commandBuffer.addScheduledHandler { commandBuffer in
        precondition(commandBuffer.status == .scheduled, commandBuffer.errorMessage)
        self.numScheduledBatches.wrappingIncrement(ordering: .relaxed)
      }
      
      encodingContext.commandBuffer.addCompletedHandler { commandBuffer in
        precondition(commandBuffer.status == .completed, commandBuffer.errorMessage)
        let numCommitted = self.numCommittedBatches.load(ordering: .relaxed)
        let numCompleted = self.numCompletedBatches.wrappingIncrementThenLoad(ordering: .relaxed)
        
        // For when the CPU does something I/O blocking, yet the GPU has commands to execute. The
        // frontend never calls into the backend, leaving the GPU starved of work.
        let shouldFlush = numCommitted == numCompleted
        DispatchQueue.global().async {
          Context.global.sync {
            // Captures `currentCommandBufferID` declared above.
            let ctx = Context.global
            ctx.commandBufferDictionary[currentCommandBufferID] = nil
            retainer.release()
            if shouldFlush {
              ctx.flushStream()
            }
          }
        }
      }
      encodingContext.commandBuffer.commit()
    }
    
    var i = 0
    var rangeStart = 0
    repeat {
      let operation = instructions[i]
      var encounteredError = false
      do {
        try encodeCompiledOperation(operation, into: &encodingContext)
      } catch AllocationError.exceededSystemRAM {
        Context.global.permitExceedingSystemRAM = true
        
        // Retry the command that failed in the next command buffer.
        i -= 1
        encounteredError = true
      } catch {
        fatalError(error.localizedDescription)
      }
      
      let nextIterator = i + 1
      let isLoopEnd = (nextIterator == instructions.count)
      if encounteredError || isLoopEnd {
        if nextIterator > rangeStart {
          // This function increments the command buffer ID.
          submitBatch(range: rangeStart..<nextIterator)
          rangeStart = nextIterator
          if encounteredError {
            encodingContext = EncodingContext(
              commandBuffer: commandQueue.makeCommandBuffer()!, commandBufferID: commandBufferID)
            commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
          }
        }
        if encounteredError {
          if numCommittedBatches.load(ordering: .relaxed) == 0 {
            if Context.profilingEncoding || HeapAllocator.debugInfoEnabled {
              print("""
                One of the first commands ever submitted was to interact with an exorbitant amount \
                of memory. An allocation may have exceeded the size of your GPU's RAM.
                """)
            }
          } else {
            _internalBarrier()
          }
        }
      }
      i = nextIterator
    }
    while i < instructions.count
  }
}
