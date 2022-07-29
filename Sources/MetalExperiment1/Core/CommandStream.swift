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
    let commandBufferID = chosenID ?? (_fastLoadCommittedBatches() - 1)
    if let lastCommandBuffer = commandBufferDictionary[commandBufferID] {
      if lastCommandBuffer.status != .completed {
        releaseMutex()
        lastCommandBuffer.waitUntilCompleted()
        acquireMutex()
        justFinishedBarrier = true
      }
    }
  }
  
  func maybeFlushStream() {
    let numCommitted = _fastLoadCommittedBatches()
    let numCompleted = numCompletedBatches.load(ordering: .relaxed)
    let backPressure = numCommitted - numCompleted
    
    var shouldFlush: Bool
    if backPressure == 0 {
      shouldFlush = true
    } else if backPressure == 1 {
      let numScheduled = numScheduledBatches.load(ordering: .relaxed)
      shouldFlush = numCommitted == numScheduled
    } else {
      shouldFlush = false
    }
    guard eagerOperations.count > Context.maxCommandsPerBatch ||
          shouldFlush else {
      return
    }
    precondition(eagerOperations.count <= Context.maxCommandsPerBatch + 1)
    
    // The stream gets off to a very slow start right after reading a buffer's contents, being
    // unable to fuse unary operations and creating a no-op pass through the compiler. This
    // mechanism waits X microseconds before flushing again, letting the queue fill up. The timer's
    // delay is ~1/4 the round-trip latency of a command buffer, but almost exactly enough time for
    // 128 commands to queue up.
    if !waitingOnTimer && justFinishedBarrier && backPressure == 0 {
      justFinishedBarrier = false
      if eagerOperations.count <= Context.maxCommandsPerBatch {
        waitingOnTimer = true
        
        let delay = Int(schedulingLatency.average)
        let currentNumCommittedBatches = _fastLoadCommittedBatches()
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        schedulingQueue.async {
          func loadCommitted() -> Int {
            self.numCommittedBatches.load(ordering: .relaxed)
          }
          if loadCommitted() > currentNumCommittedBatches {
            self.waitingOnTimer = false
            return
          }
          
          // Use a spin loop. Asking the dispatch queue to execute at a specific deadline drives the
          // actual delay to 300 µs (it should be 50 µs).
          var end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
          while end - start < delay {
            usleep(10)
            if loadCommitted() > currentNumCommittedBatches {
              self.waitingOnTimer = false
              return
            }
            end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
          }
          
          self.sync {
            self.waitingOnTimer = false
            guard self._fastLoadCommittedBatches() == currentNumCommittedBatches else {
              return
            }
            self.flushStream()
          }
        }
      }
    }
    if eagerOperations.count <= Context.maxCommandsPerBatch,
       waitingOnTimer && backPressure == 0 {
      return
    }
    
    // If it flushed because the batch was really long, don't include the very last operation.
    // `maybeFlushStream` is called in the middle of submitting an operation, and all of the inputs
    // are being retained. That prevents fusion.
    var currentOperation: EagerOperation?
    if eagerOperations.count > Context.maxCommandsPerSmallBatch {
      currentOperation = eagerOperations.removeLast()
    }
    defer {
      if let currentOperation = currentOperation {
        precondition(eagerOperations.count == 0)
        eagerOperations.append(currentOperation)
      }
    }
    
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
    //
    // I could also provide an option that limits the compiled command stream size. But the user
    // shouldn't have to (and probably won't) play around with hyperparameters like that just to
    // train their model.
    //
    // Benefits of AI approach:
    // - Stops the frontend from freezing someone's computer if they are using an Apple or Intel
    //   integrated GPU. They can do other things on their computer while the model is training.
    // - Can scale up or down the intensity of the graph compiler. When operations are taking a long
    //   time, the AI gives it more freedom to take longer to compile.
    // - Can exceed the 128 operation limit, going to 1000 in some instances. That could decrease
    //   sequential latency to the theoretical minimum.
    // - Allows for 4 or 5 inputs to an elementwise shader, depending on the GPU architecture and
    //   its register size. Could determine the register size at runtime by profiling register
    //   spills, although the shader must be compiled multiple times to do this.
    // - Generalizes to different workloads, requiring no prior knowledge of the frontend.
    // - An opportunity to show off S4TF.
    //
    // Drawbacks of AI approach:
    // - Takes a lot of time to develop.
    // - Might be over-optimizing for a specific feature.
    // - There might be an easier way to solve this problem, but it doesn't generalize to other
    //   domains and you have to hard-code certain parameters.
    // - If it depends on disk storage inside the package bundle to aggregate data over multiple
    //   startups, that data would clear when you purge build products.
    // - Time spent on this is time spent not developing the OpenCL backend.
    //
    // Current stance: Wait until real-world data indicates that there's no better solution, then
    // pursue the AI approach. This will definitely come *after* integrating the Metal backend into
    // Swift for TensorFlow.
    flushStream(precomputedBackPressure: backPressure)
  }
}

// Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
// allocation happens afterwards, during `flushStream`.
private extension Context {
  @inline(__always)
  func queryQueueBackPressure() -> Int {
    let numCommitted = _fastLoadCommittedBatches()
    let numCompleted = numCompletedBatches.load(ordering: .relaxed)
    return numCommitted - numCompleted
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
    
    // If the compiler removes all eager operations with constant folding, avoid the overhead of
    // creating a command buffer.
    if instructions.count == 0 {
      return
    }
    
    var commandBufferID = _fastLoadCommittedBatches()
    var encodingContext = EncodingContext(
      commandBuffer: commandQueue.makeCommandBufferWithUnretainedReferences()!,
      commandBufferID: commandBufferID)
    commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
    
    // Only called in one location, the loop that iterates over each operation.
    func submitBatch(range: Range<Int>) {
      encodingContext.finishCommandEncoder()
      
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
      
      let currentCommandBufferID = commandBufferID
      commandBufferID += 1
      numCommittedBatches.wrappingIncrement(ordering: .relaxed)
      
      let scheduleStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
      encodingContext.commandBuffer.addScheduledHandler { commandBuffer in
        precondition(commandBuffer.status == .scheduled, commandBuffer.errorMessage)
        self.numScheduledBatches.wrappingIncrement(ordering: .relaxed)
        let scheduleEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let scheduleDuration = scheduleEnd - scheduleStart
        
        DispatchQueue.global().async {
          self.sync {
            self.schedulingLatency.append(scheduleDuration)
          }
        }
      }
      
      encodingContext.commandBuffer.addCompletedHandler { commandBuffer in
        precondition(commandBuffer.status == .completed, commandBuffer.errorMessage)
        self.numCompletedBatches.wrappingIncrement(ordering: .relaxed)
        
        // For when the CPU does something I/O blocking, yet the GPU has commands to execute. The
        // frontend never calls into the backend, leaving the GPU starved of work.
        DispatchQueue.global().async {
          self.sync {
            // Captures `currentCommandBufferID` declared above.
            self.commandBufferDictionary[currentCommandBufferID] = nil
            retainer.release()
            
            if self.eagerOperations.count > 0,
               self.queryQueueBackPressure() == 0 {
              self.flushStream()
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
              commandBuffer: commandQueue.makeCommandBufferWithUnretainedReferences()!,
              commandBufferID: commandBufferID)
            commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
          }
        }
        if encounteredError {
          if _fastLoadCommittedBatches() == 0 {
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
