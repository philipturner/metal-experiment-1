//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  public static func commitIncrement(inputID: UInt64, outputID: UInt64, size: Int) {
    withDispatchQueue {
      Context.global.commitIncrement(inputID: inputID, outputID: outputID, size: size)
    }
  }
  
  static func barrier() {
    withDispatchQueue {
      let ctx = Context.global
      ctx._compilerFlushStream()
      ctx._compilerBarrier()
    }
  }
  
  @inline(__always)
  internal func _compilerFlushStream() {
    flushStream()
  }
  
  // Flush the command stream before calling this function.
  @inline(__always)
  internal func _compilerBarrier(commandBufferID chosenID: Int? = nil) {
    // Using relaxed memory ordering because it doesn't need to be atomic in the first place. It is
    // never modified by threads other than the encapsulating dispatch queue.
    let commandBufferID = chosenID ?? (numCommittedBatches.load(ordering: .relaxed) - 1)
    if let lastCommandBuffer = commandBufferDictionary[commandBufferID] {
      lastCommandBuffer.waitUntilCompleted()
    }
  }
}

extension Context {
  public static func executeOperation(
    _ name: UnsafePointer<Int8>,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<UInt64>
  ) {
    _dispatchQueue.sync {
      Context.global._executeOperation(name, inputs, outputs)
    }
  }
  
  // TODO: Figure out how to pass attributes in. Try the OpenCL approach, which should have the
  // lowest overhead.
  private func _executeOperation(
    _ name: UnsafePointer<Int8>,
    _ inputs: UnsafeBufferPointer<UInt64>,
    _ outputs: UnsafeMutableBufferPointer<UInt64>
  ) {
    // When there are multiple ops, instead of switching over an enumeration, query a dictionary of
    // strings to @convention(c) function pointers. To avoid the overhead of creating a string
    // object, try using something besides a `String` for the dictionary's key. For example, limit
    // strings to 63 bytes and make an efficient hash table. How fast is Swift's automatic hashing
    // mechanism?
    precondition(String(cString: name) == "increment")
    precondition(inputs.count == 1)
    precondition(outputs.count == 1)
    
    // TODO: Optimize this by only fetching from the dictionary once. Use the retrieved allocation
    // for both extracting size and incrementing its reference count. Also, make a force-inlined
    // internal function for generating IDs, which returns the newly generated `Allocation` without
    // requiring that it be fetched from a dictionary later on.
    let input1 = inputs[0]
    let allocationSize = _compilerFetchAllocation(id: input1).size
    let output1 = _compilerGenerateID(allocationSize: allocationSize)
    let size = allocationSize / MemoryLayout<Float>.stride
    commitIncrement(inputID: input1, outputID: output1, size: size)
  }
}

// Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
// allocation happens afterwards, during `flushStream`.
private extension Context {
  @inline(__always)
  func queryQueueBackPressure() -> Int {
    let numCommitted = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    let numScheduled = numScheduledBatches.load(ordering: .sequentiallyConsistent)
    return numCommitted - numScheduled
  }
  
  func commitIncrement(inputID: UInt64, outputID: UInt64, size: Int) {
    _compilerRetain(id: inputID)
    _compilerRetain(id: outputID)
    let operation = EagerOperation.Unary(
      type: .increment, input: inputID, output: outputID, size: size)
    eagerOperations.append(.unary(operation))
    
    // TODO: Extract this code into the top-level function that calls into `eagerOperations.append`.
    
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
    var commandBufferID = numCommittedBatches.load(ordering: .relaxed)
    var encodingContext = EncodingContext(
      commandBuffer: commandQueue.makeCommandBuffer()!, commandBufferID: commandBufferID)
    commandBufferDictionary[commandBufferID] = encodingContext.commandBuffer
    
    // Only called in one location, the loop that iterates over each operation.
    func submitBatch(range: Range<Int>) {
      encodingContext.finishEncoder()
      
      // Force the memory allocations to stay alive until the command buffer finishes.
      var retainClosure: () -> Void
      if range == compiledOperations.indices {
        // If `commandBufferID` is captured without doing anything else, it will register as
        // something greater than we want inside the closure. To fix this, each closure captures the
        // ID inside its explicit capture list.
        retainClosure = { [commandBufferID = commandBufferID] in
          Context.global.commandBufferDictionary[commandBufferID] = nil
          _ = compiledOperations
        }
      } else {
        // Capture only the range of operations encoded in this batch.
        let submittedOperations = Array(compiledOperations[range])
        retainClosure = { [commandBufferID = commandBufferID] in
          Context.global.commandBufferDictionary[commandBufferID] = nil
          _ = submittedOperations
        }
      }
      
      // Instead of `wrappingIncrement(ordering:)`, this code section uses `store(_:ordering:)`. It
      // always executes on the same thread, so it's okay to increment in a way that isn't perfectly
      // atomic.
      commandBufferID += 1
      numCommittedBatches.store(commandBufferID, ordering: .sequentiallyConsistent)
      
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
      encodingContext.commandBuffer.addScheduledHandler { _ in
        self.numScheduledBatches.wrappingIncrement(ordering: .sequentiallyConsistent)
      }
      
      encodingContext.commandBuffer.addCompletedHandler { _ in
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
    
    var i = 0
    var rangeStart = 0
    repeat {
      let operation = compiledOperations[i]
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
      let isLoopEnd = nextIterator == compiledOperations.count
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
          if numCommittedBatches.load(ordering: .sequentiallyConsistent) == 0 {
            if Context.profilingEncoding || HeapAllocator.debugInfoEnabled {
              print("""
                One of the first commands ever submitted was to interact with an exorbitant amount \
                of memory. An allocation may have exceeded the size of your GPU's RAM.
                """)
            }
          } else {
            _compilerBarrier()
          }
        }
      }
      i = nextIterator
    }
    while i < compiledOperations.count
  }
}
