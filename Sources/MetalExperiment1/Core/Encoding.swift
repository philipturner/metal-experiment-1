//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  static let dispatchQueue = DispatchQueue(label: "com.s4tf.metal.Context.dispatchQueue")
  
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
    let lastOutputBuffer = buffer1
    let ptr = lastOutputBuffer.contents().assumingMemoryBound(to: Float.self)
    precondition(ptr[0] == Float(operationCount))
  }
}

private extension Context {
  // Compile a stream of commands to optimize it, transforming into a lower-level IR. Memory
  // allocation happens afterwards, during `flushStream`.
  
  func commitStreamedCommand() {
    let operation = EagerOperation.Unary(
      type: .increment, input: buffer1, output: buffer2, size: Context.numBufferElements)
    bufferedOperations.append(.unary(operation))
    operationCount += 1
    swap(&buffer1, &buffer2)
    
    let backPressure = queryQueueBackPressure()
    if bufferedOperations.count < Context.maxCommandsPerBatch,
       backPressure >= 1 {
      return
    }
    flushStream(precomputedBackPressure: backPressure)
  }
  
  func flushStream(precomputedBackPressure: Int? = nil) {
    guard bufferedOperations.count > 0 else {
      return
    }
    defer {
      bufferedOperations.removeAll(keepingCapacity: true)
    }
    let previousBackPressure = precomputedBackPressure ?? queryQueueBackPressure()
    
    var startTime: UInt64 = 0
    if Context.profilingEncoding {
      startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    for operation in bufferedOperations {
      encodeEagerOperation(operation, into: encoder)
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
    commandBuffer.addCompletedHandler { _ in
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
      let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
      let duration = Int(endTime - startTime) / 1000
      print("""
        Encoding duration: \(duration) \(Profiler.timeUnit), \
        Batches in flight: \(previousBackPressure), \
        Encoded commands: \(bufferedOperations.count)
        """)
    }
  }
}
