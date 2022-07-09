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
}

private extension Context {
  func barrier() {
    if let commandBuffer = lastCommandBuffer {
      commandBuffer.waitUntilCompleted()
    }
  }
  
  func queryQueueBackPressure() -> Int {
    let committedCount = numCommittedBatches.load(ordering: .sequentiallyConsistent)
    let scheduledCount = numScheduledBatches.load(ordering: .sequentiallyConsistent)
    return committedCount - scheduledCount
  }
  
  func validate() {
    flushStream()
    barrier()
    let lastOutputBuffer = buffer1
    let ptr = lastOutputBuffer.contents().assumingMemoryBound(to: Float.self)
    precondition(ptr[0] == Float(operationCount))
  }
}

private extension Context {
  func commitStreamedCommand() {
    let operation = Operation.Unary(
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
      encodeSingleOperation(operation, into: encoder)
    }
    encoder.endEncoding()
    
    numCommittedBatches.wrappingIncrement(ordering: .sequentiallyConsistent)
    let atomic = numScheduledBatches
    commandBuffer.addScheduledHandler { _ in
      atomic.wrappingIncrement(ordering: .sequentiallyConsistent)
      // Send a thread-safe call to `flushStream`.
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
