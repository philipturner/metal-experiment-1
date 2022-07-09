//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  func commitComputeCommand(profilingEncoding: Bool = false) {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.enqueue()
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    var startTime: UInt64 = 0
    if profilingEncoding {
      startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    
    computeEncoder.setComputePipelineState(computePipeline)
    computeEncoder.setBuffer(buffer1, offset: 0, index: 0)
    computeEncoder.setBuffer(buffer2, offset: 0, index: 1)
    computeEncoder.dispatchThreads([Context.bufferNumElements], threadsPerThreadgroup: 1)
    computeEncoder.endEncoding()
//    let event = cycleEvents()
//    commandBuffer.encodeSignalEvent(event, value: event.signaledValue + 1)
    
    let atomic = cycleAtomics()
    while atomic.load(ordering: .sequentiallyConsistent) == -1 {
      print("Command queue overfilled, deadlocking.")
    }
    atomic.store(-1, ordering: .sequentiallyConsistent)
    commandBuffer.addCompletedHandler { _ in
      atomic.store(0, ordering: .sequentiallyConsistent)
    }
    
    if profilingEncoding {
      let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
      let duration = Int(endTime - startTime) / 1000
      print("""
        Encoding duration: \(duration) \(Profiler.timeUnit), \
        Cmdbufs in flight: \(queryActiveCommandBuffers())
        """)
    }
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
    
    operationCount += 1
    let previousBuffer1 = buffer1
    let previousBuffer2 = buffer2
    swap(&buffer1, &buffer2)
    precondition(previousBuffer1 === buffer2)
    precondition(previousBuffer2 === buffer1)
  }
  
  func queryActiveCommandBuffers() -> Int {
    var count = 0
    for atomic in atomics {
      if atomic.load(ordering: .sequentiallyConsistent) == -1 {
        count += 1
      }
    }
    return count
  }
  
  func validate(withBarrier: Bool = true, showingStats: Bool = false) {
    if withBarrier {
      barrier(showingStats: showingStats)
    }
    let lastOutputBuffer = buffer1
    let ptr = lastOutputBuffer.contents().assumingMemoryBound(to: Float.self)
    precondition(ptr[0] == Float(operationCount))
  }
}

extension Context {
  func commitStreamedCommand(profilingEncoding: Bool = false) {
    var commandBuffer: MTLCommandBuffer
    var computeEncoder: MTLComputeCommandEncoder
    if let currentCommandBuffer = currentCommandBuffer {
      commandBuffer = currentCommandBuffer
      computeEncoder = currentComputeEncoder!
    } else {
      commandBuffer = commandQueue.makeCommandBuffer()!
      commandBuffer.enqueue()
      computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    }
    var startTime: UInt64 = 0
    if profilingEncoding {
      startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    
    computeEncoder.setComputePipelineState(computePipeline)
    computeEncoder.setBuffer(buffer1, offset: 0, index: 0)
    computeEncoder.setBuffer(buffer2, offset: 0, index: 1)
    computeEncoder.dispatchThreads([Context.bufferNumElements], threadsPerThreadgroup: 1)
    
    let cmdbufsInFlight = queryActiveCommandBuffers()
    var shouldFlushStream = numEncodedCommands >= Context.maxCommandsPerCmdbuf
    shouldFlushStream = shouldFlushStream || (cmdbufsInFlight == 0)
    currentCommandBuffer = commandBuffer
    currentComputeEncoder = computeEncoder
    numEncodedCommands += 1
    
    if profilingEncoding {
      let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
      let duration = Int(endTime - startTime) / 1000
      print("""
        Encoding duration: \(duration) \(Profiler.timeUnit), \
        Cmdbufs in flight: \(cmdbufsInFlight), \
        Encoded commands: \(numEncodedCommands), \
        Flushed stream: \(shouldFlushStream)
        """)
    }
    
    if shouldFlushStream {
      flushStream()
    }
    
    operationCount += 1
    let previousBuffer1 = buffer1
    let previousBuffer2 = buffer2
    swap(&buffer1, &buffer2)
    precondition(previousBuffer1 === buffer2)
    precondition(previousBuffer2 === buffer1)
  }
  
  func flushStream() {
    guard let commandBuffer = currentCommandBuffer else {
      return
    }
    let computeEncoder = currentComputeEncoder!
    currentCommandBuffer = nil
    currentComputeEncoder = nil
    numEncodedCommands = 0
    
    computeEncoder.endEncoding()
    let atomic = cycleAtomics()
    while atomic.load(ordering: .sequentiallyConsistent) == -1 {
      print("Command queue overfilled, deadlocking.")
    }
    atomic.store(-1, ordering: .sequentiallyConsistent)
    commandBuffer.addCompletedHandler { _ in
      atomic.store(0, ordering: .sequentiallyConsistent)
    }
    
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
  }
}
