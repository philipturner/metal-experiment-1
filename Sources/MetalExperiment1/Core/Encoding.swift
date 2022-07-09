//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

extension Context {
  func barrier() {
    if let commandBuffer = lastCommandBuffer {
      commandBuffer.waitUntilCompleted()
    }
  }
  
  func queryActiveCommandBuffers() -> Int {
    committedCmdbufCount - completedCmdbufCount.load(ordering: .sequentiallyConsistent)
  }
  
  func validate() {
    flushStream()
    barrier()
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
    committedCmdbufCount += 1
    let atomic = completedCmdbufCount
    commandBuffer.addCompletedHandler { _ in
      atomic.wrappingIncrement(ordering: .sequentiallyConsistent)
    }
    
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
  }
}
