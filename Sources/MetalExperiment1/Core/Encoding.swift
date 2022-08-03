//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import MetalPerformanceShaders

struct EncodingContext {
  let device: MTLPluggableDevice
  let commandBuffer: MTLCommandBuffer
  let commandBufferID: Int
  private var blitEncoder: MTLBlitCommandEncoder?
  private var computeEncoder: MTLComputeCommandEncoder?
  private var computeEncoderID: Int = -1
  var pipelineStateID: Int = -1
  var synchronizedResources: Set<AllocationHandle> = []
  var barrierResources: [MTLBuffer] = []
  
  @inline(__always)
  init(device: MTLPluggableDevice, commandBuffer: MTLCommandBuffer, commandBufferID: Int) {
    self.device = device
    self.commandBuffer = commandBuffer
    self.commandBufferID = commandBufferID
  }
  
  @inline(__always)
  func encodeWaitForEvent() {
    commandBuffer.encodeWaitForEvent(
      device.synchronizationEvent, value: device.synchronizationCounter)
  }
  
  @inline(__always)
  func encodeSignalEvent() {
    let newCounter = device.synchronizationCounter + 1
    device.synchronizationCounter = newCounter
    commandBuffer.encodeSignalEvent(device.synchronizationEvent, value: newCounter)
  }
  
  // MARK: - Starting Encoding Passes
  
  @inline(__always)
  mutating func makeBlitCommandEncoder() -> MTLBlitCommandEncoder {
    if let blitEncoder = blitEncoder {
      return blitEncoder
    } else {
      return startBlitPass()
    }
  }
  
  private mutating func startBlitPass() -> MTLBlitCommandEncoder {
    var usingFence = false
    if let computeEncoder = computeEncoder {
      usingFence = true
      finishComputePass(computeEncoder, usingEvent: false)
    } else {
      encodeWaitForEvent()
    }
    
    let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
    if usingFence {
      blitEncoder.waitForFence(device.synchronizationFence)
    }
    self.blitEncoder = blitEncoder
    return blitEncoder
  }
  
  @inline(__always)
  mutating func makeComputeCommandEncoder() -> MTLComputeCommandEncoder {
    precondition(barrierResources.isEmpty, "Did not finish memory barrier in last compute pass.")
    if let computeEncoder = computeEncoder {
      return computeEncoder
    } else {
      return startComputePass()
    }
  }
  
  private mutating func startComputePass() -> MTLComputeCommandEncoder {
    var usingFence = false
    if let blitEncoder = blitEncoder {
      usingFence = true
      finishBlitPass(blitEncoder, usingEvent: false)
    } else {
      encodeWaitForEvent()
    }
    
    precondition(synchronizedResources.isEmpty, "This should never happen.")
    let computeEncoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)!
    if usingFence {
      computeEncoder.waitForFence(device.synchronizationFence)
    }
    self.computeEncoder = computeEncoder
    self.computeEncoderID += 1
    return computeEncoder
  }
  
  // MARK: - Finishing Encoding Passes
  
  @inline(__always)
  mutating func finishCommandEncoder() {
    if let blitEncoder = blitEncoder {
      finishBlitPass(blitEncoder, usingEvent: true)
    } else if let computeEncoder = computeEncoder {
      finishComputePass(computeEncoder, usingEvent: true)
    }
  }
  
  private mutating func finishComputePass(
    _ computeEncoder: MTLComputeCommandEncoder,
    usingEvent: Bool
  ) {
    if !usingEvent {
      computeEncoder.waitForFence(device.synchronizationFence)
    }
    computeEncoder.endEncoding()
    self.computeEncoder = nil
    self.pipelineStateID = -1
    self.synchronizedResources.removeAll(keepingCapacity: true)
    if usingEvent {
      encodeSignalEvent()
    }
  }
  
  private mutating func finishBlitPass(
    _ blitEncoder: MTLBlitCommandEncoder,
    usingEvent: Bool
  ) {
    if !usingEvent {
      blitEncoder.waitForFence(device.synchronizationFence)
    }
    blitEncoder.endEncoding()
    self.blitEncoder = nil
    if usingEvent {
      encodeSignalEvent()
    }
  }
  
  // MARK: - Memory Barriers
  
  // Always call this before reading from an allocation's `MTLBuffer`.
  @inline(__always)
  func markAsRead(_ allocation: Allocation) {
    precondition(
      allocation.materialized, "Materialize an allocation's 'MTLBuffer' before marking it as read.")
    allocation.lastReadCommandBufferID = commandBufferID
  }
  
  // Always call this before writing to an allocation's `MTLBuffer`.
  @inline(__always)
  mutating func markAsModified(_ allocation: Allocation) {
    precondition(
      allocation.materialized,
      "Materialize an allocation's 'MTLBuffer' before marking it as modified.")
    precondition(allocation.lastModifiedCommandBufferID == -1, "Cannot initialize something twice.")
    allocation.lastModifiedCommandBufferID = commandBufferID
    if computeEncoder != nil {
      allocation.lastModifiedCommandEncoderID = computeEncoderID
      synchronizedResources.insert(allocation.handle)
    }
  }
  
  // Returns whether the encoder should wait on the resource.
  @inline(__always)
  mutating func addBarrierResource(_ allocation: Allocation, buffer: MTLBuffer) {
    guard allocation.lastModifiedCommandBufferID == commandBufferID,
          allocation.lastModifiedCommandEncoderID == computeEncoderID else {
      return
    }
    barrierResources.append(buffer)
    
    // This check is costly, so only perform it in debug mode.
    assert(
      synchronizedResources.contains(allocation.handle),
      "Did not erase encoder ID after executing memory barrier on resource.")
    synchronizedResources.remove(allocation.handle)
    allocation.lastModifiedCommandEncoderID = -1
  }
  
  @inline(__always)
  mutating func memoryBarrier() {
    if barrierResources.count > 0 {
      computeEncoder!.memoryBarrier(resources: barrierResources)
      barrierResources.removeAll(keepingCapacity: true)
    }
  }
}

// MARK: - Encoding

extension MTLPluggableDevice {
  func encodeInstruction(
    _ instruction: Instruction,
    into ectx: inout EncodingContext
  ) throws {
    switch instruction {
    case .elementwise(let elementwise):
      try encodeElementwise(elementwise, into: &ectx)
    case .explicitCopy(let explicitCopy):
      try encodeExplicitCopy(explicitCopy, into: &ectx)
    }
  }
}

extension MTLPluggableDevice {
  func encodeElementwise(
    _ instruction: Instruction.Elementwise,
    into ectx: inout EncodingContext
  ) throws {
    let encoder = ectx.makeComputeCommandEncoder()
    switch instruction.dataGroup {
    case .f32_i32:
      if ectx.pipelineStateID == 1 {
        // Avoid overhead of Metal API call.
      } else {
        encoder.setComputePipelineState(shaderCache.elementwise_f32_i32)
        ectx.pipelineStateID = 1
      }
    case .u32_i64_u64:
      if ectx.pipelineStateID == 2 {
        // Avoid overhead of Metal API call.
      } else {
        encoder.setComputePipelineState(shaderCache.elementwise_u32_i64_u64)
        ectx.pipelineStateID = 2
      }
    }
    
    for i in 0..<5 {
      var allocation: Allocation?
      switch i {
      case 0: allocation = instruction.input1
      case 1: allocation = instruction.input2
      case 2: allocation = instruction.input3
      case 3: allocation = instruction.input4
      default: /*4*/
        allocation = instruction.output
      }
      guard let allocation = allocation else {
        continue
      }
      
      if i < 4, let constantData = allocation.constantData {
        // Don't materialize; instead bind the constant data.
        let byteCount = allocation.handle.byteCount
        encoder.setBytes(constantData, length: byteCount, index: 3 + i)
      } else {
        try allocation.materialize()
        let buffer = allocation.mtlBuffer!
        encoder.setBuffer(buffer, offset: 0, index: 3 + i)
        if i < 4 {
          // Allocation is an input.
          ectx.addBarrierResource(allocation, buffer: buffer)
          ectx.markAsRead(allocation)
        } else {
          // Allocation is the output.
          ectx.markAsModified(allocation)
        }
      }
    }
    ectx.memoryBarrier()
    
    // MARK: - Encode Virtual Assembly Instructions
    
    typealias DispatchParams = Instruction.Elementwise.DispatchParams
    
    let numOperations = instruction.operations.count
    var params = DispatchParams(
      numOperations: instruction.operations.count,
      inputHandle1: instruction.input1.handle,
      inputHandle2: instruction.input2?.handle,
      inputHandle3: instruction.input3?.handle,
      inputHandle4: instruction.input4?.handle,
      outputHandle: instruction.output.handle,
      usingLargeRepresentation: instruction.dataGroup == .u32_i64_u64)
    encoder.setBytes(&params, length: MemoryLayout.stride(ofValue: params), index: 0)
    
    withUnsafeTemporaryAllocation(of: UInt16.self, capacity: numOperations) { bufferPointer in
      let operations = bufferPointer.baseAddress!
      for i in 0..<numOperations {
        operations[i] = instruction.operations[i]
      }
      let length = numOperations * MemoryLayout<UInt16>.stride
      encoder.setBytes(operations, length: length, index: 1)
    }
    
    // One unit of metadata, but not exactly one operation's total allocation of metadata.
    typealias Atom = SmallVector<SIMD2<UInt64>>.Scalar
    let numMetadataAtoms = instruction.metadata.count
    withUnsafeTemporaryAllocation(of: Atom.self, capacity: numMetadataAtoms) { bufferPointer in
      let metadata = bufferPointer.baseAddress!
      for i in 0..<numMetadataAtoms {
        metadata[i] = instruction.metadata[i]
      }
      let length = numMetadataAtoms * MemoryLayout<Atom>.stride
      encoder.setBytes(metadata, length: length, index: 2)
    }
    
    var numThreads: Int
    switch instruction.dataGroup {
    case .f32_i32:
      numThreads = (instruction.size + 3) / 4
    case .u32_i64_u64:
      numThreads = (instruction.size + 1) / 2
    }
    encoder.dispatchThreadgroups(MTLSize(numThreads), threadsPerThreadgroup: 1)
    
    if Instruction.Elementwise.enableDump {
      print("~~~ DUMP START")
      print(instruction.dump())
      print("~~~ DUMP END")
    }
  }
  
  func encodeExplicitCopy(
    _ instruction: Instruction.ExplicitCopy,
    into ectx: inout EncodingContext
  ) throws {
    let input = instruction.input
    try input.materialize()
    ectx.markAsRead(input)
    
    let output = instruction.output
    try output.materialize()
    ectx.markAsModified(output)
    
    // Use a blit encoder because this command's dominant use case is marshalling data over PCIe. I
    // don't know the performance characteristics of PCIe, so it's best to delegate that to Apple's
    // optimized blit commands. If it were just copying between GPU memory regions, a specialized
    // compute shader would be just as fast and have less CPU-side overhead.
    let blitEncoder = ectx.makeBlitCommandEncoder()
    
    blitEncoder.copy(
      from: input.mtlBuffer!, sourceOffset: 0, to: output.mtlBuffer!, destinationOffset: 0,
      size: instruction.byteCount)
  }
}
