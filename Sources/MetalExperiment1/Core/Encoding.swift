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
  
  @inline(__always)
  mutating func useAllocation(_ allocation: Allocation) {
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
      
      try allocation.materialize()
      let buffer = allocation.mtlBuffer!
      encoder.setBuffer(buffer, offset: 0, index: 3 + i)
      if i < 4 {
        // Allocation is an input.
        ectx.addBarrierResource(allocation, buffer: buffer)
      } else {
        // Allocation is the output.
        ectx.useAllocation(allocation)
      }
    }
    ectx.memoryBarrier()
    
    enum MemoryCast: UInt16, RawRepresentable {
      case f32_i32_native = 0
      case f16_as_f32 = 1
      case i8_as_i32 = 2
      case i16_as_i32 = 3
      case u8_as_i32 = 4
      case u16_as_i32 = 5
      
      @inline(__always)
      init(dataTypeRawValue: UInt16) {
        let dataType = DataType(rawValue: dataTypeRawValue)
        switch dataType.unsafelyUnwrapped {
        case .float16:
          self = .f16_as_f32
        case .float32:
          self = .f32_i32_native
        case .bool:
          self = .u8_as_i32
        case .int8:
          self = .i8_as_i32
        case .int16:
          self = .i16_as_i32
        case .int32:
          self = .f32_i32_native
        case .uint8:
          self = .u8_as_i32
        case .uint16:
          self = .u16_as_i32
        default:
          let description = dataType?.description ?? "invalid"
          fatalError("'unary_f32_i32' does not support data type '\(description)'.")
        }
      }
      
      @inline(__always)
      var readSize: UInt16 {
        switch self {
        case .f32_i32_native: return 4
        case .f16_as_f32: return 2
        case .i8_as_i32, .u8_as_i32: return 1
        case .i16_as_i32, .u16_as_i32: return 2
        }
      }
    }
    
    enum MemoryCast2: UInt16, RawRepresentable {
      case i64_u64_native = 0
      case i32_as_i64 = 1
      case i16_as_i64 = 2
      case i8_as_i64 = 3
      
      case u32_as_i64 = 4
      case u16_as_i64 = 5
      case u8_as_i64 = 6
      case f32_padded = 7
      case f16_as_f32_padded = 8
      
      @inline(__always)
      init(dataTypeRawValue: UInt16) {
        let dataType = DataType(rawValue: dataTypeRawValue)
        switch dataType.unsafelyUnwrapped {
        case .float16:
          self = .f16_as_f32_padded
        case .float32:
          self = .f32_padded
        case .bool:
          self = .u8_as_i64
        case .int8:
          self = .i8_as_i64
        case .int16:
          self = .i16_as_i64
        case .int32:
          self = .i32_as_i64
        case .int64:
          self = .i64_u64_native
        case .uint8:
          self = .u8_as_i64
        case .uint16:
          self = .u16_as_i64
        case .uint32:
          self = .u32_as_i64
        case .uint64:
          self = .i64_u64_native
        }
      }
      
      @inline(__always)
      var readSize: UInt16 {
        switch self {
        case .i64_u64_native: return 8
        case .i32_as_i64, .u32_as_i64: return 4
        case .i16_as_i64, .u16_as_i64: return 2
        case .i8_as_i64, .u8_as_i64: return 1
        case .f32_padded: return 4
        case .f16_as_f32_padded: return 2
        }
      }
    }
    
    struct ReadParams {
      var layout: UInt16
      var memory_cast: MemoryCast.RawValue
    }
    
    // The memory casts must be stored as explicit raw values, instead of Swift enums. Doing the
    // latter makes Metal code run incorrectly. Swift probably doesn't store an `enum` in memory
    // using its exact raw value.
    struct DispatchParams {
      var read_param_1: ReadParams
      var read_param_2: ReadParams
      var read_param_3: ReadParams
      var read_param_4: ReadParams
      var num_inputs: UInt16
      var num_operations: UInt16
      var write_memory_cast: MemoryCast.RawValue
      
      init(
        numOperations: Int,
        inputHandle1: AllocationHandle,
        inputHandle2: AllocationHandle?,
        inputHandle3: AllocationHandle?,
        inputHandle4: AllocationHandle?,
        outputHandle: AllocationHandle,
        usingLargeRepresentation: Bool
      ) {
        var readDataTypes = SIMD4<UInt16>(repeating: .max)
        var readByteCounts = SIMD4<Int32>(repeating: 0)
        readDataTypes[0] = inputHandle1.dataType.rawValue
        readByteCounts[0] = Int32(clamping: inputHandle1.byteCount)
        
        var numInputs: UInt16 = 1
        if let inputHandle2 = inputHandle2 {
          readDataTypes[1] = inputHandle2.dataType.rawValue
          readByteCounts[1] = Int32(clamping: inputHandle2.byteCount)
          numInputs = 2
        }
        if let inputHandle3 = inputHandle3 {
          // Catch compiler bugs.
          precondition(numInputs == 2, "Input 2 is missing.")
          readDataTypes[2] = inputHandle3.dataType.rawValue
          readByteCounts[2] = Int32(clamping: inputHandle3.byteCount)
          numInputs = 3
        }
        if let inputHandle4 = inputHandle4 {
          // Catch compiler bugs.
          precondition(numInputs == 3, "Input \(numInputs + 1) is missing.")
          readDataTypes[3] = inputHandle4.dataType.rawValue
          readByteCounts[3] = Int32(clamping: inputHandle4.byteCount)
          numInputs = 4
        }
        let writeDataType = outputHandle.dataType.rawValue
        
        var readLayouts = SIMD4<UInt16>(repeating: 0)
        var readMemoryCasts = SIMD4<MemoryCast.RawValue>(repeating: 0)
        var writeMemoryCast: UInt16 = 0
        for i in 0..<5 {
          var dataType: UInt16
          if i == 4 {
            dataType = writeDataType
          } else {
            dataType = readDataTypes[i]
          }
          guard dataType != .max else {
            continue
          }
          
          var memoryCastRawValue: UInt16
          var layoutMask: UInt16
          if usingLargeRepresentation {
            let memoryCast = MemoryCast2(dataTypeRawValue: dataType)
            memoryCastRawValue = memoryCast.rawValue
            layoutMask = memoryCast.readSize
          } else {
            let memoryCast = MemoryCast(dataTypeRawValue: dataType)
            memoryCastRawValue = memoryCast.rawValue
            layoutMask = memoryCast.readSize
          }
          if i == 4 {
            writeMemoryCast = memoryCastRawValue
            break
          } else {
            readMemoryCasts[i] = memoryCastRawValue
          }
          
          let stride = DataType(rawValue: dataType).unsafelyUnwrapped.stride
          if stride == readByteCounts[i] {
            layoutMask |= 128
          }
          readLayouts[i] = layoutMask
        }
        
        // 2nd, 3rd, 4th read params can be invalid, as long as they aren't read from.
        self.read_param_1 = .init(layout: readLayouts[0], memory_cast: readMemoryCasts[0])
        self.read_param_2 = .init(layout: readLayouts[1], memory_cast: readMemoryCasts[1])
        self.read_param_3 = .init(layout: readLayouts[2], memory_cast: readMemoryCasts[2])
        self.read_param_4 = .init(layout: readLayouts[3], memory_cast: readMemoryCasts[3])
        self.num_inputs = numInputs
        self.num_operations = UInt16(truncatingIfNeeded: numOperations)
        self.write_memory_cast = writeMemoryCast
      }
    }
    
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
    let output = instruction.output
    try output.materialize()
    ectx.useAllocation(output)
    
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
