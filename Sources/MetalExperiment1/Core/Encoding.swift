//
//  Encoding.swift
//  
//
//  Created by Philip Turner on 7/18/22.
//

import MetalPerformanceShaders

struct EncodingContext {
  let commandBuffer: MTLCommandBuffer
  let commandBufferID: Int
  private var encoder: MTLComputeCommandEncoder?
  
  init(commandBuffer: MTLCommandBuffer, commandBufferID: Int) {
    self.commandBuffer = commandBuffer
    self.commandBufferID = commandBufferID
  }
  
  @inline(__always)
  mutating func makeEncoder() -> MTLComputeCommandEncoder {
    if let encoder = encoder {
      return encoder
    } else {
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      self.encoder = encoder
      return encoder
    }
  }
  
  @inline(__always)
  mutating func finishEncoder() {
    if let encoder = encoder {
      encoder.endEncoding()
      self.encoder = nil
    }
  }
}

extension Context {
  func encodeCompiledOperation(
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

private extension Context {
  func encodeElementwise(
    _ instruction: Instruction.Elementwise,
    into ectx: inout EncodingContext
  ) throws {
    let encoder = ectx.makeEncoder()
    switch instruction.dataGroup {
    case .f32_i32:
      encoder.setComputePipelineState(ShaderCache.elementwise_f32_i32)
    case .u32_i64_u64:
      encoder.setComputePipelineState(ShaderCache.elementwise_u32_i64_u64)
    }
    
    let input1 = instruction.input1
    try input1.materialize()
    encoder.setBuffer(input1.mtlBuffer!, offset: 0, index: 3)
    
    if let input2 = instruction.input2 {
      try input2.materialize()
      encoder.setBuffer(input2.mtlBuffer!, offset: 0, index: 4)
    }
    
    if let input3 = instruction.input3 {
      try input3.materialize()
      encoder.setBuffer(input3.mtlBuffer!, offset: 0, index: 5)
    }
    
    let output = instruction.output
    try output.materialize()
    output.lastModifiedCommandBufferID = ectx.commandBufferID
    encoder.setBuffer(output.mtlBuffer!, offset: 0, index: 6)
    
    enum MemoryCast: UInt16, RawRepresentable {
      case f32_i32_native = 0
      case f16_as_f32 = 1
      case i8_as_i32 = 2
      case i16_as_i32 = 3
      case u8_as_i32 = 4
      case u16_as_i32 = 5
      
      @inline(__always)
      init(dataType: DataType) {
        switch dataType {
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
        case .uint32, .int64, .uint64:
          fatalError("'unary_f32_i32' does not support data type '\(dataType)'.")
        }
      }
      
      @inline(__always)
      var read_size: UInt16 {
        switch self {
        case .f32_i32_native: return 4
        case .f16_as_f32: return 2
        case .i8_as_i32: return 1
        case .i16_as_i32: return 2
        case .u8_as_i32: return 1
        case .u16_as_i32: return 2
        }
      }
    }
    
    // The memory casts must be stored as explicit raw values, instead of Swift enums. Doing the
    // latter makes Metal code run incorrectly. Swift probably doesn't store an `enum` in memory
    // using its exact raw value.
    struct DispatchParams {
      var read_scalar_broadcast: Bool
      var read_size: UInt16
      var read_memory_cast: MemoryCast.RawValue
      var num_operations: UInt16
      var write_memory_cast: MemoryCast.RawValue
      
      init(inputDataType: DataType, numOperations: Int, outputDataType: DataType) {
        let read_memory_cast = MemoryCast(dataType: inputDataType)
        let write_memory_cast = MemoryCast(dataType: outputDataType)
        self.read_scalar_broadcast = false
        self.read_memory_cast = read_memory_cast.rawValue
        self.num_operations = UInt16(numOperations)
        self.write_memory_cast = write_memory_cast.rawValue
        self.read_size = read_memory_cast.read_size
      }
    }
    
    let numOperations = instruction.operations.count
    var params = DispatchParams(
      inputDataType: input1.handle.dataType, numOperations: numOperations,
      outputDataType: output.handle.dataType)
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
  }
  
  func encodeExplicitCopy(
    _ instruction: Instruction.ExplicitCopy,
    into ectx: inout EncodingContext
  ) throws {
    let input = instruction.input
    try input.materialize()
    let output = instruction.output
    try output.materialize()
    output.lastModifiedCommandBufferID = ectx.commandBufferID
    
    // Use a blit encoder because this command's dominant use case is marshalling data over PCIe. I
    // don't know the performance characteristics of PCIe, so it's best to delegate that to Apple's
    // optimized blit commands. If it were just copying between GPU memory regions, a specialized
    // compute shader would be just as fast and have less CPU-side overhead.
    //
    // I'm also not (yet) making optimized encoding utilities like `makeBlitEncoder` and
    // `finishBlitEncoder` just for this command. There is a very low chance that two explicit copy
    // operations appear next to each other. This is a viable optimization that I could pursue down
    // the road, alongside other enhancements to reading data from the GPU.
    ectx.finishEncoder()
    let blitEncoder = ectx.commandBuffer.makeBlitCommandEncoder()!
    defer {
      blitEncoder.endEncoding()
    }
    
    blitEncoder.copy(
      from: input.mtlBuffer!, sourceOffset: 0, to: output.mtlBuffer!, destinationOffset: 0,
      size: instruction.byteCount)
  }
}
