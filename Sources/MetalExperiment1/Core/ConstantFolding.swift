//
//  ConstantFolding.swift
//  
//
//  Created by Philip Turner on 7/30/22.
//

import Darwin

// Create the EagerOperation.Unary object in the calling function. Either pass it to the operation
// queue or pass it into this function.

fileprivate typealias DispatchParams = Instruction.Elementwise.DispatchParams

extension MTLPluggableDevice {
  @inline(__always)
  func constantFold(_ unary: EagerOperation.Unary) {
    if unary.dataGroup == .f32_i32 {
      let op = UnaryOperationType(rawValue: unary.operation)!
      print("Constant folded \(op)")
    } else {
      let op = UnaryOperationType2(rawValue: unary.operation)!
      print("Constant folded \(op)")
    }
    
    constantFold(
      operation: unary.operation,
      input1: unary.input,
      input2: nil,
      input3: nil,
      output: unary.output,
      dataGroup: unary.dataGroup,
      metadata: unary.metadata ?? 0)
  }
  
  // TODO: Add 1000 to operation for binary constant folding.
  
  @inline(never)
  func constantFold(
    operation: UInt16,
    input1: AllocationHandle,
    input2: AllocationHandle?,
    input3: AllocationHandle?,
    output: AllocationHandle,
    dataGroup: DataGroup,
    metadata: UInt64
  ) {
    let outputAllocation = output.reference!.takeUnretainedValue()
    outputAllocation.initializeConstantData { _ in }
    let outputData = outputAllocation.constantData!
    
    if operation == .max {
      // Do not execute no-ops.
      return
    }
    
    let params = DispatchParams(
      numOperations: 1,
      inputHandle1: input1,
      inputHandle2: input2,
      inputHandle3: input3,
      inputHandle4: nil,
      outputHandle: output,
      usingLargeRepresentation: dataGroup == .u32_i64_u64)
    
    let inputData1 = input1.reference!.takeUnretainedValue().constantData!
    let inputData2 = input2?.reference!.takeUnretainedValue().constantData!
    let inputData3 = input3?.reference!.takeUnretainedValue().constantData!
    
    if dataGroup == .f32_i32 {
      var elementwise = Swift_elementwise_f32_i32(
        params: params,
        operation: operation,
        metadata: metadata,
        input1: inputData1,
        input2: inputData2,
        input3: inputData3,
        output: outputData)
      elementwise.execute()
    } else {
      var elementwise = Swift_elementwise_u32_i64_u64(
        params: params,
        operation: operation,
        metadata: metadata,
        input1: inputData1,
        input2: inputData2,
        input3: inputData3,
        output: outputData)
      elementwise.execute()
    }
  }
}

extension Instruction.Elementwise {
  struct ReadParams {
    var layout: UInt16
    var memory_cast: UInt16
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
    var write_memory_cast: UInt16
    
    @inline(never)
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
      var readMemoryCasts = SIMD4<Swift_elementwise_f32_i32.MemoryCast.RawValue>(repeating: 0)
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
          let memoryCast = Swift_elementwise_u32_i64_u64.MemoryCast(dataTypeRawValue: dataType)
          memoryCastRawValue = memoryCast.rawValue
          layoutMask = memoryCast.readSize
        } else {
          let memoryCast = Swift_elementwise_f32_i32.MemoryCast(dataTypeRawValue: dataType)
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
}
