//
//  Instruction.swift
//  
//
//  Created by Philip Turner on 7/30/22.
//

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum Instruction {
  struct Elementwise {
    // `metadata` has much less vector capacity of `operations`. It doesn't need as much storage
    // because it's serialized efficiently. Metadata is only recorded after each operation that
    // needs it.
    var operations: SmallVector<SIMD8<UInt16>>
    
    // Warning: `SIMD2` does not mean 2 operations worth of metadata. It means the total capacity
    // for metadata is 16, which happens to be (2 operations) * (8 bytes/operation). The rationing
    // of metadata per operation is subject to change.
    var metadata: SmallVector<SIMD2<UInt64>>
    var dataGroup: DataGroup
    
    // Debug info for reconstructing context when fusing non-adjacent operations.
    var numFusedUnaryOperations: UInt16
    var numFusedNonUnaryOperations: UInt16
    
    // `input1`, `output`, and `dataGroup` are nullable to permit efficiently reconstructing the
    // context when
    // fusing non-adjacent operations.
    var input1: Allocation!
    var input2: Allocation?
    var input3: Allocation?
    var input4: Allocation?
    var output: Allocation!
    var size: Int
  }
  case elementwise(Elementwise)
  
  struct ExplicitCopy {
    var input: Allocation
    var output: Allocation
    var byteCount: Int
  }
  case explicitCopy(ExplicitCopy)
}

extension Instruction.Elementwise {
  static var enableDump = false
  
  func dump() -> String {
    func dumpRead(register: Int) -> String {
      "var reg\(register) = input\(register)[i]"
    }
    
    func dumpWrite() -> String {
      "output[i] = reg1"
    }
    
    func dumpUnary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = UnaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = UnaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1)"
    }
    
    func dumpBinary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = BinaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = BinaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1, reg2)"
    }
    
    func dumpTernary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = TernaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = TernaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1, reg2, reg3)"
    }
    
    func dumpSwap(code: UInt16) -> String {
      let swapType = RegisterSwapType(rawValue: code)!
      switch swapType {
      case .swap_registers_1_2:
        return "swap(&reg1, &reg2)"
      case .swap_registers_1_3:
        return "swap(&reg1, &reg3)"
      case .swap_registers_1_4:
        return "swap(&reg1, &reg4)"
      case .swap_registers_2_3:
        return "swap(&reg2, &reg3)"
      case .swap_registers_2_4:
        return "swap(&reg2, &reg4)"
      case .swap_registers_3_4:
        return "swap(&reg3, &reg4)"
      }
    }
    
    var output: [String] = []
    output.append(dumpRead(register: 1))
    if input2 != nil {
      output.append(dumpRead(register: 2))
    }
    if input3 != nil {
      output.append(dumpRead(register: 3))
    }
    if input4 != nil {
      output.append(dumpRead(register: 4))
    }
    
    for i in 0..<operations.count {
      let code = operations[i]
      if code < 1000 {
        output.append(dumpUnary(code: code - 0))
      } else if code < 2000 {
        output.append(dumpBinary(code: code - 1000))
      } else if code < 3000 {
        output.append(dumpTernary(code: code - 2000))
      } else if code < 4000 {
        output.append(dumpSwap(code: code - 3000))
      }
    }
    output.append(dumpWrite())
    return output.joined(separator: "\n")
  }
}
