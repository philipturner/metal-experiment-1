//
//  elementwise_u32_i64_u64.swift
//  
//
//  Created by Philip Turner on 8/3/22.
//

import Darwin

// Swift analogue of the `elementwise_u32_i64_u64` ubershader.
struct Swift_elementwise_u32_i64_u64 {
  enum MemoryCast: UInt16, RawRepresentable {
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
  
  let params: Instruction.Elementwise.DispatchParams
  let operation: UInt16
  let metadata: UInt64
  var input1: UnsafeRawPointer
  var input2: UnsafeRawPointer?
  var input3: UnsafeRawPointer?
  var output: UnsafeMutableRawPointer
  
  mutating func execute() {
    var register1: UInt64 = 0
    var register2: UInt64 = 0
    var register3: UInt64 = 0
    for i in 0..<params.num_inputs {
      var read_params: Instruction.Elementwise.ReadParams
      var input: UnsafeRawPointer
      switch i {
      case 0:
        read_params = params.read_param_1
        input = input1
      case 1:
        read_params = params.read_param_2
        input = input2.unsafelyUnwrapped
      default: /*2*/
        read_params = params.read_param_3
        input = input3.unsafelyUnwrapped
      }
      
      // Currently processes one scalar, which must be encoded as a scalar broadcast.
      var compressed: UInt64
      switch read_params.layout {
      case 128 + 1:
        let rebound = input.assumingMemoryBound(to: UInt8.self)
        compressed = UInt64(rebound.pointee)
      case 128 + 2:
        let rebound = input.assumingMemoryBound(to: UInt16.self)
        compressed = UInt64(rebound.pointee)
      case 128 + 4:
        let rebound = input.assumingMemoryBound(to: UInt32.self)
        compressed = UInt64(rebound.pointee)
      case 128 + 8:
        let rebound = input.assumingMemoryBound(to: UInt64.self)
        compressed = UInt64(rebound.pointee)
      default:
        fatalError("Layout '\(read_params.layout)' was not a scalar broadcast.")
      }
      
      var expanded: UInt64
      switch MemoryCast(rawValue: read_params.memory_cast)! {
      case .i64_u64_native:
        expanded = compressed
      case .i32_as_i64:
        let `in` = Int32(truncatingIfNeeded: compressed)
        expanded = UInt64(bitPattern: Int64(`in`))
      case .i16_as_i64:
        let `in` = Int16(truncatingIfNeeded: compressed)
        expanded = UInt64(bitPattern: Int64(`in`))
      case .i8_as_i64:
        let `in` = Int8(truncatingIfNeeded: compressed)
        expanded = UInt64(bitPattern: Int64(`in`))
        
      case .u32_as_i64:
        expanded = UInt64(UInt32(truncatingIfNeeded: compressed))
      case .u16_as_i64:
        expanded = UInt64(UInt16(truncatingIfNeeded: compressed))
      case .u8_as_i64:
        expanded = UInt64(UInt8(truncatingIfNeeded: compressed))
      case .f32_padded:
        var `in` = unsafeBitCast(compressed, to: SIMD2<Float>.self)
        `in`[1] = 0
        expanded = unsafeBitCast(`in`, to: UInt64.self)
      case .f16_as_f32_padded:
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let `in` = unsafeBitCast(compressed, to: SIMD4<Float16>.self)
        let casted = SIMD2<Float>(Float(`in`[0]), 0)
        expanded = unsafeBitCast(casted, to: UInt64.self)
        #else
        fatalError("'Float16' not supported yet.")
        #endif
      }
      
      switch i {
      case 0:
        register1 = expanded
      case 1:
        register2 = expanded
      default: /*2*/
        register3 = expanded
      }
    }
    
    // TODO: Start off by only constant folding unary.
    if self.operation < 1000 {
      // MARK: - Unary
      let operation = UnaryOperationType2(rawValue: self.operation)!
      func compares(_ other: UnaryOperationType2) -> Bool {
        operation.rawValue <= other.rawValue
      }
      
      @inline(__always)
      func withInt64(_ body: (Int64) -> Int64) {
        let input = Int64(bitPattern: register1)
        let output = body(input)
        register1 = UInt64(bitPattern: output)
      }
      @inline(__always)
      func withUInt64(_ body: (UInt64) -> UInt64) {
        register1 = body(register1)
      }
      @inline(__always)
      func getFloat() -> Float {
        unsafeBitCast(register1, to: SIMD2<Float>.self)[0]
      }
      @inline(__always)
      func setFloat(_ input: Float) {
        let vector = SIMD2<Float>(input, 0)
        register1 = unsafeBitCast(vector, to: UInt64.self)
      }
      
      if compares(.square_u64) {
        switch operation {
        case .abs_i64:
          withInt64(abs)
        case .neg_i64:
          withInt64(-)
        case .sign_i64:
          withInt64 { x in
            if x > 0 {
              return 1
            } else if x < 0 {
              return -1
            } else /*x == 0*/ {
              return 0
            }
          }
        case .sign_u64:
          withInt64 { x in
            if x > 0 {
              return 1
            } else /*x == 0*/ {
              return 0
            }
          }
        case .square_i64:
          print("Square i64")
          withInt64 { $0 * $0 }
        case .square_u64:
          withUInt64 { $0 * $0 }
        default: fatalError()
        }
      } else if compares(.cast_i64_u64_to_bool) {
        switch operation {
        case .cast_f32_to_u32:
          let x = getFloat()
          let casted = Int64(UInt32(x))
          register1 = UInt64(bitPattern: casted)
        case .cast_f32_to_i64:
          let x = getFloat()
          let casted = Int64(x)
          register1 = UInt64(bitPattern: casted)
        case .cast_i64_to_f16:
          #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
          let x = Int64(bitPattern: register1)
          let casted = Float(Float16(x))
          setFloat(casted)
          #else
          fatalError("'Float16' not supported yet.")
          #endif
        case .cast_i64_to_f32:
          let x = Int64(bitPattern: register1)
          let casted = Float(x)
          setFloat(casted)
        case .cast_i64_u64_to_bool:
          let x = Int64(bitPattern: register1)
          register1 = (x != 0) ? 1 : 0
        default: fatalError()
        }
      } else if compares(.cast_i64_u64_to_u32) {
        switch operation {
        case .cast_f32_to_u64:
          let x = getFloat()
          register1 = UInt64(x)
        case .cast_u64_to_f16:
          #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
          let x: UInt64 = register1
          let casted = Float(Float16(x))
          setFloat(casted)
          #else
          fatalError("'Float16' not supported yet.")
          #endif
        case .cast_u64_to_f32:
          let x: UInt64 = register1
          let casted = Float(x)
          setFloat(casted)
        default: /*.cast_i64_u64_to_i32,
                   .cast_i64_u64_to_u32*/
          var x: UInt64 = register1
          let mask: UInt64 = metadata
          x &= mask // truncate
          
          if operation == .cast_i64_u64_to_i32 { // sign extend
            let sign_mask = mask ^ (mask >> 1)
            let inverted_mask = ~mask
            if (x & sign_mask) != 0 {
              x |= inverted_mask
            }
          }
          register1 = x
        }
      } else if compares(.scalar_div_inverse_i64) {
        var x = Int64(bitPattern: register1)
        let scalar = Int64(bitPattern: metadata)
        
        switch operation {
        case .scalar_add_i64_u64:
          x += scalar
        case .scalar_sub_i64_u64:
          x -= scalar
        case .scalar_sub_inverse_i64_u64:
          x = scalar - x
        case .scalar_mul_i64:
          x *= scalar
        case .scalar_div_i64:
          x /= scalar
        case .scalar_div_inverse_i64:
          x = scalar / x
        default: fatalError()
        }
        register1 = UInt64(bitPattern: x)
      } else if compares(.scalar_div_inverse_u64) {
        var x: UInt64 = register1
        let scalar: UInt64 = metadata
        
        switch operation {
        case .scalar_mul_u64:
          x *= scalar
        case .scalar_div_u64:
          x /= scalar
        case .scalar_div_inverse_u64:
          x = scalar / x
        default: fatalError()
        }
        register1 = x
      } else {
        // Never execute constant folding if it's a no-op.
        fatalError("This should never happen.")
      }
    } else {
      // TODO: Subtract 1000 from the operation.
      // MARK: - Binary
      _ = register2
      _ = register3
    }
    
    switch MemoryCast(rawValue: params.write_memory_cast)! {
    case .i64_u64_native:
      let mem_slice = register1
      let casted = output.assumingMemoryBound(to: UInt64.self)
      casted.pointee = mem_slice
    case .i32_as_i64,
         .u32_as_i64:
      let mem_slice = UInt32(truncatingIfNeeded: register1)
      let casted = output.assumingMemoryBound(to: UInt32.self)
      casted.pointee = mem_slice
    case .i16_as_i64,
         .u16_as_i64:
      let mem_slice = UInt16(truncatingIfNeeded: register1)
      let casted = output.assumingMemoryBound(to: UInt16.self)
      casted.pointee = mem_slice
    case .i8_as_i64,
         .u8_as_i64:
      let mem_slice = UInt8(truncatingIfNeeded: register1)
      let casted = output.assumingMemoryBound(to: UInt8.self)
      casted.pointee = mem_slice
    case .f32_padded:
      let vector = unsafeBitCast(register1, to: SIMD2<Float>.self)
      let mem_slice = Float(vector[0]).bitPattern
      let casted = output.assumingMemoryBound(to: UInt32.self)
      casted.pointee = mem_slice
    case .f16_as_f32_padded:
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let vector = unsafeBitCast(register1, to: SIMD2<Float>.self)
      let mem_slice = Float16(vector[0]).bitPattern
      let casted = output.assumingMemoryBound(to: UInt16.self)
      casted.pointee = mem_slice
      #else
      fatalError("'Float16' not supported yet.")
      #endif
    }
  }
}
