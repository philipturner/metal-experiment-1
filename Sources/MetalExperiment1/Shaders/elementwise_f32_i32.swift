//
//  elementwise_f32_i32.swift
//  
//
//  Created by Philip Turner on 8/3/22.
//

import Darwin

// Swift analogue of the `elementwise_f32_i32` ubershader.
struct Swift_elementwise_f32_i32 {
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
  
  let params: Instruction.Elementwise.DispatchParams
  let operation: UInt16
  let metadata: UInt64
  var input1: UnsafeRawPointer
  var input2: UnsafeRawPointer?
  var input3: UnsafeRawPointer?
  var output: UnsafeMutableRawPointer
  
  mutating func execute() {
    var register1: UInt32 = 0
    var register2: UInt32 = 0
    var register3: UInt32 = 0
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
      var compressed: UInt32
      switch read_params.layout {
      case 128 + 1:
        let rebound = input.assumingMemoryBound(to: UInt8.self)
        compressed = UInt32(rebound.pointee)
      case 128 + 2:
        let rebound = input.assumingMemoryBound(to: UInt16.self)
        compressed = UInt32(rebound.pointee)
      case 128 + 4:
        let rebound = input.assumingMemoryBound(to: UInt32.self)
        compressed = UInt32(rebound.pointee)
      default:
        fatalError("Layout '\(read_params.layout)' was not a scalar broadcast.")
      }
      
      var expanded: UInt32
      switch MemoryCast(rawValue: read_params.memory_cast)! {
      case .f32_i32_native:
        expanded = compressed
      case .f16_as_f32:
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let `in` = Float16(bitPattern: .init(truncatingIfNeeded: compressed))
        expanded = Float(`in`).bitPattern
        #else
        fatalError("'Float16' not supported yet.")
        #endif
      case .i8_as_i32:
        let `in` = Int8(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .i16_as_i32:
        let `in` = Int16(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .u8_as_i32:
        let `in` = UInt8(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .u16_as_i32:
        let `in` = UInt16(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
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
      let operation = UnaryOperationType(rawValue: self.operation)!
      func compares(_ other: UnaryOperationType) -> Bool {
        operation.rawValue <= other.rawValue
      }
      
      @inline(__always)
      func withFloat(_ body: (Float) -> Float) {
        let input = Float(bitPattern: register1)
        let output = body(input)
        register1 = output.bitPattern
      }
      @inline(__always)
      func withInt32(_ body: (Int32) -> Int32) {
        let input = Int32(bitPattern: register1)
        let output = body(input)
        register1 = UInt32(bitPattern: output)
      }
      
      if compares(.atan_f32) {
        switch operation {
        case .abs_f32:
          withFloat(abs)
        case .abs_i32:
          withInt32(abs)
        case .acos_f32:
          withFloat(acos)
        case .acosh_f32:
          withFloat(acosh)
        case .asin_f32:
          withFloat(asin)
        case .asinh_f32:
          withFloat(asinh)
        case .atan_f32:
          withFloat(atan)
        case .atanh_f32:
          withFloat(atanh)
        default: fatalError()
        }
      } else if compares(.cast_i32_to_i32) {
        switch operation {
        case .cast_f32_to_f16:
          #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
          withFloat { Float(Float16($0)) }
          #else
          fatalError("'Float16' not supported yet.")
          #endif
        case .cast_f32_to_bool:
          // Do not call `withFloat`; that stores `Float(1).bitPattern` instead of `UInt32(1)`.
          let x = Float(bitPattern: register1)
          register1 = (x != 0) ? 1 : 0
        case .cast_f32_to_i32:
          let x = Float(bitPattern: register1)
          let bounds = unsafeBitCast(metadata, to: SIMD2<Int32>.self)
          var casted = Int32(x)
          
          casted = max(casted, /*lower bound*/bounds[0])
          casted = min(casted, /*upper bound*/bounds[1])
          register1 = UInt32(bitPattern: casted)
        case .cast_i32_to_f16:
          #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
          let x = Int32(bitPattern: register1)
          register1 = Float(Float16(x)).bitPattern
          #else
          fatalError("'Float16' not supported yet.")
          #endif
        case .cast_i32_to_f32:
          let x = Int32(bitPattern: register1)
          register1 = Float(x).bitPattern
        case .cast_i32_to_bool:
          withInt32 { ($0 != 0) ? 1 : 0 }
        case .cast_i32_to_i32:
          withInt32 {
            let masks = unsafeBitCast(metadata, to: SIMD2<Int32>.self)
            var x = $0 & masks[0] // truncate
            
            if masks[1] != 0 { // sign extend
              let inverted_mask = ~masks[0]
              if (x & masks[1]) != 0 {
                x = x | inverted_mask
              }
            }
            return x
          }
        default: fatalError()
        }
      } else if compares(.floor_f32) {
        switch operation {
        case .ceil_f32:
          withFloat(ceil)
        case .cos_f32:
          withFloat(cos)
        case .cosh_f32:
          withFloat(cosh)
        case .elu_f32:
          withFloat { x in
            (x < 0) ? expm1(x) : x
          }
        case .exp_f32:
          withFloat(exp)
        case .expm1_f32:
          withFloat(expm1)
        case .floor_f32:
          withFloat(floor)
        default: fatalError()
        }
      } else if compares(.is_nan_f32) {
        switch operation {
        case .is_finite_f32:
          let x = Float(bitPattern: register1)
          register1 = x.isFinite ? 1 : 0
        case .is_inf_f32:
          let x = Float(bitPattern: register1)
          register1 = x.isInfinite ? 1 : 0
        case .is_nan_f32:
          let x = Float(bitPattern: register1)
          register1 = x.isNaN ? 1 : 0
        default: fatalError()
        }
      } else if compares(.round_f32) {
        switch operation {
        case .leaky_relu_f32:
          withFloat { x in
            let alpha = unsafeBitCast(metadata, to: SIMD2<Float>.self)[0]
            return max(x, x * alpha)
          }
        case .log_f32:
          withFloat(log)
        case .log1p_f32:
          withFloat(log1p)
        case .logical_not_bool:
          withInt32 { x in
            let casted = x != 0
            return (!casted) ? 1 : 0
          }
        case .neg_f32:
          withFloat(-)
        case .neg_i32:
          withInt32(-)
        case .relu_f32:
          withFloat { x in
            max(0, x)
          }
        case .relu6_f32:
          withFloat {
            var x = $0
            x = max(x, /*lower bound*/0)
            x = min(x, /*lower bound*/6)
            return x
          }
        case .round_f32:
          withFloat(rint)
        default: fatalError()
        }
      } else if compares(.softplus_f32) {
        switch operation {
        case .rsqrt_f32:
          withFloat { 1 / sqrt($0) }
        case .selu_f32:
          withFloat { x in
            let alpha: Float = 1.6732632423543772848170429916717
            let scale: Float = 1.0507009873554804934193349852946
            return (x < 0) ? (scale * alpha * expm1(x)) : (scale * x)
          }
        case .sigmoid_f32:
          withFloat {
            var x = $0
            x = 1 + exp(-x)
            x = 1 / x
            return x
          }
        case .sign_f32:
          withFloat { x in
            if x > 0 {
              return 1
            } else if x < 0 {
              return -1
            } else /*x == 0*/ {
              return 0
            }
          }
        case .sign_i32:
          withInt32 { x in
            if x > 0 {
              return 1
            } else if x < 0 {
              return -1
            } else /*x == 0*/ {
              return 0
            }
          }
        case .sin_f32:
          withFloat(sin)
        case .sinh_f32:
          withFloat(sinh)
        case .softplus_f32:
          withFloat {
            var x = $0
            x = exp(x) + 1
            x = log(x)
            return x
          }
        default: fatalError()
        }
      } else if compares(.tanh_f32) {
        switch operation {
        case .softsign_f32:
          withFloat {
            var x = $0
            let denominator = abs(x) + 1
            x = 1 / denominator
            return x
          }
        case .sqrt_f32:
          withFloat(sqrt)
        case .square_f32:
          withFloat { $0 * $0 }
        case .square_i32:
          withInt32 { $0 * $0 }
        case .tan_f32:
          withFloat(tan)
        case .tanh_f32:
          withFloat(tanh)
        default: fatalError()
        }
      } else if compares(.scalar_div_inverse_f32) {
        var x = Float(bitPattern: register1)
        let scalar = unsafeBitCast(metadata, to: SIMD2<Float>.self)[0]
        
        switch operation {
        case .scalar_add_f32:
          x += scalar
        case .scalar_sub_f32:
          x -= scalar
        case .scalar_sub_inverse_f32:
          x = scalar - x
        case .scalar_mul_f32:
          x *= scalar
        case .scalar_div_f32:
          x /= scalar
        case .scalar_div_inverse_f32:
          x = scalar / x
        default: fatalError()
        }
        register1 = x.bitPattern
      } else if compares(.scalar_div_inverse_i32) {
        var x = Int32(bitPattern: register1)
        let scalar = unsafeBitCast(metadata, to: SIMD2<Int32>.self)[0]
        
        switch (operation) {
        case .scalar_add_i32:
          x += scalar
        case .scalar_sub_i32:
          x -= scalar
        case .scalar_sub_inverse_i32:
          x = scalar - x
        case .scalar_mul_i32:
          x *= scalar
        case .scalar_div_i32:
          x /= scalar
        case .scalar_div_inverse_i32:
          x = scalar / x
        default: fatalError()
        }
        register1 = UInt32(bitPattern: x)
      } else if operation == .no_op {
        fatalError("Should never execute constant folding if it's a no-op.")
      }
    } else {
      // TODO: Subtract 1000 from the operation.
      // MARK: - Binary
      _ = register2
      _ = register3
    }
    
    switch MemoryCast(rawValue: params.write_memory_cast)! {
    case .f32_i32_native:
      let mem_slice = register1
      let casted = output.assumingMemoryBound(to: UInt32.self)
      casted.pointee = mem_slice
    case .f16_as_f32:
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let mem_slice = Float16(Float(bitPattern: register1)).bitPattern
      let casted = output.assumingMemoryBound(to: UInt16.self)
      casted.pointee = mem_slice
      #else
      fatalError("'Float16' not supported yet.")
      #endif
    case .i8_as_i32, .u8_as_i32:
      let mem_slice = UInt8(truncatingIfNeeded: register1)
      let casted = output.assumingMemoryBound(to: UInt8.self)
      casted.pointee = mem_slice
    case .i16_as_i32, .u16_as_i32:
      let mem_slice = UInt16(truncatingIfNeeded: register1)
      let casted = output.assumingMemoryBound(to: UInt16.self)
      casted.pointee = mem_slice
    }
  }
}
