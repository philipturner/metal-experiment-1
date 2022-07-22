//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

enum UnaryOperationType: UInt16 {
  case abs_f32 = 0
  case abs_i32 = 1 // integer operation
  case acos_f32 = 2
  case acosh_f32 = 3
  case asin_f32 = 4
  case asinh_f32 = 5
  case atan_f32 = 6
  case atanh_f32 = 7
  
  // TODO: - Support casting
//  case cast_f32_to_f16 = 10
//  case cast_f32_to_i32 = 11 // requires metadata
//  case cast_i32_to_bool = 12
//  case cast_i32_to_f16 = 13
//  case cast_i32_to_f32 = 14
//  case cast_i32_to_u8 = 15
//  case cast_i32_to_u16 = 16
  
  case ceil_f32 = 20
  case cos_f32 = 21
  case cosh_f32 = 22
  case elu_f32 = 23
  case exp_f32 = 24
  case expm1_f32 = 25
  case floor_f32 = 26
  
  case is_finite_f32 = 30 // returns bool
  case is_inf_f32 = 31 // returns bool
  case is_nan_f32 = 32 // returns bool
  
  case leaky_relu_f32 = 40 // requires metadata
  case log_f32 = 41
  case log1p_f32 = 42
  case logical_not_bool = 43 // boolean operation
  case neg_f32 = 44
  case neg_i32 = 45 // integer operation
  case relu_f32 = 46
  case relu6_f32 = 47
  case round_f32 = 48 // rounds to nearest even
  
  case rsqrt_f32 = 50
  case selu_f32 = 51
  case sigmoid_f32 = 52
  case sign_f32 = 53
  case sign_i32 = 54 // integer operation
  case sin_f32 = 55
  case sinh_f32 = 56
  case softplus_f32 = 57
  
  case softsign_f32 = 60
  case sqrt_f32 = 61
  case square_f32 = 62
  case square_i32 = 63 // integer operation
  case tan_f32 = 64
  case tanh_f32 = 65
  
  case scalar_add_f32 = 70 // requires metadata
  case scalar_add_i32 = 71 // requires metadata
  case scalar_mul_f32 = 72 // requires metadata
  case scalar_mul_i32 = 73 // requires metadata
}

enum UnaryOperationType2: UInt8 {
  case abs_i64 = 0
  case neg_i64 = 1
  case sign_i64 = 2
  case sign_u64 = 3
  case square_i64 = 4
  case square_u64 = 5
  
  case cast_f32_to_i64 = 10
  case cast_i64_to_bool = 11
  case cast_i64_to_f16 = 12
  case cast_i64_to_f32 = 13
  case cast_i64_to_u8 = 14
  case cast_i64_to_u16 = 15
  case cast_i64_to_u32 = 16
  
  case cast_f32_to_u32 = 20
  case cast_f32_to_u64 = 21
  case cast_u64_to_bool = 22
  case cast_u64_to_f16 = 23
  case cast_u64_to_f32 = 24
  case cast_u64_to_u8 = 25
  case cast_u64_to_u16 = 26
  case cast_u64_to_u32 = 27
  
  case scalar_add_i64 = 30 // requires metadata
  case scalar_mul_i64 = 31 // requires metadata
  case scalar_mul_u64 = 32 // requires metadata
  
  init?(type32: UnaryOperationType, dataType: DataType) {
    guard !dataType.representableByInt32 else {
      return nil
    }
    
    switch type32 {
    case .abs_i32:
      if dataType == .int64 {
        self = .abs_i64
      } else {
        return nil
      }
    case .neg_i32:
      if dataType == .int64 {
        self = .neg_i64
      } else {
        return nil
      }
    case .sign_i32:
      self = (dataType == .uint64) ? .sign_u64 : .sign_i64
    case .scalar_add_i32:
      self = .scalar_add_i64
    case .scalar_mul_i32:
      self = (dataType == .uint64) ? .scalar_mul_u64 : .scalar_mul_i64
    default:
      return nil
    }
  }
}

// Ordered by relative frequency, minimizing the number of conditional checks during compilation and
// encoding.
enum EagerOperation {
  struct Unary {
    // `metadata` stored before `operation` to make the memory layout more compact.
    var metadata: UInt64? = nil
    var isNoOp: Bool = false
    var operation: UnaryOperationType
    var input: OpaquePointer
    var output: OpaquePointer
  }
  case unary(Unary)
  
  struct ExplicitCopy {
    var input: OpaquePointer
    var output: OpaquePointer
  }
  case explicitCopy(ExplicitCopy)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct Elementwise {
    // `metadata` much less vector capacity of `operations`. It doesn't need as much storage because
    // it's serialized efficiently. Metadata is only recorded after each operation that needs it.
    var operations: SmallVector<SIMD8<UInt16>>
    
    // Warning: `SIMD2` does not mean 2 operations worth of metadata. It means the total capacity
    // for metadata is 16, which happens to be (2 operations) * (8 bytes/operation). The rationing
    // of metadata per operation is subject to change.
    var metadata: SmallVector<SIMD2<UInt64>>
    var input: Allocation
    var output: Allocation
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
