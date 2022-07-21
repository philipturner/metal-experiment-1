//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

// Using `UInt8` instead of `UInt16` to fit as many operations as possible into a `TypeList16`.
enum UnaryOperationType: UInt8, CaseIterable {
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
//  case cast_f32_to_i32 = 11
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
  
  // TODO: - Support casting, support boolean operations
//  case is_finite_f32 = 30 // returns bool/u8
//  case is_inf_f32 = 31 // returns bool/u8
//  case is_nan_f32 = 32 // returns bool/u8
  
  // TODO: - Support boolean operations
  case leaky_relu_f32 = 40
  case log_f32 = 41
  case log1p_f32 = 42
//  case logical_not_bool = 43 // boolean operation
  case neg_f32 = 44
  case neg_i32 = 45 // integer operation
  case relu_f32 = 46
  case relu6_f32 = 47
  case round_f32 = 48 // rounds to nearest even
  
  // TODO: - Support metadata
  case rsqrt_f32 = 50
//  case selu_f32 = 51
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
  
  case increment_f32 = 70 // for testing purposes only
  case increment_i32 = 71 // for testing purposes only
}

// Ordered by relative frequency, minimizing the number of conditional checks during compilation and
// encoding.
enum EagerOperation {
  struct Unary {
    // `metadata` stored before `operation` to make the memory layout more compact.
    var metadata: UInt64? = nil
    var operation: UnaryOperationType
    var input: UInt64
    var output: UInt64
  }
  case unary(Unary)
  
  struct ExplicitCopy {
    var input: UInt64
    var output: UInt64
  }
  case explicitCopy(ExplicitCopy)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct MultiUnary {
    // `metadata` much less vector capacity of `operations`. It doesn't need as much storage because
    // it's serialized efficiently. Metadata is only recorded after each operation that needs it.
    var operations: TypeList16<UnaryOperationType>
    
    // Warning: `SIMD2` does not mean 2 operations worth of metadata. It means the total capacity
    // for metadata is 16, which happens to be (2 operations) * (8 bytes/operation). The rationing
    // of metadata per operation is subject to change.
    var metadata: TypeListStorage<SIMD2<UInt64>>
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  case multiUnary(MultiUnary)
  
  struct ExplicitCopy {
    var input: Allocation
    var output: Allocation
    var byteCount: Int
  }
  case explicitCopy(ExplicitCopy)
}
