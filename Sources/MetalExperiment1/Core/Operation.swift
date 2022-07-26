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
  
  case cast_f32_to_f16 = 10
  case cast_f32_to_bool = 11
  case cast_f32_to_i32 = 12 // requires metadata
  case cast_i32_to_f16 = 13
  case cast_i32_to_f32 = 14
  case cast_i32_to_bool = 15
  case cast_i32_to_i32 = 16 // requires metadata
  
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

enum UnaryOperationType2: UInt16 {
  case abs_i64 = 0
  case neg_i64 = 1
  case sign_i64 = 2
  case sign_u64 = 3
  case square_i64 = 4
  case square_u64 = 5
  
  case cast_f32_to_u32 = 10
  case cast_f32_to_i64 = 11
  case cast_i64_to_f16 = 12
  case cast_i64_to_f32 = 13
  case cast_i64_u64_to_bool = 14
  
  case cast_f32_to_u64 = 20
  case cast_u64_to_f16 = 21
  case cast_u64_to_f32 = 22
  case cast_i64_u64_to_i32 = 23 // requires metadata
  case cast_i64_u64_to_u32 = 24 // requires metadata
  
  case scalar_add_i64_u64 = 30 // requires metadata
  case scalar_mul_i64 = 31 // requires metadata
  case scalar_mul_u64 = 32 // requires metadata
  
  init?(_ smallOperation: UnaryOperationType, dataType: DataType) {
    guard dataType.requiresLargeRepresentation else {
      fatalError("Data type '\(dataType)' does not require large representation.")
    }
    
    switch smallOperation {
    case .abs_i32:
      if dataType == .int64 {
        self = .abs_i64
      } else {
        return nil
      }
    case .neg_i32:
      // Produce the same behavior with signed and unsigned integers.
      self = .neg_i64
    case .sign_i32:
      self = (dataType == .int64) ? .sign_i64 : .sign_u64
    case .square_i32:
      self = (dataType == .int64) ? .square_i64 : .square_u64
    case .scalar_add_i32:
      self = .scalar_add_i64_u64
    case .scalar_mul_i32:
      self = (dataType == .int64) ? .scalar_mul_i64 : .scalar_mul_u64
    default:
      fatalError("Unary operation '\(smallOperation)' has no large counterpart.")
    }
  }
}

enum BinaryOperationType: UInt16 {
  case add_f32 = 0
  case add_i32 = 1
  case approximate_equal_f32 = 2 // requires metadata
  case comparison_f32 = 3 // requires metadata
  case comparison_i32 = 4 // requires metadata
  
  case div_f32 = 10
  case div_i32 = 11
  case elu_grad_f32 = 12
  case leaky_relu_grad_f32 = 13 // requires metadata
  case maximum_f32 = 14
  case maximum_i32 = 15
  
  case minimum_f32 = 20
  case minimum_i32 = 21
  case mod_f32 = 22
  case mod_i32 = 23
  case pow_f32 = 24
  case relu6_grad_f32 = 25
  
  case relu_grad_f32 = 30
  case rsqrt_grad_f32 = 31
  case selu_grad_f32 = 32
  case sigmoid_grad_f32 = 33
  case softplus_grad_f32 = 34
  case softsign_grad_f32 = 35
  
  case squared_difference_f32 = 40
  case squared_difference_i32 = 41
  case sub_f32 = 42
  case sub_i32 = 43
  case xdivy_f32 = 44
}

enum BinaryOperationType2: UInt16 {
  case add_i64_u64 = 0
  case comparison_i64 = 1 // requires metadata
  case comparison_u64 = 2 // requires metadata
  
  case div_i64 = 10
  case div_u64 = 11
  case maximum_i64 = 12
  case maximum_u64 = 13
  
  case minimum_i64 = 20
  case minimum_u64 = 21
  case mod_i64 = 22
  case mod_u64 = 23
  
  case squared_difference_i64 = 30
  case squared_difference_u64 = 31
  case sub_i64_u64 = 32
  
  init?(_ smallOperation: BinaryOperationType, dataType: DataType) {
    guard dataType.requiresLargeRepresentation else {
      fatalError("Data type '\(dataType)' does not require large representation.")
    }
    
    switch smallOperation {
    case .add_i32:
      self = .add_i64_u64
    case .comparison_i32:
      self = (dataType == .int64) ? .comparison_i64 : .comparison_u64
    case .div_i32:
      self = (dataType == .int64) ? .div_i64 : .div_u64
    case .maximum_i32:
      self = (dataType == .int64) ? .maximum_i64 : .maximum_u64
    case .minimum_i32:
      self = (dataType == .int64) ? .minimum_i64 : .minimum_u64
    case .mod_i32:
      self = (dataType == .int64) ? .mod_i64 : .mod_u64
    case .squared_difference_i32:
      self = (dataType == .int64) ? .squared_difference_i64 : .squared_difference_u64
    case .sub_i32:
      self = .sub_i64_u64
    default:
      fatalError("Binary operation '\(smallOperation)' has no large counterpart.")
    }
  }
}

enum DataGroup {
  case f32_i32
  case u32_i64_u64
}

// Ordered by relative frequency, minimizing the number of conditional checks during compilation and
// encoding.
enum EagerOperation {
  struct Unary {
    // `metadata` stored before `operation` to make the memory layout more compact.
    var metadata: UInt64? = nil
    var isNoOp: Bool = false
    var dataGroup: DataGroup
    
    // `operation` is the raw value of either a `UnaryOperationType` or a `UnaryOperationType2`.
    var operation: UInt16
    var input: AllocationHandle
    var output: AllocationHandle
    
    @inline(__always)
    init(
      _ operation: UInt16,
      _ input: AllocationHandle,
      _ output: AllocationHandle,
      _ dataGroup: DataGroup,
      _ metadata: UInt64?,
      _ isNoOp: Bool
    ) {
      self.operation = operation
      self.input = input
      self.output = output
      self.dataGroup = dataGroup
      self.metadata = metadata
      self.isNoOp = isNoOp
    }
  }
  case unary(Unary)
  
  struct ExplicitCopy {
    var input: AllocationHandle
    var output: AllocationHandle
  }
  case explicitCopy(ExplicitCopy)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum Instruction {
  struct Elementwise {
    // `metadata` much less vector capacity of `operations`. It doesn't need as much storage because
    // it's serialized efficiently. Metadata is only recorded after each operation that needs it.
    var operations: SmallVector<SIMD8<UInt16>>
    
    // Warning: `SIMD2` does not mean 2 operations worth of metadata. It means the total capacity
    // for metadata is 16, which happens to be (2 operations) * (8 bytes/operation). The rationing
    // of metadata per operation is subject to change.
    var metadata: SmallVector<SIMD2<UInt64>>
    var dataGroup: DataGroup
    
    var input1: Allocation
    var input2: Allocation?
    var input3: Allocation?
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
