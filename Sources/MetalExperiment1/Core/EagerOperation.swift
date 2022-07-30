//
//  EagerOperation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

enum UnaryOperationType: UInt16 {
  case no_op = 65535
  
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
  case scalar_sub_f32 = 71 // requires metadata
  case scalar_sub_inverse_f32 = 72 // requires metadata
  case scalar_mul_f32 = 73 // requires metadata
  case scalar_div_f32 = 74 // requires metadata
  case scalar_div_inverse_f32 = 75 // requires metadata
  
  case scalar_add_i32 = 80 // requires metadata
  case scalar_sub_i32 = 81 // requires metadata
  case scalar_sub_inverse_i32 = 82 // requires metadata
  case scalar_mul_i32 = 83 // requires metadata
  case scalar_div_i32 = 84 // requires metadata
  case scalar_div_inverse_i32 = 85 // requires metadata
}

enum UnaryOperationType2: UInt16 {
  case no_op = 65535
  
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
  case scalar_sub_i64_u64 = 31 // requires metadata
  case scalar_sub_inverse_i64_u64 = 32 // requires metadata
  case scalar_mul_i64 = 33 // requires metadata
  case scalar_div_i64 = 34 // requires metadata
  case scalar_div_inverse_i64 = 35 // requires metadata
  
  case scalar_mul_u64 = 40 // requires metadata
  case scalar_div_u64 = 41 // requires metadata
  case scalar_div_inverse_u64 = 42 // requires metadata
  
  init(_ smallOperation: UnaryOperationType, dataType: DataType) {
    guard dataType.requiresLargeRepresentation else {
      fatalError("Data type '\(dataType)' does not require large representation.")
    }
    
    switch smallOperation {
    case .abs_i32:
      if dataType == .int64 {
        self = .abs_i64
      } else {
        self = .no_op
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
    case .scalar_sub_i32:
      self = .scalar_sub_i64_u64
    case .scalar_sub_inverse_i32:
      self = .scalar_sub_inverse_i64_u64
    case .scalar_mul_i32:
      self = (dataType == .int64) ? .scalar_mul_i64 : .scalar_mul_u64
    case .scalar_div_i32:
      self = (dataType == .int64) ? .scalar_div_i64 : .scalar_div_u64
    case .scalar_div_inverse_i32:
      self = (dataType == .int64) ? .scalar_div_inverse_i64 : .scalar_div_inverse_u64
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
  case logical_and_bool = 14
  case logical_or_bool = 15
  
  case maximum_f32 = 20
  case maximum_i32 = 21
  case minimum_f32 = 22
  case minimum_i32 = 23
  case mod_f32 = 24
  case mod_i32 = 25
  
  case mul_f32 = 30
  case mul_i32 = 31
  case pow_f32 = 32
  case relu6_grad_f32 = 33
  case relu_grad_f32 = 34
  
  case rsqrt_grad_f32 = 40
  case selu_grad_f32 = 41
  case sigmoid_grad_f32 = 42
  case softplus_grad_f32 = 43
  case softsign_grad_f32 = 44
  
  case squared_difference_f32 = 50
  case squared_difference_i32 = 51
  case sub_f32 = 52
  case sub_i32 = 53
  case xdivy_f32 = 54
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
  
  case mul_i64 = 30
  case mul_u64 = 31
  case squared_difference_i64 = 32
  case squared_difference_u64 = 33
  
  case sub_i64_u64 = 40
  
  init(_ smallOperation: BinaryOperationType, dataType: DataType) {
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
    case .mul_i32:
      self = (dataType == .int64) ?.mul_i64 : .mul_u64
    case .squared_difference_i32:
      self = (dataType == .int64) ? .squared_difference_i64 : .squared_difference_u64
    case .sub_i32:
      self = .sub_i64_u64
    default:
      fatalError("Binary operation '\(smallOperation)' has no large counterpart.")
    }
  }
}

enum TernaryOperationType: UInt16 {
  case clip_by_value_f32 = 0
  case clip_by_value_i32 = 1
  case select_f32_i32 = 2
}

enum TernaryOperationType2: UInt16 {
  case clip_by_value_i64 = 0
  case clip_by_value_u64 = 1
  case select_i64_u64 = 2
  
  init(_ smallOperation: TernaryOperationType, dataType: DataType) {
    guard dataType.requiresLargeRepresentation else {
      fatalError("Data type '\(dataType)' does not require large representation.")
    }
    
    switch smallOperation {
    case .clip_by_value_i32:
      self = (dataType == .int64) ? .clip_by_value_i64 : .clip_by_value_u64
    case .select_f32_i32:
      self = .select_i64_u64
    default:
      fatalError("Ternary operation '\(smallOperation)' has no large counterpart.")
    }
  }
}

enum RegisterSwapType: UInt16 {
  case swap_registers_1_2 = 0
  case swap_registers_1_3 = 1
  case swap_registers_1_4 = 2
  
  case swap_registers_2_3 = 10
  case swap_registers_2_4 = 11
  case swap_registers_3_4 = 12
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
    var dataGroup: DataGroup
    
    // `operation` is the raw value of either a `UnaryOperationType` or `UnaryOperationType2`.
    var operation: UInt16
    var input: AllocationHandle
    var output: AllocationHandle
    
    @inline(__always)
    init(
      _ operation: UInt16,
      _ input: AllocationHandle,
      _ output: AllocationHandle,
      _ dataGroup: DataGroup,
      _ metadata: UInt64?
    ) {
      self.operation = operation
      self.input = input
      self.output = output
      self.dataGroup = dataGroup
      self.metadata = metadata
    }
  }
  case unary(Unary)
  
  struct Binary {
    // `metadata` stored before `operation` to make the memory layout more compact.
    var metadata: UInt64? = nil
    var dataGroup: DataGroup
    
    // `operation` is the raw value of either a `BinaryOperationType` or `BinaryOperationType2`.
    var operation: UInt16
    var input1: AllocationHandle
    var input2: AllocationHandle
    var output: AllocationHandle
    
    @inline(__always)
    init(
      _ operation: UInt16,
      _ input1: AllocationHandle,
      _ input2: AllocationHandle,
      _ output: AllocationHandle,
      _ dataGroup: DataGroup,
      _ metadata: UInt64?
    ) {
      self.operation = operation
      self.input1 = input1
      self.input2 = input2
      self.output = output
      self.dataGroup = dataGroup
      self.metadata = metadata
    }
  }
  case binary(Binary)
  
  struct Ternary {
    // `metadata` stored before `operation` to make the memory layout more compact.
    var metadata: UInt64? = nil
    var dataGroup: DataGroup
    
    // `operation` is the raw value of either a `BinaryOperationType` or `BinaryOperationType2`.
    var operation: UInt16
    var input1: AllocationHandle
    var input2: AllocationHandle
    var input3: AllocationHandle
    var output: AllocationHandle
    
    @inline(__always)
    init(
      _ operation: UInt16,
      _ input1: AllocationHandle,
      _ input2: AllocationHandle,
      _ input3: AllocationHandle,
      _ output: AllocationHandle,
      _ dataGroup: DataGroup,
      _ metadata: UInt64?
    ) {
      self.operation = operation
      self.input1 = input1
      self.input2 = input2
      self.input3 = input3
      self.output = output
      self.dataGroup = dataGroup
      self.metadata = metadata
    }
  }
  case ternary(Ternary)
  
  struct ExplicitCopy {
    var input: AllocationHandle
    var output: AllocationHandle
  }
  case explicitCopy(ExplicitCopy)
}
