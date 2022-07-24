//
//  Interface.swift
//  
//
//  Created by Philip Turner on 7/15/22.
//

import Darwin

// MARK: - Operation Execution

extension Context {
  // Rules for encoding attributes:
  //
  // Atoms of data are padded to 16 bytes. For strings and arrays, encode an `UnsafeBufferPointer`
  // to their data. This rule applies recursively with arrays of strings, arrays of arrays, etc.
  // After the first level of recursion, store elements in their native layout stride.
  @inline(never)
  public static func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    Context.global.sync {
      Context.global._executeOperation(name, attributes, inputs, outputs)
    }
  }
  
  @inline(__always)
  private func _executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    let string = StringWrapper(wrapping: name)
    guard let function = OperationRegistry.registry[string] else {
      fatalError("Could not find operation '\(string.makeString())'.")
    }
    function.call(attributes, inputs, outputs)
    self.maybeFlushStream()
  }
}

// MARK: - Operation Dispatch Table

extension OperationRegistry {
  static let registry: [StringWrapper: Function] = [
    // Unary
    
    "Abs": abs,
    "Acos": acos,
    "Acosh": acosh,
    "Asin": asin,
    "Asinh": asinh,
    "Atan": atan,
    "Atanh": atanh,
    
    "Cast": cast,
    
    "Ceil": ceil,
    "Cos": cos,
    "Cosh": cosh,
    "Elu": elu,
    "Exp": exp,
    "Expm1": expm1,
    "Floor": floor,
    
    "IsFinite": isFinite,
    "IsInf": isInf,
    "IsNan": isNan,
    
    "LeakyRelu": leakyRelu,
    "Log": log,
    "Log1p": log1p,
    "LogicalNot": logicalNot,
    "Neg": neg,
    "Relu": relu,
    "Relu6": relu6,
    "Round": round,
    
    "Rsqrt": rsqrt,
    "Selu": selu,
    "Sigmoid": sigmoid,
    "Sign": sign,
    "Sin": sin,
    "Sinh": sinh,
    "Softplus": softplus,
    
    "Softsign": softsign,
    "Sqrt": sqrt,
    "Square": square,
    "Tan": tan,
    "Tanh": tanh,
    
    "ScalarAdd": scalarAdd,
    "ScalarMul": scalarMul,
  ]
}

struct OperationRegistry {
  typealias FunctionSignature = @convention(c) (
    OpaquePointer?, Int, OpaquePointer?, Int, OpaquePointer?, Int) -> Void
  
  struct Arguments {
    var attributes: UnsafeRawBufferPointer
    var inputs: UnsafeBufferPointer<OpaquePointer>
    var outputs: UnsafeMutableBufferPointer<OpaquePointer>
    
    @inline(__always)
    init(
      _ attributesPtr: OpaquePointer?,
      _ attributesCount: Int,
      _ inputsPtr: OpaquePointer?,
      _ inputsCount: Int,
      _ outputsPtr: OpaquePointer?,
      _ outputsCount: Int
    ) {
      attributes = .init(start: .init(attributesPtr), count: attributesCount)
      inputs = .init(start: .init(inputsPtr), count: inputsCount)
      outputs = .init(start: .init(outputsPtr), count: outputsCount)
    }
  }
  
  struct Function {
    var body: FunctionSignature
    
    @inline(__always)
    init(_ body: @escaping FunctionSignature) {
      self.body = body
    }
    
    @inline(__always)
    func call(
      _ attributes: UnsafeRawBufferPointer,
      _ inputs: UnsafeBufferPointer<OpaquePointer>,
      _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
    ) {
      body(
        .init(attributes.baseAddress), attributes.count,
        .init(inputs.baseAddress), inputs.count,
        .init(outputs.baseAddress), outputs.count)
    }
  }
  
  // Pointer mutation
  
  @inline(__always)
  static func advanceAtom(_ ptr: inout UnsafeRawBufferPointer) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 16)
    let count = ptr.count - 16
    ptr = UnsafeRawBufferPointer(start: start, count: count)
  }
  
  @inline(__always)
  static func advanceInput(_ ptr: inout UnsafeBufferPointer<OpaquePointer>) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 1)
    let count = ptr.count - 1
    ptr = UnsafeBufferPointer(start: start, count: count)
  }
  
  @inline(__always)
  static func advanceOutput(_ ptr: inout UnsafeMutableBufferPointer<OpaquePointer>) {
    let baseAddress = ptr.baseAddress.unsafelyUnwrapped
    let start = baseAddress.advanced(by: 1)
    let count = ptr.count - 1
    ptr = UnsafeMutableBufferPointer(start: start, count: count)
  }
  
  // Data decoding
  
  // An optimization that prevents duplicating the assembly instructions for `advanceAtom`.
  @inline(__always)
  static func nonmutatingReadScalar<T: SIMDScalar>(_ ptr: UnsafeRawBufferPointer) -> T {
    let value = ptr.assumingMemoryBound(to: T.self)[0]
    return value
  }
  
  @inline(__always)
  static func decodeScalar<T: SIMDScalar>(_ ptr: inout UnsafeRawBufferPointer) -> T {
    let value = ptr.assumingMemoryBound(to: T.self)[0]
    advanceAtom(&ptr)
    return value
  }
  
  @inline(__always)
  static func decodeString(_ ptr: inout UnsafeRawBufferPointer) -> StringWrapper {
    let wrappedValue = ptr.assumingMemoryBound(to: UnsafeRawBufferPointer.self)[0]
    advanceAtom(&ptr)
    return StringWrapper(wrapping: wrappedValue)
  }
  
  @inline(__always)
  static func decodeInput(_ ptr: inout UnsafeBufferPointer<OpaquePointer>) -> AllocationHandle {
    let value = ptr[0]
    advanceInput(&ptr)
    return AllocationHandle(value)
  }
  
  @inline(__always)
  static func encodeOutput(
    _ ptr: inout UnsafeMutableBufferPointer<OpaquePointer>,
    _ output: AllocationHandle
  ) {
    ptr[0] = output._cHandle
    advanceOutput(&ptr)
  }
}

// MARK: - Unary Operations

extension OperationRegistry {
  @inline(__always)
  static func commonUnaryPrecondition(_ args: Arguments) {
    precondition(
      args.inputs.count == 1, "Passed \(args.inputs.count) inputs into a unary operation.")
    precondition(
      args.outputs.count == 1, "Passed \(args.outputs.count) inputs into a unary operation.")
  }
  
  static func dispatchUnary(
    _ args: inout Arguments,
    _ operation_f32: UnaryOperationType?,
    _ operation_i32: UnaryOperationType?,
    _ operation_bool: UnaryOperationType?,
    _ metadata: UInt64? = nil
  ) {
    let ctx = Context.global
    commonUnaryPrecondition(args)
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    ctx._internalRetain(input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = ctx._internalAllocate(2, input)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation with code '\(operation.rawValue)' does not accept data type '\(dataType)'."
    }
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup = .f32_i32
    if let operation_bool = operation_bool {
      precondition(dataType == .bool, dataMismatch(operation_bool))
      guard operation_f32 == nil,
            operation_i32 == nil else {
        fatalError("This should never happen.")
      }
      operation = operation_bool.rawValue
    } else {
      precondition(dataType != .bool, dataMismatch(operation_f32 ?? operation_i32!))
      if let operation_f32 = operation_f32 {
        if let operation_i32 = operation_i32 {
          if dataType.isFloatingPoint {
            operation = operation_f32.rawValue
          } else if dataType.representableByInt32 {
            operation = operation_i32.rawValue
          } else {
            operation = UnaryOperationType2(operation_i32, dataType: dataType)!.rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
          precondition(dataType.isFloatingPoint, dataMismatch(operation_f32))
          operation = operation_f32.rawValue
        }
      } else {
        if let operation_i32 = operation_i32 {
          precondition(!dataType.isFloatingPoint, dataMismatch(operation_i32))
          if dataType.representableByInt32 {
            operation = operation_i32.rawValue
          } else {
            operation = UnaryOperationType2(operation_i32, dataType: dataType)!.rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
           fatalError("This should never happen.")
        }
      }
    }
    
    // Append operation.
    ctx.eagerOperations.append(.unary(.init(
      operation, input, output, dataGroup, metadata, false)))
  }
  
  // Named after the Metal Standard Library header, `metal_relational`.
  static func dispatchUnaryRelational(
    _ args: inout Arguments,
    _ operation: UnaryOperationType
  ) {
    let ctx = Context.global
    commonUnaryPrecondition(args)
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    ctx._internalRetain(input)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation with code '\(operation.rawValue)' does not accept data type '\(dataType)'."
    }
    precondition(dataType.isFloatingPoint, dataMismatch(operation))
    let stridePowerOf2 = (dataType == .float16) ? 2.trailingZeroBitCount : 4.trailingZeroBitCount
    let byteCount = input.byteCount >> stridePowerOf2
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = ctx._internalAllocate(2, .bool, byteCount, input.shape)
    encodeOutput(&args.outputs, output)
    
    // Append operation.
    ctx.eagerOperations.append(.unary(.init(
      operation.rawValue, input, output, .f32_i32, nil, false)))
  }
  
  static func dispatchUnaryScalar(
    _ args: inout Arguments,
    _ operation_f32: UnaryOperationType,
    _ operation_i32: UnaryOperationType
  ) {
    let ctx = Context.global
    commonUnaryPrecondition(args)
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    ctx._internalRetain(input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = ctx._internalAllocate(2, input)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation with code '\(operation.rawValue)' does not accept data type '\(dataType)'."
    }
    precondition(dataType != .bool, dataMismatch(operation_f32))
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup = .f32_i32
    var metadata: UInt64
    var isNoOp: Bool
    if dataType.isFloatingPoint {
      var rhs: Float
      switch dataType {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      case .float16:
        rhs = Float(nonmutatingReadScalar(args.attributes) as Float16)
      #endif
      case .float32:
        rhs = Float(nonmutatingReadScalar(args.attributes) as Float)
      default:
        fatalError("This should never happen.")
      }
      
      operation = operation_f32.rawValue
      metadata = UInt64(rhs.bitPattern)
      isNoOp = (operation_f32 == .scalar_add_f32) ? (rhs == 0) : (rhs == 1)
    } else if dataType.representableByInt32 {
      var rhs: Int32
      switch dataType {
      case .int8:
        rhs = Int32(nonmutatingReadScalar(args.attributes) as Int8)
      case .int16:
        rhs = Int32(nonmutatingReadScalar(args.attributes) as Int16)
      case .int32:
        rhs = Int32(nonmutatingReadScalar(args.attributes) as Int32)
      case .uint8:
        rhs = Int32(nonmutatingReadScalar(args.attributes) as UInt8)
      case .uint16:
        rhs = Int32(nonmutatingReadScalar(args.attributes) as UInt16)
      default:
        fatalError("This should never happen.")
      }
      
      operation = operation_i32.rawValue
      metadata = UInt64(truncatingIfNeeded: rhs)
      isNoOp = (operation_f32 == .scalar_add_f32) ? (rhs == 0) : (rhs == 1)
    } else {
      switch dataType {
      case .uint32:
        metadata = UInt64(nonmutatingReadScalar(args.attributes) as UInt32)
      default:
        metadata = UInt64(nonmutatingReadScalar(args.attributes) as UInt64)
      }
      
      operation = UnaryOperationType2(operation_i32, dataType: dataType)!.rawValue
      dataGroup = .u32_i64_u64
      isNoOp = (operation_f32 == .scalar_add_f32) ? (metadata == 0) : (metadata == 1)
    }
    advanceAtom(&args.attributes)
    
    // Append operation.
    ctx.eagerOperations.append(.unary(.init(
      operation, input, output, dataGroup, metadata, isNoOp)))
  }
  
  static func dispatchCast(
    _ args: inout Arguments
  ) {
    let ctx = Context.global
    commonUnaryPrecondition(args)
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    ctx._internalRetain(input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let outputDataType = DataType(tensorFlowDataType: decodeScalar(&args.attributes))
    let shape = input.shape
    let byteCount = shape.reduce(outputDataType.stride, *)
    let output = ctx._internalAllocate(2, outputDataType, byteCount, shape)
    encodeOutput(&args.outputs, output)
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    var metadata: UInt64?
    var isNoOp: Bool
    let inputDataType = input.dataType
    if inputDataType.requiresLargeRepresentation || outputDataType.requiresLargeRepresentation {
      let cast = UnaryOperationType2(
        casting: inputDataType, to: outputDataType, metadata: &metadata)
      if let cast = cast {
        operation = cast.rawValue
        isNoOp = false
      } else {
        operation = .max
        isNoOp = true
      }
      dataGroup = .u32_i64_u64
    } else {
      let cast = UnaryOperationType(
        casting: inputDataType, to: outputDataType, metadata: &metadata)
      if let cast = cast {
        operation = cast.rawValue
        isNoOp = false
      } else {
        operation = .max
        isNoOp = true
      }
      dataGroup = .f32_i32
    }
    
    // Append operation.
    ctx.eagerOperations.append(.unary(.init(
      operation, input, output, dataGroup, metadata, isNoOp)))
  }
}

extension OperationRegistry {
  // Codes 0 - 7
  static let abs = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .abs_f32, .abs_i32, nil)
  }
  static let acos = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .acos_f32, nil, nil)
  }
  static let acosh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .acosh_f32, nil, nil)
  }
  static let asin = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .asin_f32, nil, nil)
  }
  static let asinh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .asinh_f32, nil, nil)
  }
  static let atan = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .atan_f32, nil, nil)
  }
  static let atanh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .atanh_f32, nil, nil)
  }
  
  // Codes 10 - 16
  static let cast = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchCast(&args)
  }
  
  // Codes 20 - 26
  static let ceil = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .ceil_f32, nil, nil)
  }
  static let cos = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .cos_f32, nil, nil)
  }
  static let cosh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .cosh_f32, nil, nil)
  }
  static let elu = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .elu_f32, nil, nil)
  }
  static let exp = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .exp_f32, nil, nil)
  }
  static let expm1 = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .expm1_f32, nil, nil)
  }
  static let floor = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .floor_f32, nil, nil)
  }
  
  // Codes 30 - 32
  static let isFinite = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnaryRelational(&args, .is_finite_f32)
  }
  static let isInf = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnaryRelational(&args, .is_inf_f32)
  }
  static let isNan = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnaryRelational(&args, .is_nan_f32)
  }
  
  // Codes 40 - 48
  static let leakyRelu = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    let alpha: Double = decodeScalar(&args.attributes)
    let metadata = UInt64(Float(alpha).bitPattern)
    dispatchUnary(&args, .leaky_relu_f32, nil, nil, metadata)
  }
  static let log = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .log_f32, nil, nil)
  }
  static let log1p = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .log1p_f32, nil, nil)
  }
  static let logicalNot = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, nil, nil, .logical_not_bool)
  }
  static let neg = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .neg_f32, .neg_i32, nil)
  }
  static let relu = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .relu_f32, nil, nil)
  }
  static let relu6 = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .relu6_f32, nil, nil)
  }
  static let round = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .round_f32, nil, nil)
  }
  
  // Codes 50 - 57
  static let rsqrt = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .rsqrt_f32, nil, nil)
  }
  static let selu = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .selu_f32, nil, nil)
  }
  static let sigmoid = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .sigmoid_f32, nil, nil)
  }
  static let sign = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .sign_f32, .sign_i32, nil)
  }
  static let sin = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .sin_f32, nil, nil)
  }
  static let sinh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .sinh_f32, nil, nil)
  }
  static let softplus = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .softplus_f32, nil, nil)
  }
  
  // Codes 60 - 65
  static let softsign = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .softsign_f32, nil, nil)
  }
  static let sqrt = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .sqrt_f32, nil, nil)
  }
  static let square = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .square_f32, .square_i32, nil)
  }
  static let tan = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .tan_f32, nil, nil)
  }
  static let tanh = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnary(&args, .tanh_f32, nil, nil)
  }
  
  // Codes 70 - 73
  static let scalarAdd = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnaryScalar(&args, .scalar_add_f32, .scalar_add_i32)
  }
  static let scalarMul = Function {
    var args = Arguments($0, $1, $2, $3, $4 ,$5)
    dispatchUnaryScalar(&args, .scalar_mul_f32, .scalar_mul_i32)
  }
}
