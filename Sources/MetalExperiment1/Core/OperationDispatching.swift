//
//  OperationDispatching.swift
//  
//
//  Created by Philip Turner on 7/15/22.
//

import Darwin

// MARK: - Operation Execution

extension MTLPluggableDevice {
  @inline(never)
  public func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    self.sync {
      let string = StringWrapper(wrapping: name)
      guard let function = OperationRegistry.registry[string] else {
        fatalError("Could not find operation '\(string.makeString())'.")
      }
      var handle: UnsafeMutableRawPointer?
      if isDefault {
        // Use internal mechanism to reduce ARC overhead. Fetch the static property which seems to
        // not reference-count.
      } else {
        handle = Unmanaged<MTLPluggableDevice>.passUnretained(self).toOpaque()
      }
      function.call(attributes, inputs, outputs, handle)
      self.maybeFlushStream()
    }
  }
}

// MARK: - Operation Dispatch Table

extension OperationRegistry {
  static let registry: [StringWrapper: Function] = [
    // Elementwise Unary
    
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
    "ScalarSub": scalarSub,
    "ScalarSubInverse": scalarSubInverse,
    "ScalarMul": scalarMul,
    "ScalarDiv": scalarDiv,
    "ScalarDivInverse": scalarDivInverse,
    
    // Elementwise Binary
    
    "AddV2": addV2,
    "ApproximateEqual": approximateEqual,
    "Equal": equal,
    "Less": less,
    "Greater": greater,
    "NotEqual": notEqual,
    "GreaterEqual": greaterEqual,
    "LessEqual": lessEqual,
    
    "Div": div,
    "EluGrad": eluGrad,
    "LeakyReluGrad": leakyReluGrad,
    "LogicalAnd": logicalAnd,
    "LogicalOr": logicalOr,
    
    "Maximum": maximum,
    "Minimum": minimum,
    "Mod": mod,
    
    "Mul": mul,
    "Pow": pow,
    "Relu6Grad": relu6Grad,
    "ReluGrad": reluGrad,
    
    "RsqrtGrad": rsqrtGrad,
    "SeluGrad": seluGrad,
    "SigmoidGrad": sigmoidGrad,
    "SoftplusGrad": softplusGrad,
    "SoftsignGrad": softsignGrad,
    
    "SquaredDifference": squaredDifference,
    "Sub": sub,
    "Xdivy": xdivy,
    
    // Elementwise Ternary
    
    "ClipByValue": clipByValue,
    "Select": select,
  ]
}

struct OperationRegistry {
  typealias FunctionSignature = @convention(c) (
    OpaquePointer?, Int, OpaquePointer?, Int, OpaquePointer?, Int, UnsafeMutableRawPointer?
  ) -> Void
  
  struct Arguments {
    var attributes: UnsafeRawBufferPointer
    var inputs: UnsafeBufferPointer<OpaquePointer>
    var outputs: UnsafeMutableBufferPointer<OpaquePointer>
    var handle: UnsafeMutableRawPointer?
    
    // Does not resolve the pluggable device during initialization. This prevents ARC from retaining
    // it outside of where it's used.
    @inline(__always)
    var device: MTLPluggableDevice {
      if let handle = handle {
        return Unmanaged<MTLPluggableDevice>.fromOpaque(handle).takeUnretainedValue()
      } else {
        return MTLPluggableDevice.default
      }
    }
    
    @inline(__always)
    init(
      _ attributesPtr: OpaquePointer?,
      _ attributesCount: Int,
      _ inputsPtr: OpaquePointer?,
      _ inputsCount: Int,
      _ outputsPtr: OpaquePointer?,
      _ outputsCount: Int,
      _ handle: UnsafeMutableRawPointer?
    ) {
      self.attributes = .init(start: .init(attributesPtr), count: attributesCount)
      self.inputs = .init(start: .init(inputsPtr), count: inputsCount)
      self.outputs = .init(start: .init(outputsPtr), count: outputsCount)
      self.handle = handle
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
      _ outputs: UnsafeMutableBufferPointer<OpaquePointer>,
      _ handle: UnsafeMutableRawPointer?
    ) {
      body(
        .init(attributes.baseAddress), attributes.count, .init(inputs.baseAddress), inputs.count,
        .init(outputs.baseAddress), outputs.count, handle)
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

// Quickly loop over a set of inputs. This inlines what would otherwise be a function call, without
// bloating the program binary. The function is not used yet, but may be used if the constant
// folding algorithm becomes more complex.
@inline(__always)
fileprivate func fastIterate(
  _ input1: AllocationHandle?,
  _ input2: AllocationHandle?,
  _ input3: AllocationHandle?,
  _ body: (AllocationHandle) -> Void
) {
  let storage = SIMD3<Int>(
    unsafeBitCast(input1, to: Int.self),
    unsafeBitCast(input2, to: Int.self),
    unsafeBitCast(input3, to: Int.self))
  for i in 0..<3 {
    if let input = unsafeBitCast(storage[i], to: AllocationHandle?.self) {
      body(input)
    }
  }
}

// MARK: - Elementwise Unary Operations

extension OperationRegistry {
  @inline(never)
  static func constantFoldingHelper(
    _ inputsCount: Int,
    _ outputsCount: Int,
    _ device: MTLPluggableDevice,
    _ input: AllocationHandle
  ) -> (shouldConstantFold: Bool, outputReferenceCount: Int) {
    precondition(inputsCount == 1, "Passed \(inputsCount) inputs into a unary operation.")
    precondition(outputsCount == 1, "Passed \(outputsCount) outputs into a unary operation.")
    
    let byteCount = input.byteCount
    if byteCount <= kConstantDataThreshold,
       byteCount == input.dataType.stride {
      let inputAllocation = input.reference!.takeUnretainedValue()
      if inputAllocation.constantData != nil {
        return (shouldConstantFold: true, outputReferenceCount: 1)
      }
    }
    device._internalRetain(input)
    return (shouldConstantFold: false, outputReferenceCount: 2)
  }
  
  static func dispatchUnary(
    _ args: inout Arguments,
    _ operation_f32: UnaryOperationType?,
    _ operation_i32: UnaryOperationType?,
    _ operation_bool: UnaryOperationType?,
    _ metadata: UInt64? = nil
  ) {
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, input)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    if let operation_bool = operation_bool {
      precondition(dataType == .bool, dataMismatch(operation_bool))
      guard operation_f32 == nil,
            operation_i32 == nil else {
        fatalError("This should never happen.")
      }
      operation = operation_bool.rawValue
      dataGroup = .f32_i32
    } else {
      precondition(dataType != .bool, dataMismatch(operation_f32 ?? operation_i32!))
      if let operation_f32 = operation_f32 {
        if let operation_i32 = operation_i32 {
          if dataType.isFloatingPoint {
            operation = operation_f32.rawValue
            dataGroup = .f32_i32
          } else if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = UnaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
          precondition(dataType.isFloatingPoint, dataMismatch(operation_f32))
          operation = operation_f32.rawValue
          dataGroup = .f32_i32
        }
      } else {
        if let operation_i32 = operation_i32 {
          precondition(!dataType.isFloatingPoint, dataMismatch(operation_i32))
          if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = UnaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
           fatalError("This should never happen.")
        }
      }
    }
    
    let unary = EagerOperation.Unary(
      operation, input, output, dataGroup, metadata)
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.unary(unary))
    } else {
      // Execute operation
      device.constantFold(unary)
    }
  }
  
  // Named after the Metal Standard Library header, `metal_relational`.
  static func dispatchUnaryRelational(
    _ args: inout Arguments,
    _ operation: UnaryOperationType
  ) {
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(dataType.isFloatingPoint, dataMismatch(operation))
    let stridePowerOf2 = (dataType == .float16) ? 2.trailingZeroBitCount : 4.trailingZeroBitCount
    let byteCount = input.byteCount >> stridePowerOf2
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, .bool, byteCount, input.shape)
    encodeOutput(&args.outputs, output)
    
    let unary = EagerOperation.Unary(
      operation.rawValue, input, output, .f32_i32, nil)
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.unary(unary))
    } else {
      // Execute operation.
      device.constantFold(unary)
    }
  }
  
  static func dispatchUnaryScalar(
    _ args: inout Arguments,
    _ operation_f32: UnaryOperationType,
    _ operation_i32: UnaryOperationType
  ) {
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, input)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input.dataType
    func dataMismatch(_ operation: UnaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(dataType != .bool, dataMismatch(operation_f32))
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    var metadata: UInt64
    var scalarIsZero: Bool
    var scalarIsOne: Bool
    if dataType.isFloatingPoint {
      var scalar: Float
      switch dataType {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      case .float16:
        scalar = Float(nonmutatingReadScalar(args.attributes) as Float16)
      #endif
      case .float32:
        scalar = Float(nonmutatingReadScalar(args.attributes) as Float)
      default:
        fatalError("This should never happen.")
      }
      
      operation = operation_f32.rawValue
      metadata = UInt64(scalar.bitPattern)
      dataGroup = .f32_i32
      scalarIsZero = scalar == 0
      scalarIsOne = scalar == 1
    } else if dataType.representableByInt32 {
      var scalar: Int32
      switch dataType {
      case .int8:
        scalar = Int32(nonmutatingReadScalar(args.attributes) as Int8)
      case .int16:
        scalar = Int32(nonmutatingReadScalar(args.attributes) as Int16)
      case .int32:
        scalar = Int32(nonmutatingReadScalar(args.attributes) as Int32)
      case .uint8:
        scalar = Int32(nonmutatingReadScalar(args.attributes) as UInt8)
      case .uint16:
        scalar = Int32(nonmutatingReadScalar(args.attributes) as UInt16)
      default:
        fatalError("This should never happen.")
      }
      
      operation = operation_i32.rawValue
      metadata = UInt64(truncatingIfNeeded: scalar)
      dataGroup = .f32_i32
      scalarIsZero = scalar == 0
      scalarIsOne = scalar == 1
    } else {
      switch dataType {
      case .uint32:
        metadata = UInt64(nonmutatingReadScalar(args.attributes) as UInt32)
      default:
        metadata = UInt64(nonmutatingReadScalar(args.attributes) as UInt64)
      }
      
      operation = UnaryOperationType2(operation_i32, dataType: dataType).rawValue
      dataGroup = .u32_i64_u64
      scalarIsZero = metadata == 0
      scalarIsOne = metadata == 1
    }
    advanceAtom(&args.attributes)
    
    // Detect no-ops.
    if scalarIsZero {
      if operation_f32 == .scalar_add_f32 || operation_f32 == .scalar_add_f32 {
        operation = .max
      }
    } else if scalarIsOne {
      if operation_f32 == .scalar_mul_f32 || operation_f32 == .scalar_div_f32 {
        operation = .max
      }
    }
    
    let unary = EagerOperation.Unary(
      operation, input, output, dataGroup, metadata)
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.unary(unary))
    } else {
      // Execute operation.
      device.constantFold(unary)
    }
  }
  
  static func dispatchCast(
    _ args: inout Arguments
  ) {
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch input.
    let input = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input)
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let outputDataType = DataType(tensorFlowDataType: decodeScalar(&args.attributes))
    let shape = input.shape
    let byteCount = shape.reduce(outputDataType.stride, *)
    let output = device._internalAllocate(refCount, outputDataType, byteCount, shape)
    encodeOutput(&args.outputs, output)
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    var metadata: UInt64?
    let inputDataType = input.dataType
    if inputDataType.requiresLargeRepresentation || outputDataType.requiresLargeRepresentation {
      if let cast = UnaryOperationType2(
           casting: inputDataType, to: outputDataType, metadata: &metadata) {
        operation = cast.rawValue
      } else {
        operation = .max
      }
      dataGroup = .u32_i64_u64
    } else {
      if let cast = UnaryOperationType(
           casting: inputDataType, to: outputDataType, metadata: &metadata) {
        operation = cast.rawValue
      } else {
        operation = .max
      }
      dataGroup = .f32_i32
    }
    
    let unary = EagerOperation.Unary(
      operation, input, output, dataGroup, metadata)
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.unary(unary))
    } else {
      // Execute operation.
      device.constantFold(unary)
    }
  }
}

extension OperationRegistry {
  // Codes 0 - 7
  static let abs = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .abs_f32, .abs_i32, nil)
  }
  static let acos = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .acos_f32, nil, nil)
  }
  static let acosh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .acosh_f32, nil, nil)
  }
  static let asin = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .asin_f32, nil, nil)
  }
  static let asinh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .asinh_f32, nil, nil)
  }
  static let atan = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .atan_f32, nil, nil)
  }
  static let atanh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .atanh_f32, nil, nil)
  }
  
  // Codes 10 - 16
  static let cast = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchCast(&args)
  }
  
  // Codes 20 - 26
  static let ceil = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .ceil_f32, nil, nil)
  }
  static let cos = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .cos_f32, nil, nil)
  }
  static let cosh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .cosh_f32, nil, nil)
  }
  static let elu = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .elu_f32, nil, nil)
  }
  static let exp = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .exp_f32, nil, nil)
  }
  static let expm1 = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .expm1_f32, nil, nil)
  }
  static let floor = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .floor_f32, nil, nil)
  }
  
  // Codes 30 - 32
  static let isFinite = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryRelational(&args, .is_finite_f32)
  }
  static let isInf = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryRelational(&args, .is_inf_f32)
  }
  static let isNan = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryRelational(&args, .is_nan_f32)
  }
  
  // Codes 40 - 48
  static let leakyRelu = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let alpha: Double = decodeScalar(&args.attributes)
    let metadata = UInt64(Float(alpha).bitPattern)
    dispatchUnary(&args, .leaky_relu_f32, nil, nil, metadata)
  }
  static let log = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .log_f32, nil, nil)
  }
  static let log1p = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .log1p_f32, nil, nil)
  }
  static let logicalNot = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, nil, nil, .logical_not_bool)
  }
  static let neg = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .neg_f32, .neg_i32, nil)
  }
  static let relu = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .relu_f32, nil, nil)
  }
  static let relu6 = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .relu6_f32, nil, nil)
  }
  static let round = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .round_f32, nil, nil)
  }
  
  // Codes 50 - 57
  static let rsqrt = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .rsqrt_f32, nil, nil)
  }
  static let selu = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .selu_f32, nil, nil)
  }
  static let sigmoid = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .sigmoid_f32, nil, nil)
  }
  static let sign = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .sign_f32, .sign_i32, nil)
  }
  static let sin = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .sin_f32, nil, nil)
  }
  static let sinh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .sinh_f32, nil, nil)
  }
  static let softplus = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .softplus_f32, nil, nil)
  }
  
  // Codes 60 - 65
  static let softsign = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .softsign_f32, nil, nil)
  }
  static let sqrt = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .sqrt_f32, nil, nil)
  }
  static let square = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .square_f32, .square_i32, nil)
  }
  static let tan = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .tan_f32, nil, nil)
  }
  static let tanh = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnary(&args, .tanh_f32, nil, nil)
  }
  
  // Codes 70 - 85
  static let scalarAdd = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_add_f32, .scalar_add_i32)
  }
  static let scalarSub = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_sub_f32, .scalar_sub_i32)
  }
  static let scalarSubInverse = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_sub_inverse_f32, .scalar_sub_inverse_i32)
  }
  static let scalarMul = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_mul_f32, .scalar_mul_i32)
  }
  static let scalarDiv = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_div_f32, .scalar_div_i32)
  }
  static let scalarDivInverse = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchUnaryScalar(&args, .scalar_div_inverse_f32, .scalar_div_inverse_i32)
  }
}

// MARK: - Elementwise Binary Operations

extension OperationRegistry {
  @inline(__always)
  static func constantFoldingHelper(
    _ inputsCount: Int,
    _ outputsCount: Int,
    _ device: MTLPluggableDevice,
    _ input1: AllocationHandle,
    _ input2: AllocationHandle
  ) -> (shouldConstantFold: Bool, outputReferenceCount: Int) {
    precondition(inputsCount == 2, "Passed \(inputsCount) inputs into a binary operation.")
    precondition(outputsCount == 1, "Passed \(outputsCount) outputs into a binary operation.")
    
    let byteCount1 = input1.byteCount
    let byteCount2 = input2.byteCount
    if false &&
       byteCount1 <= kConstantDataThreshold,
       byteCount2 <= kConstantDataThreshold,
       byteCount1 == input1.dataType.stride,
       byteCount2 == input2.dataType.stride {
      let inputAllocation1 = input1.reference!.takeUnretainedValue()
      let inputAllocation2 = input2.reference!.takeUnretainedValue()
      if inputAllocation1.constantData != nil,
         inputAllocation2.constantData != nil {
        return (shouldConstantFold: true, outputReferenceCount: 1)
      }
    }
    device._internalRetain(input1)
    device._internalRetain(input2)
    return (shouldConstantFold: false, outputReferenceCount: 2)
  }
  
  enum ComparisonBase: UInt16 {
    case equal = 0
    case less = 1
    case greater = 2
  }
  
  @inline(__always)
  static func createComparisonMetadata(_ base: ComparisonBase, _ flipping: Bool) -> UInt64 {
    let vector = SIMD4<UInt16>(base.rawValue, flipping ? 1 : 0, 0, 0)
    return unsafeBitCast(vector, to: UInt64.self)
  }
  
  static func dispatchBinary(
    _ args: inout Arguments,
    _ operation_f32: BinaryOperationType?,
    _ operation_i32: BinaryOperationType?,
    _ operation_bool: BinaryOperationType?,
    _ metadata: UInt64? = nil,
    _ allowBroadcasting: Bool = false
  ) {
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch inputs.
    let input1 = decodeInput(&args.inputs)
    let input2 = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input1, input2)
    
    // Determine output shape.
    var referenceInput: AllocationHandle
    if input1.shape.elementsEqual(input2.shape) {
      referenceInput = input1
    } else if allowBroadcasting {
      if input1.dataType.stride == input1.byteCount {
        // Scalar broadcasting supported natively.
        referenceInput = input2
      } else if input2.dataType.stride == input2.byteCount {
        // Scalar broadcasting supported natively.
        referenceInput = input1
      } else {
        fatalError("Binary operations do not yet support broadcasting.")
      }
    } else {
      fatalError("""
        Operation '\((operation_f32 ?? operation_i32 ?? operation_bool)!)' does not support \
        broadcasting.
        """)
    }
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, referenceInput)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input1.dataType
    func dataMismatch(_ operation: BinaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(
      dataType == input2.dataType, "Data type '\(dataType)' does not match '\(input2.dataType)'.")
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    if let operation_bool = operation_bool {
      precondition(dataType == .bool, dataMismatch(operation_bool))
      guard operation_f32 == nil,
            operation_i32 == nil else {
        fatalError("This should never happen.")
      }
      operation = operation_bool.rawValue
      dataGroup = .f32_i32
    } else {
      precondition(dataType != .bool, dataMismatch(operation_f32 ?? operation_i32!))
      if let operation_f32 = operation_f32 {
        if let operation_i32 = operation_i32 {
          if dataType.isFloatingPoint {
            operation = operation_f32.rawValue
            dataGroup = .f32_i32
          } else if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = BinaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
          precondition(dataType.isFloatingPoint, dataMismatch(operation_f32))
          operation = operation_f32.rawValue
          dataGroup = .f32_i32
        }
      } else {
        if let operation_i32 = operation_i32 {
          precondition(!dataType.isFloatingPoint, dataMismatch(operation_i32))
          if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = BinaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
           fatalError("This should never happen.")
        }
      }
    }
    
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.binary(.init(
        operation, input1, input2, output, dataGroup, metadata)))
    } else {
      
    }
  }
  
  static func dispatchBinaryComparison(
    _ args: inout Arguments,
    _ operation_f32: BinaryOperationType,
    _ metadata: UInt64,
    _ allowBool: Bool
  ) {
    let allowBroadcasting = operation_f32 != .approximate_equal_f32
    let device = args.device
    let inputsCount = args.inputs.count
    let outputsCount = args.outputs.count
    
    // Fetch inputs.
    let input1 = decodeInput(&args.inputs)
    let input2 = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(
      inputsCount, outputsCount, device, input1, input2)
    
    // Determine output shape.
    var referenceInput: AllocationHandle
    if input1.shape.elementsEqual(input2.shape) {
      referenceInput = input1
    } else if allowBroadcasting {
      if input1.dataType.stride == input1.byteCount {
        // Scalar broadcasting supported natively.
        referenceInput = input2
      } else if input2.dataType.stride == input2.byteCount {
        // Scalar broadcasting supported natively.
        referenceInput = input1
      } else {
        fatalError("Binary operations do not yet support broadcasting.")
      }
    } else {
      fatalError("Operation '\(operation_f32)' does not support broadcasting.")
    }
    
    // Fetch data type.
    let dataType = referenceInput.dataType
    func dataMismatch(_ operation: BinaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    if dataType == .bool {
      precondition(allowBool, dataMismatch(operation_f32))
    }
    if operation_f32 == .approximate_equal_f32 {
      precondition(dataType.isFloatingPoint, dataMismatch(operation_f32))
    }
    let stridePowerOf2 = dataType.stride.trailingZeroBitCount
    let byteCount = referenceInput.byteCount >> stridePowerOf2
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, .bool, byteCount, referenceInput.shape)
    encodeOutput(&args.outputs, output)
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    let operation_i32 = BinaryOperationType.comparison_i32
    if dataType.isFloatingPoint {
      operation = operation_f32.rawValue
      dataGroup = .f32_i32
    } else if dataType.representableByInt32 {
      operation = operation_i32.rawValue
      dataGroup = .f32_i32
    } else {
      operation = BinaryOperationType2(operation_i32, dataType: dataType).rawValue
      dataGroup = .u32_i64_u64
    }
    
    // Append operation.
    if !shouldConstantFold {
      device.eagerOperations.append(.binary(.init(
        operation, input1, input2, output, dataGroup, metadata)))
    } else {
      
    }
  }
}

// TODO: When implementing constant folding, transform scalar add/sub/mul/div into faster unary
// equivalent.

extension OperationRegistry {
  // Codes 0 - 4
  static let addV2 = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .add_f32, .add_i32, nil, nil, true)
  }
  static let approximateEqual = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let tolerance: Double = decodeScalar(&args.attributes)
    let metadata = UInt64(Float(tolerance).bitPattern)
    dispatchBinaryComparison(&args, .approximate_equal_f32, metadata, false)
  }
  static let equal = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.equal, false)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, true)
  }
  static let less = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.less, false)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, false)
  }
  static let greater = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.greater, false)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, false)
  }
  static let notEqual = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.equal, true)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, true)
  }
  static let greaterEqual = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.less, true)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, false)
  }
  static let lessEqual = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let metadata = createComparisonMetadata(.greater, true)
    dispatchBinaryComparison(&args, .comparison_f32, metadata, false)
  }
  
  // Codes 10 - 15
  static let div = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .div_f32, .div_i32, nil, nil, true)
  }
  static let eluGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .elu_grad_f32, nil, nil, nil, false)
  }
  static let leakyReluGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    let alpha: Double = decodeScalar(&args.attributes)
    let metadata = UInt64(Float(alpha).bitPattern)
    dispatchBinary(&args, .leaky_relu_grad_f32, nil, nil, metadata, false)
  }
  static let logicalAnd = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, nil, nil, .logical_and_bool, nil, false)
  }
  static let logicalOr = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, nil, nil, .logical_or_bool, nil, false)
  }
  
  // Codes 20 - 25
  static let maximum = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .maximum_f32, .maximum_i32, nil, nil, true)
  }
  static let minimum = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .minimum_f32, .minimum_i32, nil, nil, true)
  }
  static let mod = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .mod_f32, .mod_i32, nil, nil, true)
  }
  
  // Codes 30 - 34
  static let mul = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .mul_f32, .mul_i32, nil, nil, true)
  }
  static let pow = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .pow_f32, nil, nil, nil, false)
  }
  static let relu6Grad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .relu6_grad_f32, nil, nil, nil, false)
  }
  static let reluGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .relu_grad_f32, nil, nil, nil, false)
  }
  
  // Codes 40 - 44
  static let rsqrtGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .rsqrt_grad_f32, nil, nil, nil, false)
  }
  static let seluGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .selu_grad_f32, nil, nil, nil, false)
  }
  static let sigmoidGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .sigmoid_grad_f32, nil, nil, nil, false)
  }
  static let softplusGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .softplus_grad_f32, nil, nil, nil, false)
  }
  static let softsignGrad = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .softsign_grad_f32, nil, nil, nil, false)
  }
  
  // Codes 50 - 54
  static let squaredDifference = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .squared_difference_f32, .squared_difference_i32, nil, nil, true)
  }
  static let sub = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .sub_f32, .sub_i32, nil, nil, true)
  }
  static let xdivy = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchBinary(&args, .xdivy_f32, nil, nil, nil, false)
  }
}

// MARK: - Elementwise Ternary Operations

extension OperationRegistry {
  @inline(__always)
  static func commonTernaryPrecondition(_ args: Arguments) {
    precondition(
      args.inputs.count == 3, "Passed \(args.inputs.count) inputs into a ternary operation.")
    precondition(
      args.outputs.count == 1, "Passed \(args.outputs.count) outputs into a ternary operation.")
  }
  
  @inline(never)
  static func constantFoldingHelper(
    _ device: MTLPluggableDevice,
    _ input1: AllocationHandle,
    _ input2: AllocationHandle,
    _ input3: AllocationHandle
  ) -> (shouldConstantFold: Bool, outputReferenceCount: Int) {
    let byteCount1 = input1.byteCount
    let byteCount2 = input2.byteCount
    let byteCount3 = input3.byteCount
    if false &&
       byteCount1 <= kConstantDataThreshold,
       byteCount2 <= kConstantDataThreshold,
       byteCount3 <= kConstantDataThreshold,
       byteCount1 == input1.dataType.stride,
       byteCount2 == input2.dataType.stride,
       byteCount3 == input3.dataType.stride {
      return (shouldConstantFold: true, outputReferenceCount: 1)
    } else {
      device._internalRetain(input1)
      device._internalRetain(input2)
      device._internalRetain(input3)
      return (shouldConstantFold: false, outputReferenceCount: 2)
    }
  }
  
  static func dispatchTernary(
    _ args: inout Arguments,
    _ operation_f32: TernaryOperationType?,
    _ operation_i32: TernaryOperationType?,
    _ operation_bool: TernaryOperationType?,
    _ metadata: UInt64? = nil,
    _ allowBroadcasting: Bool = false
  ) {
    let device = args.device
    commonTernaryPrecondition(args)
    
    // Fetch inputs.
    let input1 = decodeInput(&args.inputs)
    let input2 = decodeInput(&args.inputs)
    let input3 = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(device, input1, input2, input3)
    
    // Determine output shape.
    var referenceInput: AllocationHandle
    if input1.shape.elementsEqual(input2.shape),
       input1.shape.elementsEqual(input3.shape) {
      referenceInput = input1
    } else if allowBroadcasting {
      // Logic for scalar broadcasting would be very complex. Since it is currently unused, don't
      // spend time implementing it.
      fatalError("No generic ternary operations support broadcasting.")
    } else {
      fatalError("""
        Operation '\((operation_f32 ?? operation_i32 ?? operation_bool)!)' does not support \
        broadcasting.
        """)
    }
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, referenceInput)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input1.dataType
    func dataMismatch(_ operation: TernaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(
      dataType == input2.dataType, "Data type '\(dataType)' does not match '\(input2.dataType)'.")
    precondition(
      dataType == input3.dataType, "Data type '\(dataType)' does not match '\(input3.dataType)'.")
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    if let operation_bool = operation_bool {
      precondition(dataType == .bool, dataMismatch(operation_bool))
      guard operation_f32 == nil,
            operation_i32 == nil else {
        fatalError("This should never happen.")
      }
      operation = operation_bool.rawValue
      dataGroup = .f32_i32
    } else {
      precondition(dataType != .bool, dataMismatch(operation_f32 ?? operation_i32!))
      if let operation_f32 = operation_f32 {
        if let operation_i32 = operation_i32 {
          if dataType.isFloatingPoint {
            operation = operation_f32.rawValue
            dataGroup = .f32_i32
          } else if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = TernaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
          precondition(dataType.isFloatingPoint, dataMismatch(operation_f32))
          operation = operation_f32.rawValue
          dataGroup = .f32_i32
        }
      } else {
        if let operation_i32 = operation_i32 {
          precondition(!dataType.isFloatingPoint, dataMismatch(operation_i32))
          if dataType.representableByInt32 {
            operation = operation_i32.rawValue
            dataGroup = .f32_i32
          } else {
            operation = TernaryOperationType2(operation_i32, dataType: dataType).rawValue
            dataGroup = .u32_i64_u64
          }
        } else {
           fatalError("This should never happen.")
        }
      }
    }
    
    // Append operation.
    if !shouldConstantFold {
      device.eagerOperations.append(.ternary(.init(
        operation, input1, input2, input3, output, dataGroup, metadata)))
    } else {
      
    }
  }
  
  static func dispatchTernaryClipByValue(
    _ args: inout Arguments
  ) {
    // Scalar broadcasting allowed for 2nd and 3rd arguments, no other form of broadcasting allowed.
    let device = args.device
    commonTernaryPrecondition(args)
    
    // Fetch inputs.
    let input1 = decodeInput(&args.inputs)
    let input2 = decodeInput(&args.inputs)
    let input3 = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(device, input1, input2, input3)
    
    var shouldFail = false
    if !input1.shape.elementsEqual(input2.shape) {
      if input2.dataType.stride != input2.byteCount {
        shouldFail = true
      }
    }
    if !input1.shape.elementsEqual(input3.shape) {
      if input3.dataType.stride != input3.byteCount {
        shouldFail = true
      }
    }
    if shouldFail {
      fatalError("""
        Operation '\(TernaryOperationType.clip_by_value_f32)' support scalar broadcasting, but not \
        other forms of broadcasting.
        """)
    }
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, input1)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input1.dataType
    func dataMismatch(_ operation: TernaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(
      dataType == input2.dataType, "Data type '\(dataType)' does not match '\(input2.dataType)'.")
    precondition(
      dataType == input3.dataType, "Data type '\(dataType)' does not match '\(input3.dataType)'.")
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    if dataType.isFloatingPoint {
      operation = TernaryOperationType.clip_by_value_f32.rawValue
      dataGroup = .f32_i32
    } else if dataType.representableByInt32 {
      operation = TernaryOperationType.clip_by_value_i32.rawValue
      dataGroup = .f32_i32
    } else {
      operation = TernaryOperationType2(.clip_by_value_i32, dataType: dataType).rawValue
      dataGroup = .u32_i64_u64
    }
    
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.ternary(.init(
        operation, input1, input2, input3, output, dataGroup, nil)))
    } else {
      
    }
  }
  
  static func dispatchTernarySelect(
    _ args: inout Arguments
  ) {
    // Scalar broadcasting not supported.
    let device = args.device
    commonTernaryPrecondition(args)
    
    // Fetch inputs.
    let input1 = decodeInput(&args.inputs)
    let boolType = DataType.bool
    precondition(
      boolType == input1.dataType, "Data type '\(boolType)' does not match '\(input1.dataType)'.")
    
    let input2 = decodeInput(&args.inputs)
    let input3 = decodeInput(&args.inputs)
    let (shouldConstantFold, refCount) = constantFoldingHelper(device, input1, input2, input3)
    
    // Determine output shape.
    var referenceInput: AllocationHandle
    if input1.shape.elementsEqual(input2.shape),
       input1.shape.elementsEqual(input3.shape) {
      // Do not replicate the first input; it's always boolean. Instead, replicate `input2`.
      referenceInput = input2
    } else {
      // TODO: According to the TensorFlow documentation, this operation supports some form of
      // broadcasting. If `condition` is rank 1, the other inputs may have higher rank, but their
      // first dimension must match the size of `condition`.
      //
      // Investigate this in the TensorFlow-backed S4TF, determine whether `condition` can be 0-D
      // and broadcasted. Also, look back through the entire `_Raw` namespace to find peculiar ways
      // that broadcasting happens. Broadcasting from 1D will occur via the MPSGraph broadcast op.
      // This is incredibly expensive on the CPU, but it's a rare use case.
      fatalError(
        "Operation '\(TernaryOperationType.select_f32_i32)' does not support broadcasting.")
    }
    
    // Generate output.
    // Setting initial refcount to 2 creates an imbalanced retain.
    let output = device._internalAllocate(refCount, referenceInput)
    encodeOutput(&args.outputs, output)
    
    // Fetch data type.
    let dataType = input2.dataType
    func dataMismatch(_ operation: TernaryOperationType) -> String {
      "Operation '\(operation)' does not accept data type '\(dataType)'."
    }
    precondition(
      dataType == input3.dataType, "Data type '\(dataType)' does not match '\(input3.dataType)'.")
    
    // Select operation.
    var operation: UInt16
    var dataGroup: DataGroup
    if dataType.requiresLargeRepresentation {
      operation = TernaryOperationType2.select_i64_u64.rawValue
      dataGroup = .u32_i64_u64
    } else {
      operation = TernaryOperationType.select_f32_i32.rawValue
      dataGroup = .f32_i32
    }
    
    if !shouldConstantFold {
      // Append operation.
      device.eagerOperations.append(.ternary(.init(
        operation, input1, input2, input3, output, dataGroup, nil)))
    } else {
      
    }
  }
}

extension OperationRegistry {
  // Codes 0 - 2
  static let clipByValue = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchTernaryClipByValue(&args)
  }
  static let select = Function {
    var args = Arguments($0, $1, $2, $3, $4, $5, $6)
    dispatchTernarySelect(&args)
  }
}
