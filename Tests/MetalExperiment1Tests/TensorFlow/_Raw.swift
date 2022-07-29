//
//  _Raw.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

import MetalExperiment1

// MARK: - _Raw Helpers

@inlinable @inline(__always)
func dispatchUnary<T0, T1>(
  _ name: StaticString,
  _ input1: Tensor<T0>
) -> Tensor<T1> {
  return decodeOutputs { outputs in
    encodeInputs(input1) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
    }
  }
}

@inlinable @inline(__always)
func dispatchUnary<T0, T1, U0>(
  _ name: StaticString,
  _ input1: Tensor<T0>,
  _ attribute1: U0
) -> Tensor<T1> {
  return decodeOutputs { outputs in
    encodeInputs(input1) { inputs in
      encodeAttributes(attribute1) { attributes in
        let name = encodeName(name)
        Context.executeOperation(name, attributes, inputs, outputs)
      }
    }
  }
}

@inlinable @inline(__always)
func dispatchUnaryRelational<T0>(
  _ name: StaticString,
  _ input1: Tensor<T0>
) -> Tensor<Bool> {
  return decodeOutputs { outputs in
    encodeInputs(input1) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
    }
  }
}

@inlinable @inline(__always)
func dispatchBinary<T0, T1, T2>(
  _ name: StaticString,
  _ input1: Tensor<T0>,
  _ input2: Tensor<T1>
) -> Tensor<T2> {
  return decodeOutputs { outputs in
    encodeInputs(input1, input2) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
    }
  }
}

@inlinable @inline(__always)
func dispatchBinary<T0, T1, T2, U0>(
  _ name: StaticString,
  _ input1: Tensor<T0>,
  _ input2: Tensor<T1>,
  _ attribute1: U0
) -> Tensor<T2> {
  return decodeOutputs { outputs in
    encodeInputs(input1, input2) { inputs in
      encodeAttributes(attribute1) { attributes in
        let name = encodeName(name)
        Context.executeOperation(name, attributes, inputs, outputs)
      }
    }
  }
}

@inlinable @inline(__always)
func encodeName(_ name: StaticString) -> UnsafeRawBufferPointer {
  let start = name.utf8Start
  let count = name.utf8CodeUnitCount
  return UnsafeRawBufferPointer(start: start, count: count)
}

@inlinable @inline(__always)
func encodeAttributes() -> UnsafeRawBufferPointer {
  return UnsafeRawBufferPointer(start: nil, count: 0)
}

@inlinable @inline(__always)
func encodeAttributes<T0>(
  _ input1: T0,
  _ body: (UnsafeRawBufferPointer) -> Void
) {
  withUnsafeTemporaryAllocation(of: (UInt64, UInt64).self, capacity: 1) { bufferPointer in
    bufferPointer[0] = (0, 0)
    let baseAddress = bufferPointer.baseAddress.unsafelyUnwrapped
    let rawAddress = UnsafeMutablePointer<T0>(CTensorHandle(baseAddress))
    rawAddress[0] = input1
    body(UnsafeRawBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func encodeInputs<T0>(
  _ input1: Tensor<T0>,
  _ body: (UnsafeBufferPointer<CTensorHandle>) -> Void
) {
  withUnsafeTemporaryAllocation(of: CTensorHandle.self, capacity: 1) { bufferPointer in
    bufferPointer[0] = input1._rawTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func encodeInputs<T0, T1>(
  _ input1: Tensor<T0>,
  _ input2: Tensor<T1>,
  _ body: (UnsafeBufferPointer<CTensorHandle>) -> Void
) {
  withUnsafeTemporaryAllocation(of: CTensorHandle.self, capacity: 2) { bufferPointer in
    bufferPointer[0] = input1._rawTensorHandle
    bufferPointer[1] = input2._rawTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func decodeOutput<T>(
  _ ptr: UnsafeMutableBufferPointer<CTensorHandle>, _ index: Int
) -> Tensor<T> {
  let handle = TensorHandle<T>(_owning: ptr[index])
  return Tensor(handle: handle)
}

@inlinable @inline(__always)
func decodeOutputs<T0>(
  _ body: (UnsafeMutableBufferPointer<CTensorHandle>) -> Void
) -> (Tensor<T0>) {
  withUnsafeTemporaryAllocation(of: CTensorHandle.self, capacity: 1) { bufferPointer in
    body(bufferPointer)
    return (
      decodeOutput(bufferPointer, 0)
    )
  }
}

@inlinable @inline(__always)
func decodeOutputs<T0, T1>(
  _ body: (UnsafeMutableBufferPointer<CTensorHandle>) -> Void
) -> (Tensor<T0>, Tensor<T1>) {
  withUnsafeTemporaryAllocation(of: CTensorHandle.self, capacity: 2) { bufferPointer in
    body(bufferPointer)
    return (
      decodeOutput(bufferPointer, 0),
      decodeOutput(bufferPointer, 1)
    )
  }
}

// MARK: - _Raw

@usableFromInline
protocol _RawEncodable {
  @inlinable
  func createAtom() -> (UInt64, UInt64)
}

public enum _Raw {
  // Unary
  
  @inlinable @inline(__always)
  public static func abs<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Abs", x)
  }
  
  @inlinable @inline(__always)
  public static func acos<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Acos", x)
  }
  
  @inlinable @inline(__always)
  public static func acosh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Acosh", x)
  }
  
  @inlinable @inline(__always)
  public static func asin<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Asin", x)
  }
  
  @inlinable @inline(__always)
  public static func asinh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Asinh", x)
  }
  
  @inlinable @inline(__always)
  public static func atan<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Atan", x)
  }
  
  @inlinable @inline(__always)
  public static func atanh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Atanh", x)
  }
  
  
  
  @inlinable @inline(__always)
  public static func cast<T, U>(
    _ x: Tensor<T>
  ) -> Tensor<U> {
    dispatchUnary("Cast", x, U.tensorFlowDataType)
  }
  
  
  
  @inlinable @inline(__always)
  public static func ceil<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Ceil", x)
  }
  
  @inlinable @inline(__always)
  public static func cos<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Cos", x)
  }
  
  @inlinable @inline(__always)
  public static func cosh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Cosh", x)
  }
  
  @inlinable @inline(__always)
  public static func elu<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Elu", features)
  }
  
  @inlinable @inline(__always)
  public static func exp<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Exp", x)
  }
  
  @inlinable @inline(__always)
  public static func expm1<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Expm1", x)
  }
  
  @inlinable @inline(__always)
  public static func floor<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Floor", x)
  }
  
  
  
  @inlinable @inline(__always)
  public static func isFinite<T>(
    _ x: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchUnaryRelational("IsFinite", x)
  }
  
  @inlinable @inline(__always)
  public static func isInf<T>(
    _ x: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchUnaryRelational("IsInf", x)
  }
  
  @inlinable @inline(__always)
  public static func isNan<T>(
    _ x: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchUnaryRelational("IsNan", x)
  }
  
  
  
  @inlinable @inline(__always)
  public static func leakyRelu<T>(
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    dispatchUnary("LeakyRelu", features, alpha)
  }
  
  @inlinable @inline(__always)
  public static func log<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Log", x)
  }
  
  @inlinable @inline(__always)
  public static func log1p<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Log1p", x)
  }
  
  @inlinable @inline(__always)
  public static func logicalNot<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("LogicalNot", x)
  }
  
  @inlinable @inline(__always)
  public static func neg<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Neg", x)
  }
  
  @inlinable @inline(__always)
  public static func relu<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Relu", features)
  }
  
  @inlinable @inline(__always)
  public static func relu6<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Relu6", features)
  }
  
  @inlinable @inline(__always)
  public static func round<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Round", x)
  }
  
  
  
  @inlinable @inline(__always)
  public static func rsqrt<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Rsqrt", x)
  }
  
  @inlinable @inline(__always)
  public static func selu<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Selu", features)
  }
  
  @inlinable @inline(__always)
  public static func sigmoid<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Sigmoid", x)
  }
  
  @inlinable @inline(__always)
  public static func sign<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Sign", x)
  }
  
  @inlinable @inline(__always)
  public static func sin<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Sin", x)
  }
  
  @inlinable @inline(__always)
  public static func sinh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Sinh", x)
  }
  
  @inlinable @inline(__always)
  public static func softplus<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Softplus", features)
  }
  
  
  
  @inlinable @inline(__always)
  public static func softsign<T>(
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Softsign", features)
  }
  
  @inlinable @inline(__always)
  public static func sqrt<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Sqrt", x)
  }
  
  @inlinable @inline(__always)
  public static func square<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Square", x)
  }
  
  @inlinable @inline(__always)
  public static func tan<T>(
    _ x: Tensor<T>
  )-> Tensor<T> {
    dispatchUnary("Tan", x)
  }
  
  @inlinable @inline(__always)
  public static func tanh<T>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    dispatchUnary("Tanh", x)
  }
  
  
  
  @inlinable @inline(__always)
  public static func scalarAdd<T>(
    _ x: Tensor<T>,
    _ y: T
  ) -> Tensor<T> {
    dispatchUnary("ScalarAdd", x, y)
  }
  
  @inlinable @inline(__always)
  public static func scalarMul<T>(
    _ x: Tensor<T>,
    _ y: T
  ) -> Tensor<T> {
    dispatchUnary("ScalarMul", x, y)
  }
  
  // Binary
  
  @inlinable @inline(__always)
  public static func addV2<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("AddV2", x, y)
  }
  
  @inlinable @inline(__always)
  public static func approximateEqual<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    tolerance: Double = 1e-05
  ) -> Tensor<Bool> {
    dispatchBinary("ApproximateEqual", x, y, tolerance)
  }
  
  @inlinable @inline(__always)
  public static func equal<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("Equal", x, y)
  }
  
  @inlinable @inline(__always)
  public static func less<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("Less", x, y)
  }
  
  @inlinable @inline(__always)
  public static func greater<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("Greater", x, y)
  }
  
  @inlinable @inline(__always)
  public static func notEqual<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("NotEqual", x, y)
  }
  
  @inlinable @inline(__always)
  public static func greaterEqual<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("GreaterEqual", x, y)
  }
  
  @inlinable @inline(__always)
  public static func lessEqual<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<Bool> {
    dispatchBinary("LessEqual", x, y)
  }
  
  
  
  @inlinable @inline(__always)
  public static func div<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Div", x, y)
  }
  
  @inlinable @inline(__always)
  public static func eluGrad<T>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("EluGrad", gradients, outputs)
  }
  
  @inlinable @inline(__always)
  public static func leakyReluGrad<T>(
    gradients: Tensor<T>,
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    dispatchBinary("LeakyReluGrad", gradients, features, alpha)
  }
  
  @inlinable @inline(__always)
  public static func logicalAnd(
    _ x: Tensor<Bool>,
    _ y: Tensor<Bool>
  ) -> Tensor<Bool> {
    dispatchBinary("LogicalAnd", x, y)
  }
  
  @inlinable @inline(__always)
  public static func logicalOr(
    _ x: Tensor<Bool>,
    _ y: Tensor<Bool>
  ) -> Tensor<Bool> {
    dispatchBinary("LogicalOr", x, y)
  }
  
  
  
  @inlinable @inline(__always)
  public static func maximum<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Maximum", x, y)
  }
  
  @inlinable @inline(__always)
  public static func minimum<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Minimum", x, y)
  }
  
  @inlinable @inline(__always)
  public static func mod<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Mod", x, y)
  }
  
  
  
  @inlinable @inline(__always)
  public static func mul<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Mul", x, y)
  }
  
  @inlinable @inline(__always)
  public static func pow<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Pow", x, y)
  }
  
  @inlinable @inline(__always)
  public static func relu6Grad<T>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Relu6Grad", gradients, features)
  }
  
  @inlinable @inline(__always)
  public static func reluGrad<T>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("ReluGrad", gradients, features)
  }
  
  
  
  @inlinable @inline(__always)
  public static func rsqrtGrad<T>(
    _ y: Tensor<T>,
    dy: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("RsqrtGrad", y, dy)
  }
  
  @inlinable @inline(__always)
  public static func seluGrad<T>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("ReluGrad", gradients, outputs)
  }
  
  @inlinable @inline(__always)
  public static func sigmoidGrad<T>(
    _ y: Tensor<T>,
    dy: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("SigmoidGrad", y, dy)
  }
  
  @inlinable @inline(__always)
  public static func softplusGrad<T>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("SoftplusGrad", gradients, features)
  }
  
  @inlinable @inline(__always)
  public static func softsignGrad<T>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("SoftsignGrad", gradients, features)
  }
  
  
  
  @inlinable @inline(__always)
  public static func squaredDifference<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("SquaredDifference", x, y)
  }
  
  @inlinable @inline(__always)
  public static func sub<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Sub", x, y)
  }
  
  @inlinable @inline(__always)
  public static func xdivy<T>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    dispatchBinary("Xdivy", x, y)
  }
}
