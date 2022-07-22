//
//  _Raw.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

import MetalExperiment1

// MARK: - _Raw Helpers

@inlinable @inline(__always)
func dispatchUnary<T>(
  _ name: StaticString,
  _ input: Tensor<T>
) -> Tensor<T> {
  return decodeOutputs { outputs in
    encodeInputs(input) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
    }
  }
}

@inlinable @inline(__always)
func dispatchUnary<T, A0: PluggableDeviceEncodable>(
  _ name: StaticString,
  _ input: Tensor<T>,
  _ attribute1: A0
) -> Tensor<T> {
  return decodeOutputs { outputs in
    encodeInputs(input) { inputs in
      encodeAttributes(attribute1) { attributes in
        let name = encodeName(name)
        Context.executeOperation(name, attributes, inputs, outputs)
      }
    }
  }
}

@inlinable @inline(__always)
func dispatchUnaryRelational<T>(
  _ name: StaticString,
  _ input: Tensor<T>
) -> Tensor<Bool> {
  return decodeOutputs { outputs in
    encodeInputs(input) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
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
func encodeAttributes<T0: PluggableDeviceEncodable>(
  _ input1: T0,
  _ body: (UnsafeRawBufferPointer) -> Void
) {
  withUnsafeTemporaryAllocation(of: (UInt64, UInt64).self, capacity: 1) { bufferPointer in
    bufferPointer[0] = input1.createAtom()
    body(UnsafeRawBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func encodeInputs<T0>(
  _ input1: Tensor<T0>,
  _ body: (UnsafeBufferPointer<UInt64>) -> Void
) {
  withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 1) { bufferPointer in
    bufferPointer[0] = input1._rawTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func encodeInputs<T0, T1>(
  _ input1: Tensor<T0>,
  _ input2: Tensor<T1>,
  _ body: (UnsafeBufferPointer<UInt64>) -> Void
) {
  withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 2) { bufferPointer in
    bufferPointer[0] = input1._rawTensorHandle
    bufferPointer[1] = input2._rawTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inlinable @inline(__always)
func decodeOutputAtom<T>(
  _ ptr: UnsafeMutableBufferPointer<(UInt64, Int)>, _ index: Int
) -> Tensor<T> {
  let handle = TensorHandle<T>(_owning: ptr[index].0, rank: ptr[index].1)
  return Tensor(handle: handle)
}

@inlinable @inline(__always)
func decodeOutputs<T0>(
  _ body: (UnsafeMutableBufferPointer<(UInt64, Int)>) -> Void
) -> (Tensor<T0>) {
  withUnsafeTemporaryAllocation(of: (UInt64, Int).self, capacity: 1) { bufferPointer in
    body(bufferPointer)
    return (
      decodeOutputAtom(bufferPointer, 0)
    )
  }
}

@inlinable @inline(__always)
func decodeOutputs<T0, T1>(
  _ body: (UnsafeMutableBufferPointer<(UInt64, Int)>) -> Void
) -> (Tensor<T0>, Tensor<T1>) {
  withUnsafeTemporaryAllocation(of: (UInt64, Int).self, capacity: 2) { bufferPointer in
    body(bufferPointer)
    return (
      decodeOutputAtom(bufferPointer, 0),
      decodeOutputAtom(bufferPointer, 1)
    )
  }
}

// MARK: - PluggableDeviceEncodable

@usableFromInline
protocol PluggableDeviceEncodable {
  @inlinable
  func createAtom() -> (UInt64, UInt64)
}

extension Float: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(bitPattern), 0)
  }
}

extension Double: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(bitPattern), 0)
  }
}

extension Int32: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(truncatingIfNeeded: self), 0)
  }
}

extension Int64: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(truncatingIfNeeded: self), 0)
  }
}

extension UInt32: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(self), 0)
  }
}

extension UInt64: PluggableDeviceEncodable {
  @inlinable @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(self), 0)
  }
}

// MARK: - _Raw

public enum _Raw {
  @inlinable @inline(__always)
  public static func increment<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Increment", input)
  }
  
  // Unary
  
  @inlinable @inline(__always)
  public static func abs<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Abs", input)
  }
  
  @inlinable @inline(__always)
  public static func acos<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Acos", input)
  }
  
  @inlinable @inline(__always)
  public static func acosh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Acosh", input)
  }
  
  @inlinable @inline(__always)
  public static func asin<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Asin", input)
  }
  
  @inlinable @inline(__always)
  public static func asinh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Asinh", input)
  }
  
  @inlinable @inline(__always)
  public static func atan<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Atan", input)
  }
  
  @inlinable @inline(__always)
  public static func atanh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Atanh", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func ceil<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Ceil", input)
  }
  
  @inlinable @inline(__always)
  public static func cos<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Cos", input)
  }
  
  @inlinable @inline(__always)
  public static func cosh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Cosh", input)
  }
  
  @inlinable @inline(__always)
  public static func elu<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Elu", features)
  }
  
  @inlinable @inline(__always)
  public static func exp<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Exp", input)
  }
  
  @inlinable @inline(__always)
  public static func expm1<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Expm1", input)
  }
  
  @inlinable @inline(__always)
  public static func floor<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Floor", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func isFinite<T>(_ input: Tensor<T>) -> Tensor<Bool> {
    dispatchUnaryRelational("IsFinite", input)
  }
  
  @inlinable @inline(__always)
  public static func isInf<T>(_ input: Tensor<T>) -> Tensor<Bool> {
    dispatchUnaryRelational("IsInf", input)
  }
  
  @inlinable @inline(__always)
  public static func isNan<T>(_ input: Tensor<T>) -> Tensor<Bool> {
    dispatchUnaryRelational("IsNan", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func leakyRelu<T>(
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    dispatchUnary("LeakyRelu", features, alpha)
  }
  
  @inlinable @inline(__always)
  public static func log<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Log", input)
  }
  
  @inlinable @inline(__always)
  public static func log1p<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Log1p", input)
  }
  
  @inlinable @inline(__always)
  public static func logicalNot<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("LogicalNot", input)
  }
  
  @inlinable @inline(__always)
  public static func neg<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Neg", input)
  }
  
  @inlinable @inline(__always)
  public static func relu<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Relu", features)
  }
  
  @inlinable @inline(__always)
  public static func relu6<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Relu6", features)
  }
  
  @inlinable @inline(__always)
  public static func round<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Round", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func rsqrt<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Rsqrt", input)
  }
  
  @inlinable @inline(__always)
  public static func selu<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Selu", input)
  }
  
  @inlinable @inline(__always)
  public static func sigmoid<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Sigmoid", input)
  }
  
  @inlinable @inline(__always)
  public static func sign<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Sign", input)
  }
  
  @inlinable @inline(__always)
  public static func sin<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Sin", input)
  }
  
  @inlinable @inline(__always)
  public static func sinh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Sinh", input)
  }
  
  @inlinable @inline(__always)
  public static func softplus<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("Softplus", input)
  }
}
