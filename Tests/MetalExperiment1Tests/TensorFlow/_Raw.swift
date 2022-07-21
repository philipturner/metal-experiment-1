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
    dispatchUnary("increment", input)
  }
  
  // Unary
  
  @inlinable @inline(__always)
  public static func abs<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("abs", input)
  }
  
  @inlinable @inline(__always)
  public static func acos<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("acos", input)
  }
  
  @inlinable @inline(__always)
  public static func acosh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("acosh", input)
  }
  
  @inlinable @inline(__always)
  public static func asin<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("asin", input)
  }
  
  @inlinable @inline(__always)
  public static func asinh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("asinh", input)
  }
  
  @inlinable @inline(__always)
  public static func atan<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("atan", input)
  }
  
  @inlinable @inline(__always)
  public static func atanh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("atanh", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func ceil<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("ceil", input)
  }
  
  @inlinable @inline(__always)
  public static func cos<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("cos", input)
  }
  
  @inlinable @inline(__always)
  public static func cosh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("cosh", input)
  }
  
  @inlinable @inline(__always)
  public static func elu<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("elu", features)
  }
  
  @inlinable @inline(__always)
  public static func exp<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("exp", input)
  }
  
  @inlinable @inline(__always)
  public static func expm1<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("expm1", input)
  }
  
  @inlinable @inline(__always)
  public static func floor<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("floor", input)
  }
  
  
  
  @inlinable @inline(__always)
  public static func leakyRelu<T>(
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    dispatchUnary("leakyRelu", features, alpha)
  }
  
  @inlinable @inline(__always)
  public static func log<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("log", input)
  }
  
  @inlinable @inline(__always)
  public static func log1p<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("log1p", input)
  }
  
  @inlinable @inline(__always)
  public static func logicalNot<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("logicalNot", input)
  }
  
  @inlinable @inline(__always)
  public static func neg<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("neg", input)
  }
  
  @inlinable @inline(__always)
  public static func relu<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("relu", features)
  }
  
  @inlinable @inline(__always)
  public static func relu6<T>(features: Tensor<T>) -> Tensor<T> {
    dispatchUnary("relu6", features)
  }
  
  @inlinable @inline(__always)
  public static func round<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("round", input)
  }
}
