//
//  _Raw.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

import MetalExperiment1

// MARK: - _Raw Helpers

@inline(__always)
fileprivate func dispatchUnary<T>(_ name: StaticString, _ input: Tensor<T>) -> Tensor<T> {
  return decodeOutputs { outputs in
    encodeInputs(input) { inputs in
      let name = encodeName(name)
      let attributes = encodeAttributes()
      Context.executeOperation(name, attributes, inputs, outputs)
    }
  }
}

@inline(__always)
fileprivate func encodeName(_ name: StaticString) -> UnsafeRawBufferPointer {
  let start = name.utf8Start
  let count = name.utf8CodeUnitCount
  return UnsafeRawBufferPointer(start: start, count: count)
}

@inline(__always)
fileprivate func encodeAttributes() -> UnsafeRawBufferPointer {
  return UnsafeRawBufferPointer(start: nil, count: 0)
}

@inline(__always)
fileprivate func encodeInputs<T0>(
  _ input1: Tensor<T0>,
  _ body: (UnsafeBufferPointer<UInt64>) -> Void
) {
  withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 1) { bufferPointer in
    bufferPointer[0] = input1._rawTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inline(__always)
fileprivate func encodeInputs<T0, T1>(
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

@inline(__always)
fileprivate func decodeOutputAtom<T>(
  _ ptr: UnsafeMutableBufferPointer<(UInt64, Int)>, _ index: Int
) -> Tensor<T> {
  let handle = TensorHandle<T>(_owning: ptr[index].0, rank: ptr[index].1)
  return Tensor(handle: handle)
}

@inline(__always)
fileprivate func decodeOutputs<T0>(
  _ body: (UnsafeMutableBufferPointer<(UInt64, Int)>) -> Void
) -> (Tensor<T0>) {
  withUnsafeTemporaryAllocation(of: (UInt64, Int).self, capacity: 1) { bufferPointer in
    body(bufferPointer)
    return (
      decodeOutputAtom(bufferPointer, 0)
    )
  }
}


@inline(__always)
fileprivate func decodeOutputs<T0, T1>(
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

protocol PluggableDeviceEncodable {
  func createAtom() -> (UInt64, UInt64)
}

// TODO: Set the `Float` TF_DataType enumeration as an attribute's value. The raw value of this
// enumeration is `Int32`.

extension Int32: PluggableDeviceEncodable {
  @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(truncatingIfNeeded: self), 0)
  }
}

// MARK: - _Raw

// Differs from the old S4TF in that the function bodies aren't emitted into the client.
public enum _Raw {
  @inline(__always)
  public static func abs<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("abs", input)
  }
  
  @inline(__always)
  public static func acos<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("acos", input)
  }
  
  @inline(__always)
  public static func acosh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("acosh", input)
  }
  
  @inline(__always)
  public static func asin<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("asin", input)
  }
  
  @inline(__always)
  public static func asinh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("asinh", input)
  }
  
  @inline(__always)
  public static func atan<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("atan", input)
  }
  
  @inline(__always)
  public static func atanh<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("atanh", input)
  }
  
  @inline(__always)
  public static func increment<T>(_ input: Tensor<T>) -> Tensor<T> {
    dispatchUnary("increment", input)
  }
}
