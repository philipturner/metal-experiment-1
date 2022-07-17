//
//  Tensor.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import MetalExperiment1

// MARK: - PluggableDeviceTensorHandle

// Mirrors the functionality of `TFETensorHandle`.
public class PluggableDeviceTensorHandle {
  public let _cTensorHandle: UInt64
  
  public let rank: Int
  
  @usableFromInline
  internal var _shape: [Int]?
  
  public init(_owning base: UInt64, rank: Int) {
    self._cTensorHandle = base
    self.rank = rank
  }
  
  public init(_owning base: UInt64, rank: Int, shape: [Int]) {
    self._cTensorHandle = base
    self.rank = rank
    self._shape = shape
  }
  
  deinit {
    Context.releaseBuffer(_cTensorHandle)
  }
  
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get {
      if _shape == nil {
        _shape = Array(unsafeUninitializedCapacity: rank) { bufferPointer, count in
          count = rank
          Context.copyBufferShape(_cTensorHandle, bufferPointer)
        }
      }
      return TensorShape(_shape.unsafelyUnwrapped)
    }
  }
}

// MARK: - TensorHandle

public struct TensorHandle<Scalar> {
  @usableFromInline let handle: PluggableDeviceTensorHandle
  
  public var _cTensorHandle: UInt64 { handle._cTensorHandle }
  
  public init(_owning cTensorHandle: UInt64, rank: Int) {
    self.handle = PluggableDeviceTensorHandle(_owning: cTensorHandle, rank: rank)
  }
  
  public init(_owning cTensorHandle: UInt64, rank: Int, shape: [Int]) {
    self.handle = PluggableDeviceTensorHandle(_owning: cTensorHandle, rank: rank, shape: shape)
  }
  
  public init(handle: PluggableDeviceTensorHandle) {
    self.handle = handle
  }
  
  @inlinable
  public init(
    shape: [Int],
    scalarsInitializer: (UnsafeMutablePointer<Scalar>) -> Void
  ) {
    let (cTensorHandle, rank) = shape.withUnsafeBufferPointer {
      Context.allocateBuffer(Scalar.self, $0)
    }
    Context.initializeBuffer(cTensorHandle) { buffer in
      let pointer = buffer.assumingMemoryBound(to: Scalar.self)
      scalarsInitializer(pointer.baseAddress!)
    }
    self.init(_owning: cTensorHandle, rank: rank, shape: shape)
  }
  
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  /// The shape of the `Tensor`.
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }
  
  @usableFromInline
  @inline(never)
  func makeHostCopy() -> [Scalar] {
    var output: [Scalar]?
    Context.readBuffer(_cTensorHandle) { tensorBuffer in
      let tensorPointer = tensorBuffer.assumingMemoryBound(to: Scalar.self)
      output = Array(unsafeUninitializedCapacity: tensorPointer.count) { arrayPointer, count in
        _ = arrayPointer.initialize(from: tensorPointer)
        count = tensorPointer.count
      }
    }
    return output!
  }
}

// MARK: - Tensor

public struct Tensor<Scalar> {
  public let handle: TensorHandle<Scalar>
  
  @usableFromInline
  internal var _isScalarZero = false
  
  @inlinable
  public init(handle: TensorHandle<Scalar>) {
    self.handle = handle
  }
  
  public var _rawTensorHandle: UInt64 { return handle._cTensorHandle }
  public var scalarType: Any.Type { return Scalar.self }
  
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }
  
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }

  @inlinable
  public var scalarCount: Int {
    @_semantics("autodiff.nonvarying")
    get { shape.contiguousSize }
  }
  
  public var scalars: [Scalar] {
    return handle.makeHostCopy()
  }
  
  @inlinable
  public init(repeating repeatedValue: Scalar, shape: TensorShape) {
    let handle = TensorHandle<Scalar>(shape: shape.dimensions) {
      $0.initialize(repeating: repeatedValue, count: shape.contiguousSize)
    }
    self.init(handle: handle)
  }
  
  @inlinable
  public func incremented() -> Tensor {
    return _Raw.increment(self)
  }
}

// MARK: - _Raw

// Differs from the old S4TF in that the function bodies aren't emitted into the client.
public enum _Raw {
  @inline(__always)
  public static func increment<T>(_ input: Tensor<T>) -> Tensor<T> {
    return decodeOutputs { outputs in
      encodeInputs(input) { inputs in
        let name = encodeName("increment")
        let attributes = encodeAttributes()
        Context.executeOperation(name, attributes, inputs, outputs)
      }
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
