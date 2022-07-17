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
  // Vector elements:
  // 0..<1 - id
  // 1..<2 - rank
  // 2..<3 - shape exists
  // 3..<8 - shape
  @usableFromInline
  internal var storage: SIMD8<UInt64>
  
  @inlinable
  public var _cTensorHandle: UInt64 { storage[0] }
  
  public init(_owning base: UInt64, rank: Int) {
    storage = .zero
    storage[0] = base
    storage[1] = UInt64(truncatingIfNeeded: rank)
  }
  
  deinit {
    Context.deleteTensor(_cTensorHandle)
  }
  
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get {
      Int(truncatingIfNeeded: storage[1])
    }
  }
  
  @usableFromInline @inline(never)
  internal func _fetchStorage() {
    withUnsafeTemporaryAllocation(of: Int.self, capacity: rank) { bufferPointer in
      Context.copyTensorShape(_cTensorHandle, bufferPointer)
      for i in bufferPointer.indices {
        storage[3 + i] = UInt64(truncatingIfNeeded: bufferPointer[i])
      }
    }
    storage[2] = 1
  }
  
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get {
      if storage[2] == 0 {
        _fetchStorage()
      }
      let castedStorage = SIMD8<Int>(truncatingIfNeeded: storage)
      return TensorShape(tensorHandleStorage: castedStorage)
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
  
  public init(handle: PluggableDeviceTensorHandle) {
    self.handle = handle
  }
  
  @inlinable
  public init(
    shape: [Int],
    scalarsInitializer: (UnsafeMutablePointer<Scalar>) -> Void
  ) {
    let (cTensorHandle, rank) = shape.withUnsafeBufferPointer {
      Context.allocateTensor(Scalar.self, $0)
    }
    Context.initializeTensor(cTensorHandle) { buffer in
      let pointer = buffer.assumingMemoryBound(to: Scalar.self)
      scalarsInitializer(pointer.baseAddress!)
    }
    self.init(_owning: cTensorHandle, rank: rank)
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
    Context.readTensor(_cTensorHandle) { tensorBuffer in
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

//extension TensorHandle {
//  // This code will become part of `TensorHandle<Scalar>`, which wraps the future incarnation of
//  // the current `TensorHandle`.
//  @inlinable
//  convenience init(repeating repeatedValue: Float, count: Int) {
//    let (cTensorHandle, rank) = withUnsafeTemporaryAllocation(of: Int.self, capacity: 1) { shape in
//      shape[0] = count
//      return Context.allocateTensor(Float.self, UnsafeBufferPointer(shape))
//    }
//    self.init(_owning: cTensorHandle, rank: rank)
//
//    Context.initializeTensor(cTensorHandle) { buffer in
//      let pointer = buffer.assumingMemoryBound(to: Float.self)
//      pointer.initialize(repeating: repeatedValue)
//    }
//  }
//}
//
//extension TensorHandle {
//  func incremented() -> TensorHandle {
//    _Raw.increment(self)
//  }
//}

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
