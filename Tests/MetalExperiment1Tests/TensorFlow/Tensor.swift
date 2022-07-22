//
//  Tensor.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import MetalExperiment1

// TODO: Mark all `OpaquePointer` here as `CTensorHandle`.
public typealias CTensorHandle = OpaquePointer

// MARK: - PluggableDeviceTensorHandle

// Mirrors the functionality of `TFETensorHandle`.
public class PluggableDeviceTensorHandle {
  public let _cTensorHandle: CTensorHandle
  
  public let rank: Int
  
  @usableFromInline
  internal var _shape: [Int]?
  
  public init(_owning base: CTensorHandle, rank: Int) {
    self._cTensorHandle = base
    self.rank = rank
  }
  
  public init(_owning base: CTensorHandle, rank: Int, shape: [Int]) {
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

public struct TensorHandle<Scalar: _TensorFlowDataTypeCompatible> {
  @usableFromInline let handle: PluggableDeviceTensorHandle
  
  public var _cTensorHandle: CTensorHandle { handle._cTensorHandle }
  
  public init(_owning cTensorHandle: CTensorHandle, rank: Int) {
    self.handle = PluggableDeviceTensorHandle(_owning: cTensorHandle, rank: rank)
  }
  
  public init(_owning cTensorHandle: CTensorHandle, rank: Int, shape: [Int]) {
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

public struct Tensor<Scalar: TensorFlowScalar> {
  public let handle: TensorHandle<Scalar>
  
  @usableFromInline
  internal var _isScalarZero = false
  
  @inlinable
  public init(handle: TensorHandle<Scalar>) {
    self.handle = handle
  }
  
  public var _rawTensorHandle: CTensorHandle { return handle._cTensorHandle }
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
}
