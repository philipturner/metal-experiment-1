//
//  Tensor.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import MetalExperiment1
import Darwin

public typealias CTensorHandle = OpaquePointer

// MARK: - TFETensorHandle

public class TFETensorHandle {
  public let _cTensorHandle: CTensorHandle
  
  @usableFromInline
  internal var _shape: [Int]?
  
  public init(_owning base: CTensorHandle) {
    // Only check tensor leaks in debug mode because that's costly.
    assert({
      FrontendContext.local.globalTensorCount += 1
      return true
    }())
    self._cTensorHandle = base
  }
  
  public init(_owning base: CTensorHandle, shape: [Int]) {
    // Only check tensor leaks in debug mode because that's costly.
    assert({
      FrontendContext.local.globalTensorCount += 1
      return true
    }())
    self._cTensorHandle = base
    self._shape = shape
  }
  
  deinit {
    // Only check tensor leaks in debug mode because that's costly.
    assert({
      FrontendContext.local.globalTensorCount -= 1
      return true
    }())
    
    // TODO: Fetch pluggable device from `_ExecutionContext` fast-path.
//    Context.global.releaseTensor(_cTensorHandle)
    let handle = PluggableDeviceTensorHandle(_cTensorHandle).pluggableDeviceHandle
    let device = _ExecutionContext.global.getDevice(handle: handle)
    device.releaseTensor(_cTensorHandle)
  }
  
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get {
      PluggableDeviceTensorHandle(_cTensorHandle).rank
    }
  }
  
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get {
      if _slowPath(_shape == nil) {
        _shape = Array(PluggableDeviceTensorHandle(_cTensorHandle).shape)
      }
      return TensorShape(_shape.unsafelyUnwrapped)
    }
  }
}

// MARK: - TensorHandle

public struct TensorHandle<Scalar: _TensorFlowDataTypeCompatible> {
  @usableFromInline let handle: TFETensorHandle
  
  public var _cTensorHandle: CTensorHandle { handle._cTensorHandle }
  
  public init(_owning cTensorHandle: CTensorHandle) {
    self.handle = TFETensorHandle(_owning: cTensorHandle)
  }
  
  public init(_owning cTensorHandle: CTensorHandle, shape: [Int]) {
    self.handle = TFETensorHandle(_owning: cTensorHandle, shape: shape)
  }
  
  @inlinable
  public init(
    shape: [Int],
    scalarsInitializer: (UnsafeMutablePointer<Scalar>) -> Void
  ) {
    let cTensorHandle = shape.withUnsafeBufferPointer {
      Context.global.createTensor(Scalar.tensorFlowDataType._cDataType, $0, { buffer in
        let pointer = buffer.assumingMemoryBound(to: Scalar.self)
        scalarsInitializer(pointer.baseAddress!)
      })
    }
    self.init(_owning: cTensorHandle, shape: shape)
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
    Context.global.readTensor(_cTensorHandle, false) { tensorBuffer in
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
  
  @inlinable
  public init(shape: TensorShape, scalars: [Scalar]) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    let handle = TensorHandle<Scalar>(shape: shape.dimensions) {
      memcpy($0, scalars, scalars.count * MemoryLayout<Scalar>.stride)
    }
    self.init(handle: handle)
  }
  
  @inlinable
  public init(_ scalars: [Scalar]) {
    self.init(shape: [scalars.count], scalars: scalars)
  }
}
