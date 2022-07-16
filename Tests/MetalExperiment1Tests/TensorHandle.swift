//
//  TensorHandle.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import MetalExperiment1

// Mirrors the functionality of `TFETensorHandle`.
//
// TODO: Rename to `PluggableDeviceTensorHandle` and wrap in `TensorHandle<Scalar>`, allowing more
// data types than just `Float`.
public class TensorHandle {
  // Vector elements:
  // 0..<1 - id
  // 1..<2 - rank
  // 2..<3 - shape exists
  // 4..<8 - shape
  @usableFromInline
  var storage: SIMD8<UInt64>
  
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
  
  // TODO: Implement `shape` in a performant way, using SIMD storage until an array is requested.
//  @inlinable
//  public var shape: TensorShape {
//    @_semantics("autodiff.nonvarying")
//    get {
//      fatalError("`TensorShape` not yet implemented.")
//    }
//  }
  
}

extension TensorHandle {
  // This code will become part of `TensorHandle<Scalar>`, which wraps the future incarnation of
  // the current `TensorHandle`.
  @inlinable
  convenience init(repeating repeatedValue: Float, count: Int) {
    let (cTensorHandle, rank) = withUnsafeTemporaryAllocation(of: Int.self, capacity: 1) { shape in
      shape[0] = count
      return Context.allocateTensor(Float.self, UnsafeBufferPointer(shape))
    }
    self.init(_owning: cTensorHandle, rank: rank)
    
    Context.initializeTensor(cTensorHandle) { buffer in
      let pointer = buffer.assumingMemoryBound(to: Float.self)
      pointer.initialize(repeating: repeatedValue)
    }
  }
  
  // TODO: Rename to `makeHostCopy()` and update tests.
  @usableFromInline
  @inline(never)
  func copyScalars() -> [Float] {
    var output: [Float]?
    Context.readTensor(_cTensorHandle) { tensorBuffer in
      let tensorPointer = tensorBuffer.assumingMemoryBound(to: Float.self)
      output = Array(unsafeUninitializedCapacity: tensorPointer.count) { arrayPointer, count in
        _ = arrayPointer.initialize(from: tensorPointer)
        count = tensorPointer.count
      }
    }
    return output!
  }
}

extension TensorHandle {
  func incremented() -> TensorHandle {
    _Raw.increment(self)
  }
}

// MARK: - _Raw

enum _Raw {
  static func increment(_ input: TensorHandle) -> TensorHandle {
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
fileprivate func encodeInputs(
  _ input1: TensorHandle,
  _ body: (UnsafeBufferPointer<UInt64>) -> Void
) {
  withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 1) { bufferPointer in
    bufferPointer[0] = input1._cTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inline(__always)
fileprivate func encodeInputs(
  _ input1: TensorHandle,
  _ input2: TensorHandle,
  _ body: (UnsafeBufferPointer<UInt64>) -> Void
) {
  withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 2) { bufferPointer in
    bufferPointer[0] = input1._cTensorHandle
    bufferPointer[1] = input2._cTensorHandle
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inline(__always)
fileprivate func decodeOutputAtom(
  _ bufferPointer: UnsafeMutableBufferPointer<(UInt64, Int)>, _ index: Int
) -> TensorHandle {
  TensorHandle(_owning: bufferPointer[index].0, rank: bufferPointer[index].1)
}

@inline(__always)
fileprivate func decodeOutputs(
  _ body: (UnsafeMutableBufferPointer<(UInt64, Int)>) -> Void
) -> TensorHandle {
  withUnsafeTemporaryAllocation(of: (UInt64, Int).self, capacity: 1) { bufferPointer in
    body(bufferPointer)
    return decodeOutputAtom(bufferPointer, 0)
  }
}


@inline(__always)
fileprivate func decodeOutputs(
  _ body: (UnsafeMutableBufferPointer<(UInt64, Int)>) -> Void
) -> (TensorHandle, TensorHandle) {
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
