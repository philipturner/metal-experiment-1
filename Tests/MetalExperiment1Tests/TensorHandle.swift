//
//  TensorHandle.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import MetalExperiment1

class TensorHandle {
  private(set) var id: UInt64
  private(set) var count: Int
  
  @inline(__always)
  init(owning id: UInt64, byteCount: Int) {
    self.id = id
    self.count = byteCount / MemoryLayout<Float>.stride
  }
  
  convenience init(repeating repeatedValue: Float, count: Int) {
    let allocationSize = count * MemoryLayout<Float>.stride
    let id = Context.generateID(allocationSize: allocationSize)
    try! Context.initialize(id: id) { bufferPointer in
      let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
      ptr.initialize(repeating: repeatedValue)
    }
    self.init(owning: id, byteCount: allocationSize)
  }
  
  func copyScalars() -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: count) { destination, count in
      count = self.count
      try! Context.read(id: id) { bufferPointer in
        let source = bufferPointer.assumingMemoryBound(to: Float.self)
        _ = destination.initialize(from: source)
      }
    }
  }
  
  deinit {
    try! Context.release(id: id)
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
    bufferPointer[0] = input1.id
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
    bufferPointer[0] = input1.id
    bufferPointer[1] = input2.id
    body(UnsafeBufferPointer(bufferPointer))
  }
}

@inline(__always)
fileprivate func decodeOutputAtom(
  _ bufferPointer: UnsafeMutableBufferPointer<(UInt64, Int)>, _ index: Int
) -> TensorHandle {
  TensorHandle(owning: bufferPointer[index].0, byteCount: bufferPointer[index].1)
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

// MARK: - TFEncodable

protocol TFEncodable {
  func createAtom() -> (UInt64, UInt64)
}

// TODO: Set the `Float` TF type enumeration as an attribute's value.

extension Int32: TFEncodable {
  @inline(__always)
  func createAtom() -> (UInt64, UInt64) {
    (UInt64(truncatingIfNeeded: self), 0)
  }
}
