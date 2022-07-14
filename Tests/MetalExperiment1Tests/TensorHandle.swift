//
//  TensorHandle.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

@testable import MetalExperiment1

class TensorHandle {
  private(set) var id: UInt64
  private(set) var count: Int
  
  init(unsafeUninitializedCount count: Int) {
    let allocationSize = count * MemoryLayout<Float>.stride
    self.id = Context.generateID(allocationSize: allocationSize)
    self.count = count
  }
  
  convenience init(repeating repeatedValue: Float, count: Int) {
    self.init(unsafeUninitializedCount: count)
    try! Context.initialize(id: id) { bufferPointer in
      let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
      ptr.initialize(repeating: repeatedValue)
    }
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

enum _Raw {
  static func increment(_ input: TensorHandle) -> TensorHandle {
    Context.withDispatchQueue {
      let output = TensorHandle(unsafeUninitializedCount: input.count)
      Context.commitIncrement(inputID: input.id, outputID: output.id)
      return output
    }
  }
}
