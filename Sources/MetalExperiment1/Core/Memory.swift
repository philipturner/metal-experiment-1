//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Metal
import MetalPerformanceShadersGraph

extension Context {
  func generateID() -> UInt64 {
    let output = nextAllocationID
    nextAllocationID += 1
    allocations[output] = Allocation()
    return output
  }
  
  // Returns `nil` if the memory was deallocated. If the memory never existed in the first place, it
  // crashes because that's probably erroneous behavior on the frontend. Never retain the allocation
  // because that messes with ARC for deallocation. Instead, retain just the ID.
  func fetchAllocation(id: UInt64) -> Allocation? {
    guard id < nextAllocationID else {
      fatalError("No memory has ever been allocated with ID #\(id).")
    }
    // Dictionary subscript returns an optional.
    return allocations[id]
  }
  
  // Makes the ID invald and releases the Swift object. Its erasure prevents memory leaks in a
  // program running forever.
  func deallocate(id: UInt64) {
    // Catch reference-counting bugs.
    precondition(allocations[id] != nil, "Cannot deallocate something twice.")
    allocations[id] = nil
  }
}

class Allocation {
  var mtlBuffer: MTLBuffer?
  // TODO: Shape
  // TODO: Data Type
  var mpsGraphTensorData: MPSGraphTensorData?
  
  init() {}
  
  struct SystemOutOfMemory: Error {}
  
  // Lazily allocates the physical memory. If the system ran out of memory, it throws an error.
  // The caller should flush the command stream and try again. If this function automatically
  // flushed the command stream, it would be hard to track control flow and performance.
  func materialize() throws {
    throw SystemOutOfMemory()
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  func initialize(_ body: (UnsafeMutableRawPointer) -> Void) {
    guard let mtlBuffer = mtlBuffer else {
      fatalError("Initialized memory with a null underlying `MTLBuffer`.")
    }
    switch mtlBuffer.storageMode {
    case .shared:
      let contents = mtlBuffer.contents()
      body(contents)
    case .private:
      // TODO
      fatalError("Haven't implemented copying memory to a discrete GPU.")
    default:
      fatalError("Memory storage mode \(mtlBuffer.storageMode) should never be used in backend.")
    }
  }
  
  deinit {
    if mpsGraphTensorData != nil {
      mpsGraphTensorData = nil
    }
    guard mtlBuffer != nil else {
      return
    }
    self.mtlBuffer = nil
  }
}
