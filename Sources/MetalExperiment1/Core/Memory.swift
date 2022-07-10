//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Metal
import MetalPerformanceShadersGraph

extension Context {
  static func generateID(allocationSize: Int) -> UInt64 {
    dispatchQueue.sync {
      Context.global.generateID(allocationSize: allocationSize)
    }
  }
  
  static func deallocate(id: UInt64) {
    dispatchQueue.sync {
      Context.global.deallocate(id: id)
    }
  }
}

private extension Context {
  func generateID(allocationSize: Int) -> UInt64 {
    let output = nextAllocationID
    nextAllocationID += 1
    allocations[output] = Allocation(size: allocationSize)
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
  var size: Int
  var isShared: Bool
  var mtlBuffer: MTLBuffer?
  // TODO: Shape
  // TODO: Data Type
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // TODO: Store the latest command buffer ID that references this. To make a conditional barrier
  // that waits until this specific command buffer completes, always store references to any command
  // buffers that are currently executing.
  var referencedCommandBufferID: Int?
  
  init(size: Int) {
    self.size = size
    self.isShared = true
  }
  
  // Lazily allocates the physical memory. If the system ran out of memory, it flushes the command
  // stream. Then, it tries once more after all possible Metal memory is deallocated. If that
  // doesn't work, it crashes.
  func materialize() {
    // Use PyTorch's optimized allocator later. For now, just make and debug an allocator that
    // works.
    guard let mtlBuffer = Context.global.device.makeBuffer(length: size) else {
      // TODO: Flush the command stream.
      fatalError("System ran out of memory.")
    }
    self.mtlBuffer = mtlBuffer
    self.isShared = true
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    guard let mtlBuffer = mtlBuffer else {
      fatalError("Initialized memory with a null underlying `MTLBuffer`.")
    }
    if isShared {
      let contents = mtlBuffer.contents()
      let ptr = UnsafeMutableRawBufferPointer(start: contents, count: size)
      body(ptr)
    } else {
      // TODO: Append a command that will copy the memory.
      fatalError("Haven't implemented copying memory to a discrete GPU.")
    }
  }
  
  // Flushes the command stream. On a discrete GPU, it appends one command to copy data from the GPU
  // before flushing the command stream. You must copy the data inside the pointer, because it will
  // deallocate or become undefined after the closure finishes.
  func read(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    guard let bufferToCopy = mtlBuffer else {
      fatalError("Read from memory with a null underlying `MTLBuffer`.")
    }
    var bufferToRead: MTLBuffer
    if isShared {
      bufferToRead = bufferToCopy
    } else {
      // TODO: Allocate a shared buffer, using a special heap reserved for shared memory.
      // TODO: Append a command that will copy the memory.
      fatalError("Haven't implemented copying memory to a discrete GPU.")
    }
    // TODO: Special barrier, waits until the last command buffer referencing this finishes. If the
    // last command referencing this hasn't yet been encoded, place a MTLEvent in the next command
    // buffer. That way, you synchronize without dividing into two separate command buffers (more
    // overhead). It would also reduce I/O bottlenecks if you have several calls to `read` in a row.
    //
    // You can also prioritize the copying op, if on a discrete GPU. Prepend the copying op to the
    // beginning of `bufferedOperations`, unless one of those operations references it. This
    // violates sequential order of execution, but produces the same end result.
    Context._unsafeBarrier()
    
    let contents = bufferToRead.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: size)
    body(ptr)
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
