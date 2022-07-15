//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import MetalPerformanceShadersGraph

extension Context {
  public static func generateID(allocationSize: Int) -> UInt64 {
    withDispatchQueue {
      let output = Context.global.generateID(allocationSize: allocationSize)
      print("Generated ID: \(output)")
      return output
    }
  }
  
  public static func initialize(id: UInt64, _ body: (UnsafeMutableRawBufferPointer) -> Void) throws {
    try withDispatchQueue {
      let ctx = Context.global
      guard let allocation = try ctx.fetchAllocation(id: id) else {
        throw AllocationError.other("Tried to initialize memory that was deallocated.")
      }
      try allocation.materialize()
      try allocation.initialize(body)
    }
  }
  
  public static func read(id: UInt64, _ body: (UnsafeRawBufferPointer) -> Void) throws {
    try withDispatchQueue {
      let ctx = Context.global
      guard let allocation = try ctx.fetchAllocation(id: id) else {
        throw AllocationError.other("Tried to read from memory that was deallocated.")
      }
      try allocation.read(body)
    }
  }
  
  public static func release(id: UInt64) throws {
    try withDispatchQueue {
      try Context.global.release(id: id)
    }
  }
  
  @inline(never)
  private func _slowFail(id: UInt64) -> Never {
    if id < nextAllocationID {
      preconditionFailure("No memory has ever been allocated with ID #\(id).")
    } else {
      preconditionFailure("Allocation #\(id) has already been deallocated.")
    }
  }
  
  @inline(__always)
  internal func _compilerFetchAllocation(id: UInt64) -> Allocation {
    guard let allocation = allocations[id] else {
      _slowFail(id: id)
    }
    return allocation
  }
  
  @inline(__always)
  internal func _compilerRetain(id: UInt64) {
    guard let allocation = allocations[id] else {
      _slowFail(id: id)
    }
    allocation.referenceCount &+= 1
    if _slowPath(Allocation.debugInfoEnabled) {
      print("Allocation #\(id) jumped to a reference count of \(allocation.referenceCount).")
    }
  }
  
  @inline(__always)
  internal func _compilerRelease(_ allocation: Allocation) {
    let id = allocation.id
    let referenceCount = allocation.referenceCount &- 1
    allocation.referenceCount = referenceCount
    if referenceCount == 0 {
      allocations[id] = nil
      if _slowPath(Allocation.debugInfoEnabled) {
        if allocation.initialized {
          print("Allocation #\(id) was deallocated after being initialized.")
        } else {
          print("Allocation #\(id) was deallocated.")
        }
      }
    } else if _slowPath(Allocation.debugInfoEnabled) {
      print("Allocation #\(id) dropped to a reference count of \(referenceCount).")
    }
  }
}

private extension Context {
  func generateID(allocationSize: Int) -> UInt64 {
    let id = nextAllocationID
    nextAllocationID += 1
    allocations[id] = Allocation(id: id, size: allocationSize)
    return id
  }
  
  // Returns `nil` if the memory was deallocated. If the memory never existed in the first place, it
  // crashes because that's probably erroneous behavior on the frontend. Never retain the allocation
  // because that messes with ARC for deallocation. Instead, retain just the ID.
  func fetchAllocation(id: UInt64) throws -> Allocation? {
    guard id < nextAllocationID else {
      throw AllocationError.other("No memory has ever been allocated with ID #\(id).")
    }
    // Dictionary subscript returns an optional.
    return allocations[id]
  }
  
  func retain(id: UInt64) throws {
    guard id < nextAllocationID else {
      throw AllocationError.other("No memory has ever been allocated with ID #\(id).")
    }
    guard let allocation = allocations[id] else {
      // Catch memory management bugs.
      throw AllocationError.other("Allocation #\(id) has already been deallocated.")
    }
    allocation.referenceCount += 1
    if Allocation.debugInfoEnabled {
      print("Allocation #\(id) jumped to a reference count of \(allocation.referenceCount).")
    }
  }
  
  func release(id: UInt64) throws {
    // Catch memory management bugs.
    guard let allocation = allocations[id] else {
      throw AllocationError.other("Cannot deallocate something twice.")
    }
    allocation.referenceCount -= 1
    if allocation.referenceCount == 0 {
      if Allocation.debugInfoEnabled {
        if allocation.initialized {
          print("Allocation #\(id) was deallocated after being initialized.")
        } else {
          print("Allocation #\(id) was deallocated.")
        }
      }
      allocations[id] = nil
    } else {
      if Allocation.debugInfoEnabled {
        print("Allocation #\(id) dropped to a reference count of \(allocation.referenceCount).")
      }
    }
  }
}

// Throw an error instead of crashing so that unit tests can check what happens when you do
// something invalid. TODO: In the final product, prevent these errors from reaching the frontend.
// The only error the frontend can accept is something conforming to `PluggableDeviceError`.
enum AllocationError: Error {
  case exceededSystemRAM
  case other(String)
}

class Allocation {
  static var debugInfoEnabled = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_REFERENCE_COUNTING")
  var referenceCount: Int
  
  var id: UInt64
  var size: Int
  var isShared: Bool
  
  // TODO: Special storage mode for scalars or small chunks of constant memory. If `size` is under
  // 4 KB and memory is never mutated, it can be initialized on the CPU and passed into
  // `MTLComputeCommandEncoder.setBytes`. When in graph mode, there are similar mechanisms like
  // `MPSGraph.constant`.
  //
  // Will call the closure in `initialize(_:)` over the CPU-backed memory instead of the GPU-backed
  // memory. You may have to copy that CPU-backed memory to the `MTLBuffer`, but it's a very small
  // overhead. Take the overhead of a CPU function call + 2 * (4096 / (main memory bandwidth)).
  //
  // Wait to implement this until other things are debugged, because it adds too much complexity to
  // this prototype backend right now.
  
  // Extracting this to its own property improves performance. It probably skips a reference count
  // to the `MTLBuffer`. Also, it could be useful if there are multiple storage modes.
//  var materialized = false
  
  // Check this before performing any ops on the allocation. Otherwise, you're accessing undefined
  // memory.
  var initialized = false
  
  var mtlBuffer: MTLBuffer?
  // TODO: Shape - mutable for zero-cost reshape op
  // TODO: Data Type - mutable but only to match style of other properties
  var mpsMatrix: MPSMatrix?
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // The last command buffer that mutated this allocation's underlying memory.
  var lastModifiedCommandBufferID: Int?
  
  // Wait for this command buffer to complete so that you don't free it until it's safe to do so.
  // There should be no need to store this... or wait, maybe there is!
  var lastReferencedCommandBufferID: Int?
  
  init(id: UInt64, size: Int) {
    self.referenceCount = 1
    self.id = id
    self.size = size
    self.isShared = true
  }
  
  // Lazily allocates the physical memory. If the system ran out of memory, it flushes the command
  // stream. Then, it tries once more after all possible Metal memory is deallocated. If that
  // doesn't work, it crashes.
  @inline(__always)
  func materialize() throws {
    precondition(mtlBuffer == nil)
    guard let mtlBuffer = HeapAllocator.global.malloc(size: size, usingShared: true) else {
      throw AllocationError.other("An attempt to allocate a `MTLBuffer` returned `nil`.")
    }
    self.mtlBuffer = mtlBuffer
  }
  
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) throws {
    let contents = mtlBuffer!.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: size)
    body(ptr)
    initialized = true
  }
  
  func read(_ body: (UnsafeRawBufferPointer) -> Void) throws {
    let contents = self.mtlBuffer!.contents()
    let ptr = UnsafeRawBufferPointer(start: contents, count: size)
    body(ptr)
  }
  
  deinit {
    precondition({
      if let commandBufferID = lastReferencedCommandBufferID {
        return Context.global.commandBufferDictionary[commandBufferID] == nil
      } else {
        return true
      }
    }())
    
    precondition(referenceCount == 0)
    guard mtlBuffer != nil else {
      return
    }
    HeapAllocator.global.free(self.mtlBuffer!)
  }
}
