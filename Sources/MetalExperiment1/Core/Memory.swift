//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import MetalPerformanceShadersGraph

extension Context {
  static func generateID(allocationSize: Int) -> UInt64 {
    dispatchQueue.sync {
      Context.global.generateID(allocationSize: allocationSize)
    }
  }
  
  static func initialize(id: UInt64, _ body: (UnsafeMutableRawBufferPointer) -> Void) throws {
    try dispatchQueue.sync {
      let ctx = Context.global
      guard let allocation = try ctx.fetchAllocation(id: id) else {
        throw AllocationError("Tried to initialize memory that was deallocated.")
      }
      try allocation.materialize()
      try allocation.initialize(body)
    }
  }
  
  static func read(id: UInt64, _ body: (UnsafeRawBufferPointer) -> Void) throws {
    try dispatchQueue.sync {
      let ctx = Context.global
      guard let allocation = try ctx.fetchAllocation(id: id) else {
        throw AllocationError("Tried to read from memory that was deallocated.")
      }
      try allocation.read(body)
    }
  }
  
  static func release(id: UInt64) throws {
    try dispatchQueue.sync {
      try Context.global.release(id: id)
    }
  }
  
  // Remove "_unsafe" wrappers once these are scoped as public or not.
  internal func _unsafeGenerateID(allocationSize: Int) -> UInt64 {
    self.generateID(allocationSize: allocationSize)
  }
  
  internal func _unsafeFetchAllocation(id: UInt64) throws -> Allocation? {
    try self.fetchAllocation(id: id)
  }
  
  internal func _unsafeRetain(id: UInt64) throws {
    try self.retain(id: id)
  }
  
  internal func _unsafeRelease(id: UInt64) throws {
    try self.release(id: id)
  }
  
  @inline(__always)
  internal func _compilerRelease(_ allocation: Allocation) {
    let id = allocation.id
    let referenceCount = allocation.referenceCount - 1
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
      throw AllocationError("No memory has ever been allocated with ID #\(id).")
    }
    // Dictionary subscript returns an optional.
    return allocations[id]
  }
  
  func retain(id: UInt64) throws {
    guard id < nextAllocationID else {
      throw AllocationError("No memory has ever been allocated with ID #\(id).")
    }
    guard let allocation = allocations[id] else {
      // Catch memory management bugs.
      throw AllocationError("Allocation #\(id) has already been deallocated.")
    }
    allocation.referenceCount += 1
    if Allocation.debugInfoEnabled {
      print("Allocation #\(id) jumped to a reference count of \(allocation.referenceCount).")
    }
  }
  
  func release(id: UInt64) throws {
    // Catch memory management bugs.
    guard let allocation = allocations[id] else {
      throw AllocationError("Cannot deallocate something twice.")
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
// something invalid.
struct AllocationError: Error {
  var message: String
  init(_ message: String) {
    self.message = message
  }
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
  var materialized = false
  
  // Check this before performing any ops on the allocation. Otherwise, you're accessing undefined
  // memory.
  var initialized = false
  
  var mtlBuffer: MTLBuffer?
  // TODO: Shape - mutable for zero-cost reshape op
  // TODO: Data Type - mutable but only to match style of other properties
  var mpsMatrix: MPSMatrix?
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // TODO: Store the latest command buffer ID that references this. To make a conditional barrier
  // that waits until this specific command buffer completes, always store references to any command
  // buffers that are currently executing.
  var referencedCommandBufferID: Int?
  
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
    if materialized {
      // Already materialized.
    } else {
      try actuallyMaterialize()
    }
  }
  
  @inline(never)
  private func actuallyMaterialize() throws {
    defer {
      materialized = true
    }
    
    let device = Context.global.device
    let allocatedSize = HeapAllocator.global.totalAllocatedMemory
    if Context.global.permitExceedingSystemRAM {
      if allocatedSize + size <= device.maxBufferLength {
        if HeapAllocator.debugInfoEnabled {
          print("Memory allocation returned to something smaller than system RAM.")
        }
        Context.global.permitExceedingSystemRAM = false
      }
    } else {
      #if os(macOS)
      let maxWorkingSize = Int(device.recommendedMaxWorkingSetSize)
      #else
      let maxWorkingSize = device.maxBufferLength
      #endif
      if allocatedSize + size > maxWorkingSize {
        if HeapAllocator.debugInfoEnabled {
          print("""
            Memory allocation reached limit of system RAM. Clearing GPU command stream to free \
            memory.
            """)
        }
        // In the caller, set `permitExceedingSystemRAM` to true.
        throw AllocationError("Memory allocation reached the limit of system RAM.")
      }
    }
    
    guard let mtlBuffer = HeapAllocator.global.malloc(size: size, usingShared: true) else {
      throw AllocationError("An attempt to allocate a `MTLBuffer` returned `nil`.")
    }
    self.mtlBuffer = mtlBuffer
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  //
  // Does not automatically materialize because doing so should be made explicit.
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) throws {
    guard let mtlBuffer = mtlBuffer else {
      throw AllocationError("Initialized memory with a null underlying `MTLBuffer`.")
    }
    // Catch memory management bugs.
    if initialized {
      throw AllocationError("Cannot initialize something twice.")
    }
    defer {
      initialized = true
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
  func read(_ body: (UnsafeRawBufferPointer) -> Void) throws {
    // Cannot materialize here because it might not be initialized. Only safe place to materialize
    // is in the compiler, where it's at least derived from something that was initialized. The
    // compiler will then mark it as initialized and safe to read from.
    var bufferToRead: MTLBuffer!
    if !isShared {
      // TODO: Allocate a shared buffer, using a special heap reserved for shared memory.
      // TODO: Append a command that will copy the memory. This command is special, in that is takes
      // a `MTLBuffer` as output but can have an unmaterialized allocation as input.
      fatalError("Haven't implemented reading memory from a discrete GPU.")
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
    // If this was the outcome of a chain of operations, it should have been declared initialized
    // during compilation.
    guard initialized else {
      throw AllocationError("Cannot read from an uninitialized allocation.")
    }
    
    if isShared {
      bufferToRead = self.mtlBuffer!
    }
    let contents = bufferToRead.contents()
    let ptr = UnsafeRawBufferPointer(start: contents, count: size)
    body(ptr)
  }
  
  // Retain a reference to this until the command buffer is finished. Hold the reference in the
  // completion handler.
  deinit {
    // Catch memory management bugs.
    precondition(referenceCount == 0)
    guard materialized else {
      return
    }
    HeapAllocator.global.free(self.mtlBuffer!)
  }
}
