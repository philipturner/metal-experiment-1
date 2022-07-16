//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import MetalPerformanceShadersGraph

extension Context {
  // Returns (ID, rank) to match the style of other function calls.
  //
  // Avoids a possible second virtual function call by transforming the generic parameter into
  // something statically typed. There is already massive overhead from calling into
  // `withDispatchQueue`, but it should still be minimized.
  public static func allocateTensor<T>(
    _ type: T.Type,
    _ shape: UnsafeBufferPointer<Int>
  ) -> (UInt64, Int) {
    let dataType = DataType(type)
    let byteCount = shape.reduce(MemoryLayout<T>.stride, *)
    let id = withDispatchQueue {
      Context.global._allocateTensor(dataType, shape, byteCount)
    }
    return (id, shape.count)
  }
  
  public static func initializeTensor(
    _ id: UInt64,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    withDispatchQueue {
      Context.global._initializeTensor(id, body)
    }
  }
  
  public static func readTensor(
    _ id: UInt64,
    _ body: (UnsafeRawBufferPointer) -> Void
  ) {
    withDispatchQueue {
      Context.global._copyTensor(id, body)
    }
  }
  
  public static func copyTensorShape(
    _ id: UInt64,
    _ shape: UnsafeMutableBufferPointer<Int>
  ) {
    withDispatchQueue {
      Context.global._copyTensorShape(id, shape)
    }
  }
  
  public static func deleteTensor(
    _ id: UInt64
  ) {
    withDispatchQueue {
      Context.global._deleteTensor(id)
    }
  }
}

private extension Context {
  @inline(__always)
  func _allocateTensor(
    _ dataType: DataType,
    _ shape: UnsafeBufferPointer<Int>,
    _ byteCount: Int
  ) -> UInt64 {
    let (id, _) = _internalAllocate(dataType, shape, byteCount)
    return id
  }
  
  @inline(__always)
  func _initializeTensor(
    _ id: UInt64,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    let allocation = _internalFetch(id)
    try! allocation.materialize()
    try! allocation.initialize(body)
  }
  
  @inline(__always)
  func _readTensor(
    _ id: UInt64,
    _ body: (UnsafeRawBufferPointer) -> Void
  ) {
    let allocation = _internalFetch(id)
    try! allocation.read(body)
  }
  
  @inline(__always)
  func _copyTensorShape(
    _ id: UInt64,
    _ shape: UnsafeMutableBufferPointer<Int>
  ) {
    let allocation = _internalFetch(id)
    let metadata = allocation.metadata
    metadata.copyShape(to: shape)
  }
  
  @inline(__always)
  func _deleteTensor(
    _ id: UInt64
  ) {
    let allocation = _internalFetch(id)
    _internalRelease(allocation)
  }
}

extension Context {
  public static func generateID(allocationSize: Int) -> UInt64 {
    withDispatchQueue {
      return Context.global.generateID(allocationSize: allocationSize)
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
  
  @inline(__always)
  func _internalAllocate(
    _ dataType: DataType,
    _ shape: UnsafeBufferPointer<Int>,
    _ byteCount: Int
  ) -> (UInt64, Allocation) {
    let id = nextAllocationID
    let allocation = Allocation(id: id, dataType: dataType, shape: shape, byteCount: byteCount)
    nextAllocationID = id + 1
    allocations[id] = allocation
    return (id, allocation)
  }
  
  @inline(__always)
  func _internalFetch(_ id: UInt64) -> Allocation {
    guard let allocation = allocations[id] else {
      _internalFetchSlowPath(id)
    }
    return allocation
  }
  
  @inline(never)
  func _internalFetchSlowPath(_ id: UInt64) -> Never {
    if id >= nextAllocationID {
      preconditionFailure("No memory has ever been allocated with ID #\(id).")
    } else {
      preconditionFailure("Allocation #\(id) has already been deallocated.")
    }
  }
  
  @inline(__always)
  func _internalRetain(_ allocation: Allocation) {
    allocation.referenceCount &+= 1
    if Allocation.debugInfoEnabled {
      _internalRetainSlowPath(allocation)
    }
  }
  
  @inline(never)
  private func _internalRetainSlowPath(_ allocation: Allocation) {
    let id = allocation.id
    let referenceCount = allocation.referenceCount
    print("Allocation #\(id) jumped to a reference count of \(referenceCount).")
  }
  
  @inline(__always)
  func _internalRelease(_ allocation: Allocation) {
    let referenceCount = allocation.referenceCount &- 1
    allocation.referenceCount = referenceCount
    if referenceCount == 0 {
      allocations[allocation.id] = nil
    }
    if Allocation.debugInfoEnabled {
      _internalReleaseSlowPath(allocation)
    }
  }
  
  @inline(never)
  private func _internalReleaseSlowPath(_ allocation: Allocation) {
    let id = allocation.id
    let referenceCount = allocation.referenceCount
    if referenceCount == 0 {
      if allocation.initialized {
        print("Allocation #\(id) was deallocated after being initialized.")
      } else {
        print("Allocation #\(id) was deallocated.")
      }
    } else {
      print("Allocation #\(id) dropped to a reference count of \(referenceCount).")
    }
  }
}

private extension Context {
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
  var id: UInt64
  var referenceCount: Int
  static var debugInfoEnabled = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_REFERENCE_COUNTING")
  
  struct Metadata {
    // Vector elements:
    // 0..<5 - shape
    // 5..<6 - data type, padded from `UInt16` to `Int`
    // 6..<7 - physical size in bytes
    // 7..<8 - rank
    private var storage: SIMD8<Int>
    @inline(__always)
    var dataType: DataType { unsafeBitCast(storage[5], to: DataType.self) }
    @inline(__always)
    var byteCount: Int { storage[6] }
    @inline(__always)
    var rank: Int { storage[7] }
    
    init(dataType: DataType, shape: UnsafeBufferPointer<Int>, byteCount: Int) {
      storage = .zero
      storage[5] = Int(truncatingIfNeeded: dataType.rawValue)
      storage[6] = byteCount
      storage[7] = shape.count
      
      let shapePtr = shape.baseAddress!
      for i in 0..<rank {
        storage[i] = shapePtr[i]
      }
    }
    
    @inline(__always)
    func shapeEquals(_ other: Metadata) -> Bool {
      storage.lowHalf == other.storage.lowHalf && storage[4] == other.storage[4]
    }
    @inline(__always)
    func copyShape(to shape: UnsafeMutableBufferPointer<Int>) {
      precondition(shape.count == rank)
      let shapePtr = shape.baseAddress!
      for i in 0..<rank {
        shapePtr[i] = storage[i]
      }
    }
  }
  var metadata: Metadata
  
  // A copy of `Context.global.preferSharedStorage`.
  var isShared: Bool
  
  // TODO: Special storage mode for scalars or small chunks of constant memory. If `size` is under
  // 4 KB and memory is never mutated, it can be initialized on the CPU and passed into
  // `MTLComputeCommandEncoder.setBytes`. When in graph mode, there are similar mechanisms like
  // `MPSGraph.constant`.
  //
  // Will call the closure in `initialize(_:)` over the CPU-backed memory instead of the GPU-backed
  // memory. You may have to copy that CPU-backed memory to the `MTLBuffer`, but it's a very small
  // overhead. Take the overhead of a CPU function call + 2 * (4096 / (main memory bandwidth)).
  
  // Extracting this to its own property improves performance. It probably skips a reference count
  // to the `MTLBuffer`. Also, it could be useful if there are multiple storage modes.
  var materialized = false
  
  // Check this before performing any ops on the allocation. Otherwise, you're accessing undefined
  // memory.
  var initialized = false
  
  // TODO: Investigate a zero-cost reshape by transferring all resources over to another allocation.
  var mtlBuffer: MTLBuffer?
  var mpsMatrix: MPSMatrix?
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // The last command buffer that mutated this allocation's underlying memory.
  var lastModifiedCommandBufferID: Int = -1
  
  init(id: UInt64, metadata: Metadata) {
    self.id = id
    self.referenceCount = 1
    self.metadata = metadata
    self.isShared = Context.global.preferSharedStorage
  }
  
  convenience init(
    id: UInt64,
    dataType: DataType,
    shape: UnsafeBufferPointer<Int>,
    byteCount: Int
  ) {
    let metadata = Metadata(dataType: dataType, shape: shape, byteCount: byteCount)
    self.init(id: id, metadata: metadata)
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
    let device = Context.global.device
    let allocatedSize = HeapAllocator.global.totalAllocatedMemory
    if Context.global.permitExceedingSystemRAM {
      // Give it some wiggle room to remain in `permitExceedingSystemRAM` mode. Maximum buffer
      // length should be >50% system memory size. If it's hovering above the system RAM size
      // because all that memory needs to exist, it won't suddenly deallocate upon flushing the
      // command stream. In that case, flushing the command stream constantly as is oscillates above
      // and below a certain threshold would seriously degrade performance. But if it jumped the
      // threshold because my backend queued up too many commands, most of the memory would quickly
      // deallocate.
      //
      // This optimization is not possible on iOS because I can't query the `maxWorkingSize`.
      if allocatedSize + metadata.byteCount <= device.maxBufferLength {
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
      if allocatedSize + metadata.byteCount > maxWorkingSize {
        if HeapAllocator.debugInfoEnabled {
          print("""
            Memory allocation reached limit of system RAM. Clearing GPU command stream to free \
            memory.
            """)
        }
        // In the caller, set `permitExceedingSystemRAM` to true.
        throw AllocationError.exceededSystemRAM
      }
    }
    
    let mtlBuffer = HeapAllocator.global.malloc(size: metadata.byteCount, usingShared: isShared)
    guard let mtlBuffer = mtlBuffer else {
      throw AllocationError.other("An attempt to allocate a `MTLBuffer` returned `nil`.")
    }
    self.mtlBuffer = mtlBuffer
    self.materialized = true
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  //
  // Does not automatically materialize because doing so should be made explicit.
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) throws {
    guard let mtlBuffer = mtlBuffer else {
      throw AllocationError.other("Initialized memory with a null underlying `MTLBuffer`.")
    }
    // Catch memory management bugs.
    if initialized {
      throw AllocationError.other("Cannot initialize something twice.")
    }
    if isShared {
      let contents = mtlBuffer.contents()
      let ptr = UnsafeMutableRawBufferPointer(start: contents, count: metadata.byteCount)
      body(ptr)
    } else {
      // TODO: Append a command that will copy the memory.
      fatalError("Haven't implemented copying memory to a discrete GPU.")
    }
    self.initialized = true
  }
  
  // Making a function like `read` that just modifies the underlying storage is technically possible
  // and even more performant on a discrete GPU. However, it would be unused in the frontend.
  
  // Flushes the command stream. On a discrete GPU, it appends one command to copy data from the GPU
  // before flushing the command stream. You must copy the data inside the pointer, because it will
  // deallocate or become undefined after the closure finishes.
  func read(_ body: (UnsafeRawBufferPointer) -> Void) throws {
    // Cannot materialize here because it might not be initialized. Only safe place to materialize
    // is in the compiler, where it's at least derived from something that was initialized. The
    // compiler will then mark it as initialized and safe to read from.
    var dataBuffer: MTLBuffer?
    if !isShared {
      // TODO: Allocate a shared buffer, using a special heap reserved for shared memory.
      // TODO: Append a command that will copy the memory. This command is special, in that is takes
      // a `MTLBuffer` as output but can have an unmaterialized allocation as input.
      fatalError("Haven't implemented reading memory from a discrete GPU.")
    }
    
    // TODO: If the last command referencing this hasn't yet been encoded, place a MTLEvent in the
    // next command buffer. That way, you synchronize without dividing into two separate command
    // buffers (more overhead). It would also reduce I/O bottlenecks if you have several calls to
    // `read` in a row. This `MTLEvent` should never cause glitches in the graph compilier, because
    // the buffer here is not deallocated and not a placeholder. Keeping the entire pending command
    // batch intact provides more opportunities for fusing non-adjacent nodes in the graph.
    //
    // TODO: Prioritize the copying op if on a discrete GPU. Prepend the copying op to the beginning
    // of `bufferedOperations`, unless one of those operations references it. This violates
    // sequential order of execution, but produces the same end result.
    Context.global._internalFlushStream()
    
    // Encode the commands beforehand because they might write to `lastModifiedCommandBufferID`.
    if lastModifiedCommandBufferID != -1 {
      Context.global._internalBarrier(commandBufferID: lastModifiedCommandBufferID)
    }
    
    // If this was the outcome of a chain of operations, it should have been declared initialized
    // during compilation.
    guard initialized else {
      throw AllocationError.other("Cannot read from an uninitialized allocation.")
    }
    
    if isShared {
      dataBuffer = mtlBuffer!
    }
    let contents = dataBuffer!.contents()
    let ptr = UnsafeRawBufferPointer(start: contents, count: metadata.byteCount)
    body(ptr)
  }
  
  // Retain a reference to this until the command buffer is finished. Hold the reference in the
  // completion handler.
  deinit {
    // Catch memory management bugs.
    precondition(referenceCount == 0)
    
    // The command buffer must be released from the context before its referenced memory can
    // deallocate. Avoiding this check in release mode because it's very costly.
    assert({
      if let commandBufferID = lastModifiedCommandBufferID {
        precondition(materialized)
        return Context.global.commandBufferDictionary[commandBufferID] == nil
      } else {
        return true
      }
    }())
    
    guard materialized else {
      return
    }
    HeapAllocator.global.free(self.mtlBuffer!)
  }
}
