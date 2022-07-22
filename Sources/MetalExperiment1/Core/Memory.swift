//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Atomics
import MetalPerformanceShadersGraph

extension Context {
  // Returns (handle, rank) to match the style of other function calls.
  //
  // Avoids a possible second virtual function call by transforming the generic parameter into
  // something statically typed. There is already massive overhead from calling into
  // `withDispatchQueue`, but it should still be minimized.
  public static func allocateBuffer(
    _ type: Any.Type,
    _ shape: UnsafeBufferPointer<Int>
  ) -> (OpaquePointer, Int) {
    let dataType = DataType(type)
    let byteCount = shape.reduce(dataType.stride, *)
    let handle = withDispatchQueue {
      Context.global._allocateBuffer(1, dataType, byteCount, shape)
    }
    return (handle, shape.count)
  }
  
  public static func initializeBuffer(
    _ handle: OpaquePointer,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    withDispatchQueue {
      Context.global._initializeBuffer(handle, body)
    }
  }
  
  public static func readBuffer(
    _ handle: OpaquePointer,
    _ body: (UnsafeRawBufferPointer) -> Void
  ) {
    withDispatchQueue {
      Context.global._readBuffer(handle, body)
    }
  }
  
  public static func copyBufferShape(
    _ handle: OpaquePointer,
    _ shape: UnsafeMutableBufferPointer<Int>
  ) {
    withDispatchQueue {
      Context.global._copyBufferShape(handle, shape)
    }
  }
  
  // TODO: Avoid calling into `withDispatchQueue` if possible.
  public static func releaseBuffer(
    _ handle: OpaquePointer
  ) {
    withDispatchQueue {
      Context.global._releaseBuffer(handle)
    }
  }
}

private extension Context {
  @inline(__always)
  func _allocateBuffer(
    _ referenceCount: Int,
    _ dataType: DataType,
    _ byteCount: Int,
    _ shape: UnsafeBufferPointer<Int>
  ) -> OpaquePointer {
    let allocation = _internalAllocate(referenceCount, dataType, byteCount, shape)
    return allocation.handle._cHandle
  }
  
  @inline(__always)
  func _initializeBuffer(
    _ handle: OpaquePointer,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    let allocation = _internalFetch(AllocationHandle(handle))
    allocation.initialize(body)
  }
  
  @inline(__always)
  func _readBuffer(
    _ handle: OpaquePointer,
    _ body: (UnsafeRawBufferPointer) -> Void
  ) {
    let allocation = _internalFetch(AllocationHandle(handle))
    allocation.read(body)
  }
  
  // TODO: Remove this; it will soon become obsolete.
  @inline(__always)
  func _copyBufferShape(
    _ handle: OpaquePointer,
    _ shape: UnsafeMutableBufferPointer<Int>
  ) {
    let allocation = _internalFetch(AllocationHandle(handle))
    _ = shape.initialize(from: allocation.handle.shape)
  }
  
  @inline(__always)
  func _releaseBuffer(
    _ handle: OpaquePointer
  ) {
    _internalRelease(AllocationHandle(handle))
  }
}

extension Context {
  // TODO: Return the allocation handle instead of the allocation. In the functions that create
  // temporary buffers for CPU-GPU copying, use the handle to fetch the allocation.
  @inline(__always)
  func _internalAllocate(
    _ referenceCount: Int,
    _ dataType: DataType,
    _ byteCount: Int,
    _ shape: UnsafeBufferPointer<Int>,
    _ isShared: Bool? = nil
  ) -> Allocation {
    let id = nextAllocationID
    nextAllocationID = id + 1
    let allocation = Allocation(
      id: id, referenceCount: referenceCount, dataType: dataType, byteCount: byteCount,
      shape: shape, isShared: isShared)
    allocations[allocation.handle] = allocation
    return allocation
  }
  
  @inline(__always)
  func _internalAllocate(
    _ referenceCount: Int,
    _ other: AllocationHandle,
    _ isShared: Bool? = nil
  ) -> Allocation {
    let id = nextAllocationID
    nextAllocationID = id + 1
    let allocation = Allocation(
      id: id, referenceCount: referenceCount, replicating: other, isShared: isShared)
    allocations[allocation.handle] = allocation
    return allocation
  }
  
  // TODO: Remove the dictionary and instead place unmanaged reference at tail end of handle. This
  // removes the need to cache the allocation during compilation. It also requires careful
  // management of ARC pointers.
  //
  // `_internalFetch` will change to just retain/release the tail end of the handle.
  @inline(__always)
  func _internalFetch(_ handle: AllocationHandle) -> Allocation {
    guard let allocation = allocations[handle] else {
      fatalError("Tried to retrieve an allocation that does not exist.")
    }
    return allocation
  }
  
  @discardableResult
  @inline(__always)
  func _internalRetain(_ handle: AllocationHandle) -> Int {
    // TODO: Use relaxed ordering.
    let referenceCount = handle.referenceCount.wrappingIncrementThenLoad(ordering: .sequentiallyConsistent)
    if Allocation.debugInfoEnabled {
      _internalRetainSlowPath(handle, referenceCount)
    }
    return referenceCount
  }
  
  @inline(never)
  private func _internalRetainSlowPath(_ handle: AllocationHandle, _ referenceCount: Int) {
    let id = _internalFetch(handle).id
    print("Allocation #\(id) jumped to a reference count of \(referenceCount)")
  }
  
  @discardableResult
  @inline(__always)
  func _internalRelease(_ handle: AllocationHandle) -> Int {
    // TODO: Use relaxed ordering.
    let referenceCount = handle.referenceCount.wrappingDecrementThenLoad(ordering: .sequentiallyConsistent)
    if Allocation.debugInfoEnabled {
      _internalReleaseSlowPath(handle, referenceCount)
    }
    if referenceCount == 0 {
      allocations[handle] = nil
    }
    return referenceCount
  }
  
  @inline(never)
  private func _internalReleaseSlowPath(_ handle: AllocationHandle, _ referenceCount: Int) {
    let allocation = _internalFetch(handle)
    let id = allocation.id
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

enum AllocationError: Error {
  case exceededSystemRAM
}

class Allocation {
  static var debugInfoEnabled = fetchEnvironmentBoolean(
    "TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_REFERENCE_COUNTING")
  
  let id: UInt64
  var handle: AllocationHandle
  
  // A copy of `Context.global.preferSharedStorage`, unless this is a transient allocation for
  // reading/writing to discrete GPU memory.
  let isShared: Bool
  
  // Extracting this to its own property improves performance. It probably skips a reference count
  // to the `MTLBuffer`. Also, it could be useful if there are multiple storage modes.
  var materialized = false
  
  // Check this before performing any ops on the allocation. Otherwise, you're accessing undefined
  // memory.
  var initialized = false
  
  // TODO: Special storage mode for scalars or small chunks of constant memory. If `size` is under
  // 4 KB and memory is never mutated, it can be initialized on the CPU and passed into
  // `MTLComputeCommandEncoder.setBytes`. When in graph mode, there are similar mechanisms like
  // `MPSGraph.constant`.
  //
  // Will call the closure in `initialize(_:)` over the CPU-backed memory instead of the GPU-backed
  // memory. You may have to copy that CPU-backed memory to the `MTLBuffer`, but it's a very small
  // overhead. Take the overhead of a CPU function call + 2 * (4096 / (main memory bandwidth)).
  
  // TODO: Investigate a zero-cost reshape by transferring all resources over to another allocation.
  var mtlBuffer: MTLBuffer?
  var mpsMatrix: MPSMatrix?
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // The last command buffer that mutated this allocation's underlying memory.
  var lastModifiedCommandBufferID: Int = -1
  
  init(
    id: UInt64,
    referenceCount: Int,
    dataType: DataType,
    byteCount: Int,
    shape: UnsafeBufferPointer<Int>,
    isShared: Bool? = nil
  ) {
    self.id = id
    self.handle = AllocationHandle(
      referenceCount: referenceCount, dataType: dataType, byteCount: byteCount, shape: shape)
    self.isShared = isShared ?? Context.global.preferSharedStorage
  }
  
  @inline(__always)
  convenience init(
    id: UInt64,
    referenceCount: Int,
    replicating handle: AllocationHandle,
    isShared: Bool? = nil
  ) {
    self.init(id: id, referenceCount: referenceCount, dataType: handle.dataType, byteCount: handle.byteCount, shape: handle.shape, isShared: isShared)
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
  private func actuallyMaterialize(checkingMemoryBounds: Bool = true) throws {
    if checkingMemoryBounds {
      let device = Context.global.device
      let allocatedSize = HeapAllocator.totalAllocatedMemory
      if Context.global.permitExceedingSystemRAM {
        // Give it some wiggle room to remain in `permitExceedingSystemRAM` mode. Maximum buffer
        // length should be >50% system memory size. If it's hovering above the system RAM size
        // because all that memory needs to exist, it won't suddenly deallocate upon flushing the
        // command stream. In that case, flushing the command stream constantly as is oscillates
        // above and below a certain threshold would seriously degrade performance. But if it jumped
        // the threshold because my backend queued up too many commands, most of the memory would
        // quickly deallocate.
        //
        // On iOS, the threshold is different because `MTLDevice.recommendedMaxWorkingSetSize` does
        // not exist.
        #if os(macOS)
        let threshold = device.maxBufferLength
        #else
        let threshold = (device.maxBufferLength * 7) / 8
        #endif
        if allocatedSize + handle.byteCount <= threshold {
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
        if allocatedSize + handle.byteCount > maxWorkingSize {
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
    }
    
    let mtlBuffer = HeapAllocator.malloc(size: handle.byteCount, usingShared: isShared)
    guard let mtlBuffer = mtlBuffer else {
      fatalError("An attempt to allocate a 'MTLBuffer' returned 'nil'.")
    }
    self.mtlBuffer = mtlBuffer
    self.materialized = true
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    // Catch memory management bugs.
    precondition(!initialized, "Cannot initialize something twice.")
    
    let ctx = Context.global
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
      do {
        try materialize()
      } catch AllocationError.exceededSystemRAM {
        ctx.permitExceedingSystemRAM = true
        ctx._internalBarrier()
      } catch {
        fatalError(error.localizedDescription)
      }
      
      let contents = mtlBuffer!.contents()
      let ptr = UnsafeMutableRawBufferPointer(start: contents, count: handle.byteCount)
      body(ptr)
    } else {
      sourceAllocation = ctx._internalAllocate(1, handle, true)
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      // Appending the explicit copy operation before `sourceAllocation` is actually initialized.
      // This is fine because the command stream won't be flushed any time soon.
      ctx._internalRetain(handle)
      let sourceHandle = sourceAllocation.handle
      let explicitCopy = EagerOperation.ExplicitCopy(input: sourceHandle, output: handle)
      Context.global.eagerOperations.append(.explicitCopy(explicitCopy))
    }
    
    let contents = sourceAllocation.mtlBuffer!.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: handle.byteCount)
    body(ptr)
    sourceAllocation.initialized = true
  }
  
  // Making a function like `read` that just modifies the underlying storage is technically
  // possible. It takes less time than a separate `read` + `initialize` on a discrete GPU, and is
  // relatively instantaneous on an integrated GPU. However, it would be unused in the frontend.
//  func modify(_ body: (UnsafeRawBufferPointer) -> Void) {}
  
  // Flushes the command stream. On a discrete GPU, it appends one command to copy data from the GPU
  // before flushing the command stream. You must copy the data inside the pointer, because it will
  // deallocate or become undefined after the closure finishes.
  func read(_ body: (UnsafeRawBufferPointer) -> Void) {
    // Cannot materialize here because it might not be initialized. Only safe place to materialize
    // is in the compiler, where it's at least derived from something that was initialized. The
    // compiler will then mark it as initialized and safe to read from.
    let ctx = Context.global
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
    } else {
      sourceAllocation = Context.global._internalAllocate(1, handle, true)
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      ctx._internalRetain(self.handle)
      let sourceHandle = sourceAllocation.handle
      let explicitCopy = EagerOperation.ExplicitCopy(input: handle, output: sourceHandle)
      Context.global.eagerOperations.append(.explicitCopy(explicitCopy))
    }
    
    // TODO: If the last command referencing this hasn't yet been encoded, place a MTLEvent in the
    // next command buffer. That way, you synchronize without dividing into two separate command
    // buffers (more overhead). It would also reduce I/O bottlenecks if you have several calls to
    // `read` in a row. This `MTLEvent` should never cause glitches in the graph compilier, because
    // the buffer here is not deallocated and not a placeholder. Keeping the entire pending command
    // batch intact provides more opportunities for fusing non-adjacent nodes in the graph. This
    // could be implemented by creating an `event` property on the allocation and materializing it
    // inside the compiler. No `MTLBuffer` should be created if this is the last operation in the
    // graph.
    //
    // TODO: Prioritize the copying op if on a discrete GPU. Prepend the copying op to the beginning
    // of `bufferedOperations`, unless one of those operations references it. This violates
    // sequential order of execution, but produces the same end result.
    ctx._internalFlushStream()
    
    // Encode the commands beforehand because they might write to `lastModifiedCommandBufferID`.
    let commandBufferID = sourceAllocation.lastModifiedCommandBufferID
    if commandBufferID != -1 {
      ctx._internalBarrier(commandBufferID: commandBufferID)
    }
    
    // If this was the outcome of a chain of operations, it should have been declared initialized
    // during compilation.
    precondition(initialized, "Cannot read from an uninitialized allocation.")
    
    let contents = sourceAllocation.mtlBuffer!.contents()
    let ptr = UnsafeRawBufferPointer(start: contents, count: handle.byteCount)
    body(ptr)
  }
  
  // Retain a reference to this until the command buffer is finished. Hold the reference in the
  // completion handler.
  deinit {
    // Catch memory management bugs.
    precondition(handle.referenceCount.destroy() == 0)
    
    // The command buffer must be released from the context before its referenced memory can
    // deallocate. Avoiding this check in release mode because it's very costly.
    assert({
      if lastModifiedCommandBufferID != -1 {
        precondition(materialized)
        return Context.global.commandBufferDictionary[lastModifiedCommandBufferID] == nil
      } else {
        return true
      }
    }())
    
    guard materialized else {
      return
    }
    HeapAllocator.free(self.mtlBuffer!)
  }
}

public struct AllocationHandle: Hashable {
  @usableFromInline
  internal var baseAddress: UnsafeMutablePointer<Int>
  
  @inlinable @inline(__always)
  public init(_ cHandle: OpaquePointer) {
    baseAddress = UnsafeMutablePointer(cHandle)
  }
  
  @inline(__always)
  internal init(
    referenceCount: Int,
    dataType: DataType,
    byteCount: Int,
    shape: UnsafeBufferPointer<Int>
  ) {
    var bufferSize = 0
    bufferSize += 1 // referenceCount
    bufferSize += 1 // dataType
    bufferSize += 1 // byteCount
    bufferSize += 1 // rank
    bufferSize += shape.count // shape
    bufferSize *= MemoryLayout<Int>.stride
    baseAddress = malloc(bufferSize)!.assumingMemoryBound(to: Int.self)
    
    baseAddress[0] = referenceCount
    baseAddress[1] = Int(dataType.rawValue)
    baseAddress[2] = byteCount
    baseAddress[3] = shape.count
    
    let shapeBuffer = UnsafeMutableBufferPointer(start: baseAddress + 4, count: shape.count)
    _ = shapeBuffer.initialize(from: shape)
  }
  
  @inlinable @inline(__always)
  public var _cHandle: OpaquePointer {
    OpaquePointer(baseAddress)
  }
  
  @inlinable @inline(__always)
  public var referenceCount: UnsafeAtomic<Int> {
    UnsafeAtomic(at: UnsafeMutablePointer(_cHandle))
  }
  
  @inline(__always)
  internal var dataType: DataType {
    let rawValue = UInt16(truncatingIfNeeded: baseAddress[1])
    return DataType(rawValue: rawValue).unsafelyUnwrapped
  }
  
  @inlinable @inline(__always)
  public var byteCount: Int {
    baseAddress[2]
  }
  
  @inlinable @inline(__always)
  public var rank: Int {
    baseAddress[3]
  }
  
  @inlinable @inline(__always)
  public var shape: UnsafeBufferPointer<Int> {
    UnsafeBufferPointer(start: baseAddress + 4, count: rank)
  }
}
