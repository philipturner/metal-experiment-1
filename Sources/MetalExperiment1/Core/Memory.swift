//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Atomics
import MetalPerformanceShadersGraph

extension Context {
  // Only for use in test suite.
  func allocateTensor(
    _ type: TF_DataType,
    _ shape: UnsafeBufferPointer<Int>
  ) -> OpaquePointer {
    let dataType = DataType(tensorFlowDataType: type)
    let byteCount = shape.reduce(dataType.stride, *)
    let handle = self.sync {
      self._internalAllocate(1, dataType, byteCount, shape)
    }
    return handle._cHandle
  }
  
  // Only for use in test suite.
  //
  // Only call this once. Initializing a tensor multiple times results in undefined behavior,
  // possibly a runtime crash.
  func initializeTensor(
    _ handle: OpaquePointer,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    self.sync {
      let reference = AllocationHandle(handle).reference!
      reference.retain().takeUnretainedValue().initialize(body)
      reference.release()
    }
  }
  
  // Allocates and initializes the tensor in a single function call, halving the overhead of calling
  // into the backend. Use this instead of calling `allocateTensor` and `initializeTensor`
  // separately whenever possible.
  @inline(never)
  public func createTensor(
    _ type: TF_DataType,
    _ shape: UnsafeBufferPointer<Int>,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) -> OpaquePointer {
    let dataType = DataType(tensorFlowDataType: type)
    let byteCount = shape.reduce(dataType.stride, *)
    let handle = self.sync {
      self._internalCreateTensor(1, dataType, byteCount, shape, body)
    }
    return handle._cHandle
  }
  
  @inline(never)
  public func readTensor(
    _ handle: OpaquePointer,
    _ mutatingContents: Bool,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) {
    self.sync {
      let reference = AllocationHandle(handle).reference!
      reference.retain().takeUnretainedValue().read(modifying: mutatingContents, body)
      reference.release()
    }
  }
  
  @inline(never)
  public func deleteTensor(
    _ handle: OpaquePointer
  ) {
    self.sync {
      let allocationHandle = AllocationHandle(handle)
      precondition(
        allocationHandle.referenceCount.load(ordering: .relaxed) == 0,
        "Deallocated a buffer with a reference count not equal to zero.")
      
      let reference = allocationHandle.reference!
      allocationHandle.reference = nil
      reference.release()
    }
  }
}

extension Context {
  @inline(__always)
  func _internalCreateTensor(
    _ referenceCount: Int,
    _ dataType: DataType,
    _ byteCount: Int,
    _ shape: UnsafeBufferPointer<Int>,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) -> AllocationHandle {
    let id = nextAllocationID
    nextAllocationID = id + 1
    let allocation = Allocation(
      id: id, referenceCount: referenceCount, context: self, dataType: dataType,
      byteCount: byteCount, shape: shape, isShared: self._prefersSharedMemory)
    
    let handle = allocation.handle
    handle.reference = .passRetained(allocation)
    allocation.initialize(body)
    return handle
  }
  
  @inline(__always)
  func _internalAllocate(
    _ referenceCount: Int,
    _ dataType: DataType,
    _ byteCount: Int,
    _ shape: UnsafeBufferPointer<Int>,
    _ isShared: Bool? = nil
  ) -> AllocationHandle {
    let id = nextAllocationID
    nextAllocationID = id + 1
    let allocation = Allocation(
      id: id, referenceCount: referenceCount, context: self, dataType: dataType,
      byteCount: byteCount, shape: shape, isShared: isShared ?? self._prefersSharedMemory)
    
    let handle = allocation.handle
    handle.reference = .passRetained(allocation)
    return handle
  }
  
  @inline(__always)
  func _internalAllocate(
    _ referenceCount: Int,
    _ other: AllocationHandle,
    _ isShared: Bool? = nil
  ) -> AllocationHandle {
    let id = nextAllocationID
    nextAllocationID = id + 1
    let allocation = Allocation(
      id: id, referenceCount: referenceCount, replicating: other,
      isShared: isShared ?? self._prefersSharedMemory)
    
    let handle = allocation.handle
    handle.reference = .passRetained(allocation)
    return handle
  }
  
  @discardableResult
  @inline(__always)
  func _internalRetain(_ handle: AllocationHandle) -> Int {
    let referenceCount = handle.referenceCount.wrappingIncrementThenLoad(ordering: .relaxed)
    if Allocation.debugInfoEnabled {
      _internalRetainSlowPath(handle, referenceCount)
    }
    return referenceCount
  }
  
  @inline(never)
  func _internalRetainSlowPath(_ handle: AllocationHandle, _ referenceCount: Int) {
    let id = handle.reference!.takeUnretainedValue().id
    print("Allocation #\(id) jumped to a reference count of \(referenceCount)")
  }
  
  @discardableResult
  @inline(__always)
  func _internalRelease(_ handle: AllocationHandle) -> Int {
    let referenceCount = handle.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
    if Allocation.debugInfoEnabled {
      _internalReleaseSlowPath(handle, referenceCount)
    }
    if referenceCount == 0 {
      let reference = handle.reference!
      handle.reference = nil
      reference.release()
    }
    return referenceCount
  }
  
  @inline(never)
  func _internalReleaseSlowPath(_ handle: AllocationHandle, _ referenceCount: Int) {
    let allocation = handle.reference!.takeUnretainedValue()
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
  static var debugInfoEnabled = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_REFERENCE_COUNTING")
  
  let id: UInt64
  var handle: AllocationHandle
  
  // A copy of `Context.global.preferSharedMemory`, unless this is a transient allocation for
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
  
  // The command encoder within the command buffer above that initialized this.
  var lastModifiedCommandEncoderID: Int = -1
  
  init(
    id: UInt64,
    referenceCount: Int,
    context: Context,
    dataType: DataType,
    byteCount: Int,
    shape: UnsafeBufferPointer<Int>,
    isShared: Bool
  ) {
    self.id = id
    self.handle = AllocationHandle(
      referenceCount: referenceCount, context: context, dataType: dataType, byteCount: byteCount,
      shape: shape)
    self.isShared = isShared
  }
  
  @inline(__always)
  convenience init(
    id: UInt64,
    referenceCount: Int,
    replicating handle: AllocationHandle,
    isShared: Bool
  ) {
    self.init(
      id: id, referenceCount: referenceCount, context: handle.pluggableDevice,
      dataType: handle.dataType, byteCount: handle.byteCount, shape: handle.shape,
      isShared: isShared)
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
    let context = handle.pluggableDevice
    if checkingMemoryBounds {
      let mtlDevice = context.mtlDevice
      let allocatedSize = context.heapAllocator.totalAllocatedMemory
      if context.permitExceedingSystemRAM {
        // Give it some wiggle room to remain in `permitExceedingSystemRAM` mode. Maximum buffer
        // length should be >50% system memory size. If it's hovering above the system RAM size
        // because all that memory needs to exist, it won't suddenly deallocate upon flushing the
        // command stream. In that case, flushing the command stream constantly as it oscillates
        // above and below a certain threshold would seriously degrade performance. But if it jumped
        // the threshold because my backend queued up too many commands, most of the memory would
        // quickly deallocate.
        //
        // On iOS, the threshold is different because `MTLDevice.recommendedMaxWorkingSetSize` does
        // not exist.
        #if os(macOS)
        let threshold = mtlDevice.maxBufferLength
        #else
        let threshold = (mtlDevice.maxBufferLength * 7) / 8
        #endif
        if allocatedSize + handle.byteCount <= threshold {
          if HeapAllocator.debugInfoEnabled {
            print("Memory allocation returned to something smaller than system RAM.")
          }
          context.permitExceedingSystemRAM = false
        }
      } else {
        #if os(macOS)
        let maxWorkingSize = Int(mtlDevice.recommendedMaxWorkingSetSize)
        #else
        let maxWorkingSize = mtlDevice.maxBufferLength
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
    
    let heapAllocator = context.heapAllocator
    let mtlBuffer = heapAllocator.malloc(size: handle.byteCount, usingShared: isShared)
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
    
    let context = handle.pluggableDevice
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
      do {
        try materialize()
      } catch AllocationError.exceededSystemRAM {
        context.permitExceedingSystemRAM = true
        context._internalBarrier()
      } catch {
        fatalError(error.localizedDescription)
      }
    } else {
      let sourceHandle = context._internalAllocate(1, handle, true)
      sourceAllocation = sourceHandle.reference!.takeUnretainedValue()
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      // Appending the explicit copy operation before `sourceAllocation` is actually initialized.
      // This is fine because the command stream won't be flushed any time soon.
      context._internalRetain(handle)
      let explicitCopy = EagerOperation.ExplicitCopy(input: sourceHandle, output: handle)
      context.eagerOperations.append(.explicitCopy(explicitCopy))
    }
    
    let contents = sourceAllocation.mtlBuffer!.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: handle.byteCount)
    body(ptr)
    sourceAllocation.initialized = true
  }
  
  // Flushes the command stream. On a discrete GPU, it appends one command to copy data from the GPU
  // before flushing the command stream. You must copy the data inside the pointer, because it will
  // deallocate or become undefined after the closure finishes.
  func read(modifying: Bool, _ body: (UnsafeMutableRawBufferPointer) -> Void) {
    // Cannot materialize here because it might not be initialized. Only safe place to materialize
    // is in the compiler, where it's at least derived from something that was initialized. The
    // compiler will then mark it as initialized and safe to read from.
    let context = handle.pluggableDevice
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
    } else {
      let sourceHandle = context._internalAllocate(1, handle, true)
      if modifying {
        context._internalRetain(sourceHandle)
      }
      sourceAllocation = sourceHandle.reference!.takeUnretainedValue()
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      context._internalRetain(handle)
      let explicitCopy = EagerOperation.ExplicitCopy(input: handle, output: sourceHandle)
      context.eagerOperations.append(.explicitCopy(explicitCopy))
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
    context._internalFlushStream()
    
    // Encode the commands beforehand because they might write to `lastModifiedCommandBufferID`.
    let commandBufferID = sourceAllocation.lastModifiedCommandBufferID
    if commandBufferID != -1 {
      context._internalBarrier(commandBufferID: commandBufferID)
    }
    
    // If this was the outcome of a chain of operations, it should have been declared initialized
    // during compilation.
    precondition(initialized, "Cannot read from an uninitialized allocation.")
    
    let contents = sourceAllocation.mtlBuffer!.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: handle.byteCount)
    body(ptr)
    
    if modifying {
      if isShared {
        // No need to copy data back to the accelerator.
      } else {
        context._internalRetain(handle)
        let sourceHandle = sourceAllocation.handle
        let explicitCopy = EagerOperation.ExplicitCopy(input: sourceHandle, output: handle)
        context.eagerOperations.append(.explicitCopy(explicitCopy))
      }
    }
  }
  
  // Retain a reference to this until the command buffer is finished. Hold the reference in the
  // completion handler.
  deinit {
    let context = handle.pluggableDevice
    
    // Catch memory management bugs.
    precondition(handle.reference == nil, "Handle reference was not erased.")
    precondition(handle.referenceCount.destroy() == 0, "Reference count was nonzero.")
    context.numDeinitializedAllocations += 1
    
    // Activate this code if you suspect there are memory leaks.
    #if false
    let tensorCount = context.nextAllocationID - context.numDeinitializedAllocations
    print("Allocation #\(id) deinitialialized. Live allocation count: \(tensorCount)")
    #endif
    
    // The command buffer must be released from the context before its referenced memory can
    // deallocate. Only perform this check in debug mode because it's very costly.
    assert({
      if lastModifiedCommandBufferID != -1 {
        precondition(materialized)
        return context.commandBufferDictionary[lastModifiedCommandBufferID] == nil
      } else {
        return true
      }
    }())
    
    guard materialized else {
      return
    }
    context.heapAllocator.free(self.mtlBuffer!)
  }
}

struct AllocationHandle: Hashable {
  private var baseAddress: UnsafeMutablePointer<Int>
  
  @inline(__always)
  init(_ cHandle: OpaquePointer) {
    baseAddress = UnsafeMutablePointer(cHandle)
  }
  
  init(
    referenceCount: Int,
    context: Context,
    dataType: DataType,
    byteCount: Int,
    shape: UnsafeBufferPointer<Int>
  ) {
    var bufferSize = 0
    bufferSize += 1 // referenceCount
    bufferSize += 1 // reference
    bufferSize += 1 // pluggable device memory address
    bufferSize += 1 // dataType
    bufferSize += 1 // byteCount
    bufferSize += 1 // rank
    bufferSize += shape.count // shape
    bufferSize *= MemoryLayout<Int>.stride
    baseAddress = malloc(bufferSize)!.assumingMemoryBound(to: Int.self)
    
    let pluggableDeviceAddress = Unmanaged.passUnretained(context).toOpaque()
    baseAddress[0] = referenceCount
    baseAddress[1] = Int(bitPattern: OpaquePointer?(nil))
    baseAddress[2] = Int(bitPattern: pluggableDeviceAddress)
    baseAddress[3] = Int(dataType.rawValue) << 32
    baseAddress[4] = byteCount
    baseAddress[5] = shape.count
    
    let shapeBuffer = UnsafeMutableBufferPointer(start: baseAddress + 6, count: shape.count)
    _ = shapeBuffer.initialize(from: shape)
  }
  
  @inline(__always)
  var _cHandle: OpaquePointer {
    OpaquePointer(baseAddress)
  }
  
  @inline(__always)
  var referenceCount: UnsafeAtomic<Int> {
    UnsafeAtomic(at: UnsafeMutablePointer(_cHandle))
  }
  
  var reference: Unmanaged<Allocation>? {
    @inline(__always)
    get {
      if let pointer = UnsafeRawPointer(bitPattern: baseAddress[1]) {
        return Unmanaged.fromOpaque(pointer)
      } else {
        return nil
      }
    }
    @inline(__always)
    nonmutating set {
      if let newValue = newValue {
        baseAddress[1] = Int(bitPattern: newValue.toOpaque())
      } else {
        baseAddress[1] = 0
      }
    }
  }
  
  // Use context object's memory address as a unique identfier. This is unique across all
  // pluggable device implementations.
  @inline(__always)
  var pluggableDevice: Context {
    let pointer = UnsafeRawPointer(bitPattern: baseAddress[2]).unsafelyUnwrapped
    return Unmanaged<Context>.fromOpaque(pointer).takeUnretainedValue()
  }
  
  
  @inline(__always)
  var tensorFlowDataType: TF_DataType {
    Int32(truncatingIfNeeded: baseAddress[3])
  }
  
  @inline(__always)
  var dataType: DataType {
    let castedAddress = UnsafeMutableRawPointer(baseAddress)
      .unsafelyUnwrapped.assumingMemoryBound(to: UInt32.self)
    let rawValue = UInt16(truncatingIfNeeded: castedAddress[7])
    return DataType(rawValue: rawValue).unsafelyUnwrapped
  }
  
  @inline(__always)
  var byteCount: Int {
    baseAddress[4]
  }
  
  @inline(__always)
  var rank: Int {
    baseAddress[5]
  }
  
  @inline(__always)
  var shape: UnsafeBufferPointer<Int> {
    UnsafeBufferPointer(start: baseAddress + 6, count: rank)
  }
}
