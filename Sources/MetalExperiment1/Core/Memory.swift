//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Atomics
import MetalPerformanceShadersGraph

extension MTLPluggableDevice {
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
      let allocation = reference.retain().takeUnretainedValue()
      allocation.initialize(body)
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
      let allocation = reference.retain().takeUnretainedValue()
      allocation.read(modifying: mutatingContents, body)
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

extension MTLPluggableDevice {
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
      id: id, referenceCount: referenceCount, device: self, dataType: dataType,
      byteCount: byteCount, shape: shape, isShared: self.prefersSharedMemory)
    
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
      id: id, referenceCount: referenceCount, device: self, dataType: dataType,
      byteCount: byteCount, shape: shape, isShared: isShared ?? self.prefersSharedMemory)
    
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
      isShared: isShared ?? self.prefersSharedMemory)
    
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
        print("Allocation #\(id) was released after being initialized.")
      } else {
        print("Allocation #\(id) was released.")
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
  
  // A copy of `MTLPluggableDevice.preferSharedMemory`, unless this is a transient allocation for
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
  var constantData: UnsafeMutableRawPointer?
  
  // TODO: Investigate a zero-cost reshape by transferring all resources over to another allocation.
  var mtlBuffer: MTLBuffer?
  var mpsMatrix: MPSMatrix?
  var mpsGraphTensorData: MPSGraphTensorData?
  
  // The last command buffer that read this allocation's underlying memory.
  var lastReadCommandBufferID: Int = -1
  
  // The last command buffer that mutated this allocation's underlying memory.
  var lastModifiedCommandBufferID: Int = -1
  
  // The command encoder within the command buffer above that initialized this.
  var lastModifiedCommandEncoderID: Int = -1
  
  init(
    id: UInt64,
    referenceCount: Int,
    device: MTLPluggableDevice,
    dataType: DataType,
    byteCount: Int,
    shape: UnsafeBufferPointer<Int>,
    isShared: Bool
  ) {
    self.id = id
    self.handle = AllocationHandle(
      referenceCount: referenceCount, device: device, dataType: dataType, byteCount: byteCount,
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
      id: id, referenceCount: referenceCount, device: handle.pluggableDevice,
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
    let device = handle.pluggableDevice
    if checkingMemoryBounds {
      let mtlDevice = device.mtlDevice
      let allocatedSize = device.heapAllocator.totalAllocatedMemory
      if device.permitExceedingSystemRAM {
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
          device.permitExceedingSystemRAM = false
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
    
    let heapAllocator = device.heapAllocator
    let mtlBuffer = heapAllocator.malloc(size: handle.byteCount, usingShared: isShared)
    guard let mtlBuffer = mtlBuffer else {
      fatalError("An attempt to allocate a 'MTLBuffer' returned 'nil'.")
    }
    self.mtlBuffer = mtlBuffer
    self.materialized = true
    
    // Transfer constant data into GPU-accessible memory. This will not recursively call
    // `actuallyMaterialize` because `materialized` is set.
    if let constantData = constantData {
      self.constantData = nil
      let ptr = UnsafeRawBufferPointer(start: constantData, count: handle.byteCount)
      initializeTensorData {
        $0.copyMemory(from: ptr)
      }
    }
  }
  
  func initialize(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    if handle.byteCount <= 4096 {
      initializeConstantData(body)
    } else {
      initializeTensorData(body)
    }
  }
  
  // Fills the memory with a user-specified closure. Do not go out of bounds, or else behavior is
  // undefined. On a discrete GPU, this calls `malloc` on CPU memory and enqueues a command to copy
  // it to device memory.
  func initializeTensorData(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    // Catch memory management bugs. If constant data exists, detach it from the allocation. Capture
    // it in `body`, then deallocate afterwards.
    precondition(!initialized && constantData == nil, "Cannot initialize something twice.")
    
    let device = handle.pluggableDevice
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
      do {
        try materialize()
      } catch AllocationError.exceededSystemRAM {
        device.permitExceedingSystemRAM = true
        device._internalBarrier()
      } catch {
        fatalError(error.localizedDescription)
      }
    } else {
      let sourceHandle = device._internalAllocate(1, handle, true)
      sourceAllocation = sourceHandle.reference!.takeUnretainedValue()
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      // Appending the explicit copy operation before `sourceAllocation` is actually initialized.
      // This is fine because the command stream won't be flushed any time soon.
      device._internalRetain(handle)
      let explicitCopy = EagerOperation.ExplicitCopy(input: sourceHandle, output: handle)
      device.eagerOperations.append(.explicitCopy(explicitCopy))
    }
    
    let contents = sourceAllocation.mtlBuffer!.contents()
    let ptr = UnsafeMutableRawBufferPointer(start: contents, count: handle.byteCount)
    body(ptr)
    sourceAllocation.initialized = true
  }
  
  func initializeConstantData(_ body: (UnsafeMutableRawBufferPointer) -> Void) {
    guard constantData == nil else {
      preconditionFailure("Constant data already existed.")
    }
    let byteCount = handle.byteCount
    constantData = .allocate(byteCount: byteCount, alignment: 0)
    let ptr = UnsafeMutableRawBufferPointer(start: constantData, count: byteCount)
    body(ptr)
  }
  
  func read(modifying: Bool, _ body: (UnsafeMutableRawBufferPointer) -> Void) {
    if constantData != nil {
      readConstantData(modifying: modifying, body)
    } else {
      readTensorData(modifying: modifying, body)
    }
  }
  
  // Flushes the command stream. On a discrete GPU, it appends one command to copy data from the GPU
  // before flushing the command stream. You must copy the data inside the pointer, because it will
  // deallocate or become undefined after the closure finishes.
  func readTensorData(modifying: Bool, _ body: (UnsafeMutableRawBufferPointer) -> Void) {
    // Cannot materialize here because it might not be initialized. Compilation is the earliest
    // stage where it's safe to materialize, as it's at least derived from something that was
    // initialized. The compiler will then mark it as initialized and safe to read from.
    let device = handle.pluggableDevice
    var sourceAllocation: Allocation
    if isShared {
      sourceAllocation = self
    } else {
      let sourceHandle = device._internalAllocate(1, handle, true)
      if modifying {
        device._internalRetain(sourceHandle)
      }
      sourceAllocation = sourceHandle.reference!.takeUnretainedValue()
      try! sourceAllocation.actuallyMaterialize(checkingMemoryBounds: false)
      
      device._internalRetain(handle)
      let explicitCopy = EagerOperation.ExplicitCopy(input: handle, output: sourceHandle)
      device.eagerOperations.append(.explicitCopy(explicitCopy))
    }
    
    // TODO: If the last command referencing this hasn't yet been encoded, place a MTLEvent in the
    // next command buffer. That way, you synchronize without dividing into two separate command
    // buffers (more overhead). It would also reduce I/O bottlenecks if you have several calls to
    // `read` in a row. This `MTLEvent` should never cause glitches in the graph compiler, because
    // the buffer here is not deallocated and not a placeholder. Keeping the entire pending command
    // batch intact provides more opportunities for fusing non-adjacent nodes in the graph. This
    // could be implemented by creating an `event` property on the allocation and materializing it
    // inside the compiler. No `MTLEvent` should be created if this is the last operation in the
    // graph.
    //
    // Exit the barrier operation below either when (a) the command buffer finishes or (b) the
    // `MTLEvent` notification handler executes.
    //
    // TODO: Prioritize the copying op if on a discrete GPU. Prepend the copying op to the beginning
    // of `eagerOperations`, unless one of those operations references it. This violates sequential
    // order of execution, but produces the same end result.
    //
    // Here's how to do this. During compilation, it optionally returns the index of the last
    // operation that references a specific tail. Intercept the completed instructions and
    // optionally insert an explicit copy after `instructions[index]`. If the index is not the end
    // of the list, signal a global `MTLSharedEvent` around `instructions[index]`. In the command
    // buffer's completion handler, signal a semaphore. The `MTLSharedEvent` also signals the
    // semaphore, but the last one to signal balances it out with a `wait`.
    //
    // These optimizations take a non-negligible amount of time to implement and create tests for.
    // It's also time-intensive to create benchmarks for them (they could make CPU-side latency
    // *worse*). I will not implement the two TODO's above in the near future.
    device._internalFlushStream()
    
    var commandBufferID = sourceAllocation.lastModifiedCommandBufferID
    if modifying {
      // Some commands might access the contents before they were modified.
      commandBufferID = max(commandBufferID, self.lastReadCommandBufferID)
    }
    if commandBufferID != -1 {
      device._internalBarrier(commandBufferID: commandBufferID)
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
        // If on a discrete GPU, this would be marked modified when first initialized by
        // user-defined scalars; it is the output of an explicit copy. Otherwise, it's the result of
        // an operation on other tensors.
        precondition(
          self.lastModifiedCommandBufferID != -1, "Allocation should be marked modified.")
        
        // Let the encoder register this as being modified twice.
        self.lastModifiedCommandBufferID = -1
        
        device._internalRetain(handle)
        let sourceHandle = sourceAllocation.handle
        let explicitCopy = EagerOperation.ExplicitCopy(input: sourceHandle, output: handle)
        device.eagerOperations.append(.explicitCopy(explicitCopy))
      }
    }
  }
  
  func readConstantData(modifying: Bool, _ body: (UnsafeMutableRawBufferPointer) -> Void) {
    guard let constantData = constantData else {
      preconditionFailure("Constant data did not exist.")
    }
    
    // If modifying, flush the command stream. Future commands might reference the allocation's
    // current value. If the contents transferred to GPU memory while encoding, switch to
    // `readTensorData`.
    if modifying {
      let device = handle.pluggableDevice
      device._internalFlushStream()
      if self.constantData == nil {
        readTensorData(modifying: true, body)
        return
      } else {
        precondition(
          !materialized && !initialized,
          "Something went wrong while modifying a tensor backed by constant memory.")
      }
    }
    
    let ptr = UnsafeMutableRawBufferPointer(start: constantData, count: handle.byteCount)
    body(ptr)
  }
  
  // Retain a reference to this until the command buffer is finished. Hold the reference in the
  // completion handler.
  deinit {
    if let constantData = constantData {
      constantData.deallocate()
    }
    let device = handle.pluggableDevice
    
    // Catch memory management bugs.
    precondition(handle.reference == nil, "Handle reference was not erased.")
    precondition(handle.referenceCount.destroy() == 0, "Reference count was nonzero.")
    device.numDeinitializedAllocations += 1
    
    // Activate this code if you suspect there are memory leaks.
    #if false
    let tensorCount = device.nextAllocationID - device.numDeinitializedAllocations
    print("Allocation #\(id) deinitialized. Live allocation count: \(tensorCount)")
    #endif
    
    // The command buffer should leave the dictionary before its referenced memory deallocates. Only
    // perform this check in debug mode because it's costly.
    assert({
      if lastModifiedCommandBufferID != -1 {
        precondition(materialized)
        return device.commandBufferDictionary[lastModifiedCommandBufferID] == nil
      } else {
        return true
      }
    }())
    
    guard materialized else {
      return
    }
    device.heapAllocator.free(self.mtlBuffer!)
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
    device: MTLPluggableDevice,
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
    
    let pluggableDeviceAddress = Unmanaged.passUnretained(device).toOpaque()
    baseAddress[0] = referenceCount
    baseAddress[1] = Int(bitPattern: OpaquePointer?(nil))
    baseAddress[2] = Int(bitPattern: pluggableDeviceAddress)
    baseAddress[3] = Int(dataType.rawValue) << 32 + Int(dataType.tensorFlowDataType)
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
  
  @inline(__always)
  var pluggableDevice: MTLPluggableDevice {
    let pointer = UnsafeRawPointer(bitPattern: baseAddress[2]).unsafelyUnwrapped
    return Unmanaged<MTLPluggableDevice>.fromOpaque(pointer).takeUnretainedValue()
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
