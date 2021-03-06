import XCTest
@testable import MetalExperiment1

fileprivate func allocate(capacity: Int) -> OpaquePointer {
  withUnsafeTemporaryAllocation(of: Int.self, capacity: 1) { shape in
    shape[0] = capacity
    return defaultPluggableDevice.allocateTensor(
      Float.tensorFlowDataType._cDataType, UnsafeBufferPointer(shape))
  }
}

fileprivate func releaseBuffer(_ handle: CTensorHandle) {
  let atomic = AllocationHandle(handle).referenceCount
  if atomic.wrappingDecrementThenLoad(ordering: .relaxed) == 0 {
    defaultPluggableDevice.deleteTensor(handle)
  }
}

final class MemoryTests: XCTestCase {
  func testSimpleAllocation() throws {
    testHeader("Simple memory allocation")
    defaultPluggableDevice.heapAllocator._releaseCachedBufferBlocks()
    
    do {
      let firstHandle = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
      let secondHandle = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
      releaseBuffer(firstHandle)
      releaseBuffer(secondHandle)
    }
    
    do {
      Profiler.checkpoint()
      let numHandles = 100
      for _ in 0..<numHandles {
        let handle = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
        releaseBuffer(handle)
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(numHandles)
      print("Unused handle creation throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    do {
      let handle = allocate(capacity: 4000 / MemoryLayout<Float>.stride)
      defer { releaseBuffer(handle) }
      
      defaultPluggableDevice.initializeTensor(handle) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        ptr.initialize(repeating: 2.5)
      }
      var wereEqual = false
      defaultPluggableDevice.readTensor(handle, false) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        let comparisonSequence = [Float](repeating: 2.5, count: 1000)
        wereEqual = ptr.elementsEqual(comparisonSequence)
      }
      XCTAssertTrue(wereEqual)
    }
    
    func trunc_page(_ x: vm_size_t) -> vm_size_t {
      ((x) & (~(vm_page_size - 1)))
    }
    func round_page(_ x: vm_size_t) -> vm_size_t {
      trunc_page((x) + (vm_page_size - 1))
    }
    
    XCTAssertEqual(round_page(1024 * 1024), 1024 * 1024)
    
    do {
      // Test that an `AllocatorBlockSet` is always ordered.
      class CustomBlockWrapped {}
      
      class CustomBlock: AllocatorBlockProtocol {
        var size: Int
        var wrapped: CustomBlockWrapped
        
        init(size: Int) {
          self.size = size
          self.wrapped = CustomBlockWrapped()
        }
      }
      
      var customSet = AllocatorBlockSet<CustomBlock>()
      customSet.insert(.init(size: 6))
      customSet.insert(.init(size: 4))
      customSet.insert(.init(size: 6))
      customSet.insert(.init(size: 6))
      customSet.insert(.init(size: 8))
      
      XCTAssertEqual(customSet.remove(at: 0).size, 4)
      XCTAssertEqual(customSet.remove(at: 0).size, 6)
      XCTAssertEqual(customSet.remove(at: 0).size, 6)
      XCTAssertEqual(customSet.remove(at: 0).size, 6)
      XCTAssertEqual(customSet.remove(at: 0).size, 8)
    }
  }
  
  func testRecyclingThroughput() throws {
    testHeader("Memory recycling throughput")
    defaultPluggableDevice.heapAllocator._releaseCachedBufferBlocks()
    
    func allocateDeallocate(bufferSize: Int, numBuffers: Int) {
      var handles: [OpaquePointer] = []
      for _ in 0..<numBuffers {
        let handle = allocate(capacity: bufferSize / MemoryLayout<Float>.stride)
        handles.append(handle)
      }
      for handle in handles {
        defaultPluggableDevice.initializeTensor(handle) { _ in }
      }
      for handle in handles {
        releaseBuffer(handle)
      }
    }
    func fakeAllocateDeallocate(numBuffers: Int) {
      var handles: [OpaquePointer?] = []
      for _ in 0..<numBuffers {
        let handle = defaultPluggableDevice.sync {
          return OpaquePointer?(nil)
        }
        handles.append(handle)
      }
      for handle in handles {
        defaultPluggableDevice.sync {
          _ = handle
        }
      }
      for handle in handles {
        defaultPluggableDevice.sync {
          _ = handle
        }
      }
    }
    func emptyAllocateDeallocate(bufferSize: Int, numBuffers: Int) {
      var handles: [OpaquePointer] = []
      for _ in 0..<numBuffers {
        let handle = allocate(capacity: bufferSize / MemoryLayout<Float>.stride)
        handles.append(handle)
      }
      for handle in handles {
        defaultPluggableDevice.sync {
          _ = handle
        }
      }
      for handle in handles {
        releaseBuffer(handle)
      }
    }
    
    allocateDeallocate(bufferSize: 4000, numBuffers: 5)
    Profiler.checkpoint()
    allocateDeallocate(bufferSize: 1000, numBuffers: 5)
    allocateDeallocate(bufferSize: 2000, numBuffers: 5)
    allocateDeallocate(bufferSize: 3000, numBuffers: 5)
    allocateDeallocate(bufferSize: 4092, numBuffers: 5)
    let totalTime = Profiler.checkpoint()
    
    Profiler.checkpoint()
    fakeAllocateDeallocate(numBuffers: 5)
    fakeAllocateDeallocate(numBuffers: 5)
    fakeAllocateDeallocate(numBuffers: 5)
    fakeAllocateDeallocate(numBuffers: 5)
    let gcdTime = Profiler.checkpoint()
    
    Profiler.checkpoint()
    emptyAllocateDeallocate(bufferSize: 1000, numBuffers: 5)
    emptyAllocateDeallocate(bufferSize: 2000, numBuffers: 5)
    emptyAllocateDeallocate(bufferSize: 3000, numBuffers: 5)
    emptyAllocateDeallocate(bufferSize: 4092, numBuffers: 5)
    let handleCycleTime = Profiler.checkpoint()
    
    let totalThroughput = Double(totalTime) / 20
    print("Memory recycling throughput: \(totalThroughput) \(Profiler.timeUnit)")
    let nonGCDThroughput = Double(totalTime - gcdTime) / 20
    print("Time excluding GCD: \(nonGCDThroughput) \(Profiler.timeUnit)")
    let allocationThroughput = Double(totalTime - handleCycleTime) / 20
    print("Time inside HeapAllocator: \(allocationThroughput) \(Profiler.timeUnit)")
  }
  
  func testComplexAllocation() throws {
    testHeader()
    let heapAllocator = defaultPluggableDevice.heapAllocator
    heapAllocator._releaseCachedBufferBlocks()
    
    func allocate(byteCount: Int) -> OpaquePointer {
      // The compiler mistakes this for `allocate(byteCount:)`.
      let _avoidNameCollision = allocate(capacity:)
      let handle = _avoidNameCollision(byteCount / MemoryLayout<Float>.stride)
      defaultPluggableDevice.initializeTensor(handle) { _ in }
      return handle
    }
    func deallocate(handle: OpaquePointer) {
      releaseBuffer(handle)
    }
    
    let handle1 = allocate(byteCount: 8_000_000)
    let handle2 = allocate(byteCount: 12_000_000)
    let handle3 = allocate(byteCount: 12_000_000)
    deallocate(handle: handle1)
    deallocate(handle: handle2)
    deallocate(handle: handle3)
    
    let handle4 = allocate(byteCount: 999_000)
    deallocate(handle: handle4)
    
    let handle5 = allocate(byteCount: 2_000_000)
    deallocate(handle: handle5)
    
    // Test mechanism for dealing with excessive memory allocation.
    
    do {
      // Choose buffer size 10_000 (larger than 4096), otherwise data resides in constant memory.
      heapAllocator._releaseCachedBufferBlocks()
      let smallBufferHandle1 = allocate(byteCount: 10_000)
      defer { deallocate(handle: smallBufferHandle1) }
      defaultPluggableDevice.sync {
        defaultPluggableDevice.permitExceedingSystemRAM = true
      }
      
      // This part of the test causes a massive bottleneck on discrete GPUs.
      if defaultPluggableDevice.prefersSharedMemory {
        let largeBufferSize = defaultPluggableDevice.mtlDevice.maxBufferLength
        let largeBufferHandle1 = allocate(byteCount: largeBufferSize)
        defer { deallocate(handle: largeBufferHandle1) }
        defaultPluggableDevice.sync {
          XCTAssertTrue(defaultPluggableDevice.permitExceedingSystemRAM)
        }
      }
      
      let smallBufferHandle2 = allocate(byteCount: 10_000)
      defer { deallocate(handle: smallBufferHandle2) }
      defaultPluggableDevice.sync {
        XCTAssertTrue(defaultPluggableDevice.permitExceedingSystemRAM)
      }
    }
    defaultPluggableDevice.sync {
      XCTAssertTrue(defaultPluggableDevice.permitExceedingSystemRAM)
    }
    
    do {
      let smallBufferHandle3 = allocate(byteCount: 10_000)
      defer { deallocate(handle: smallBufferHandle3) }
      defaultPluggableDevice.sync {
        XCTAssertTrue(defaultPluggableDevice.permitExceedingSystemRAM)
      }
      
      if defaultPluggableDevice.prefersSharedMemory {
        heapAllocator._releaseCachedBufferBlocks()
      } else {
        // Without making a barrier, the `XCTAssertFalse` below fails on discrete GPUs.
        defaultPluggableDevice.barrier()
      }
      
      let smallBufferHandle4 = allocate(byteCount: 10_000)
      defer { deallocate(handle: smallBufferHandle4) }
      defaultPluggableDevice.sync {
        XCTAssertFalse(defaultPluggableDevice.permitExceedingSystemRAM)
      }
    }
  }
  
  func testTensorHandleLifetime() throws {
    testHeader("Tensor handle lifetime")
    defaultPluggableDevice.sync {
      // Already overrode the environment variable for this in `testHeader`.
      Allocation.debugInfoEnabled = true
    }
    
    print("Start of function")
    do {
      _ = Tensor<Float>(repeating: 5, shape: [2])
    }
    
    do {
      print()
      print("Handle 1")
      let handle1 = Tensor<Float>(repeating: 5, shape: [2])
      XCTAssertEqual(handle1.scalars, [5.0, 5.0])
      
      print()
      print("Handle 2")
      let handle2 = handle1.incremented()
      XCTAssertEqual(handle2.scalars, [6.0, 6.0])
      
      print()
      print("Handle 3")
      let handle3 = handle1.incremented().incremented()
      XCTAssertEqual(handle3.scalars, [7.0, 7.0])
      
      print()
      print("Handle 4")
      let handle4 = handle2.incremented().incremented()
      XCTAssertEqual(handle4.scalars, [8.0, 8.0])
      
      print()
      print("Handle 5")
      let handle5 = handle4.incremented().incremented().incremented()
      XCTAssertEqual(handle5.scalars, [11.0, 11.0])
      
      print()
      print("End of function")
    }
    
    defaultPluggableDevice.sync {
      Allocation.debugInfoEnabled = false
    }
    
    // This is one of the reproducers for a memory management bug I encountered. Retaining it as a
    // regression test, but suppressing its print output.
    do {
      let handle1 = Tensor<Float>(repeating: 5, shape: [2])
      XCTAssertEqual(handle1.scalars, [5.0, 5.0])
      
      let handle2 = handle1.incremented()
      XCTAssertEqual(handle2.scalars, [6.0, 6.0])
      
      let handle3 = handle1.incremented().incremented()
      XCTAssertEqual(handle3.scalars, [7.0, 7.0])
      
      let handle4 = handle2.incremented().incremented()
      XCTAssertEqual(handle4.scalars, [8.0, 8.0])
      
      let handle5 = handle4.incremented().incremented().incremented()
      XCTAssertEqual(handle5.scalars, [11.0, 11.0])
    }
  }
  
  func testMassiveAllocation() throws {
    testHeader("Massive memory allocation")
    let heapAllocator = defaultPluggableDevice.heapAllocator
    heapAllocator._releaseCachedBufferBlocks()
    defer {
      heapAllocator._releaseCachedBufferBlocks()
    }
    
    let mtlDevice = defaultPluggableDevice.mtlDevice
#if os(macOS)
    var maxWorkingSize = Int(mtlDevice.recommendedMaxWorkingSetSize)
#else
    var maxWorkingSize = mtlDevice.maxBufferLength
#endif
    var maxBufferLength = mtlDevice.maxBufferLength
    
    // If most of the device's memory is allocated, this causes a command buffer abortion on
    // discrete GPUs.
    if !defaultPluggableDevice.prefersSharedMemory {
      maxWorkingSize = 2 * 1024 * 1024
      maxBufferLength = 1 * 1024 * 1024
    }
    
    let bufferSize1 = maxBufferLength - 16384
    let bufferSize2 = (maxWorkingSize - maxBufferLength) + 16384
    let bufferCount3 = 16384 / MemoryLayout<Float>.stride
    
    let bufferID1 = allocate(capacity: bufferSize1  / MemoryLayout<Float>.stride)
    let bufferID2 = allocate(capacity: bufferSize2 / MemoryLayout<Float>.stride)
    defer {
      releaseBuffer(bufferID1)
      releaseBuffer(bufferID2)
    }
    
    var tensor3: Tensor<Float>?
    defaultPluggableDevice.sync { defaultPluggableDevice.permitExceedingSystemRAM = true }
    defaultPluggableDevice.initializeTensor(bufferID1) { _ in }
    defaultPluggableDevice.sync { defaultPluggableDevice.permitExceedingSystemRAM = true }
    defaultPluggableDevice.initializeTensor(bufferID2) { _ in }
    defaultPluggableDevice.sync { defaultPluggableDevice.permitExceedingSystemRAM = true }
    tensor3 = Tensor<Float>(repeating: 0, shape: [bufferCount3])
    defaultPluggableDevice.sync { defaultPluggableDevice.permitExceedingSystemRAM = false }
    guard let tensor3 = tensor3 else {
      fatalError("This should never happen.")
    }
    
    withExtendedLifetime(tensor3) {
      let tensor4 = tensor3.incremented()

      let scalars3 = tensor3.scalars
      let scalars4 = tensor4.scalars
      XCTAssertEqual(scalars3[0], 0.0)
      XCTAssertEqual(scalars4[0], 1.0)
      print("Tensor 3: [\(scalars3[0]), ...]")
      print("Tensor 4: [\(scalars4[0]), ...]")
    }
  }
  
  func testReadPerformance() {
    testHeader("Buffer read performance")
    do {
      _ = Tensor<Float>(repeating: 4, shape: [2])
    }
    
    for i in 0..<2 {
      print()
      let loopOffset: Float = (i == 0) ? 0.0 : 0.1
      
      Profiler.checkpoint()
      let handle1 = Tensor<Float>(repeating: 5 + loopOffset, shape: [2])
      XCTAssertEqual(handle1.scalars, [5.0 + loopOffset, 5.0 + loopOffset])
      Profiler.log("Read handle 1 (fast)")
      
      let handle2 = handle1.incremented()
      XCTAssertEqual(handle2.scalars, [6.0 + loopOffset, 6.0 + loopOffset])
      Profiler.log("Read handle 2 (slow)")
      
      // This should be fast, but isn't.
      let handle3 = handle1.incremented()
      XCTAssertEqual(handle1.scalars, [5.0 + loopOffset, 5.0 + loopOffset])
      Profiler.log("Read handle 1 again (fast)")
      
      // This should be medium, but isn't.
      XCTAssertEqual(handle3.scalars, [6.0 + loopOffset, 6.0 + loopOffset])
      Profiler.log("Read handle 3 after execution (slow)")
      
      let handle4 = handle3.incremented()
      XCTAssertEqual(handle4.scalars, [7.0 + loopOffset, 7.0 + loopOffset])
      Profiler.log("Read handle 4 (slow)")
    }
  }
  
  func testInterruptedUnaryFusion() throws {
    testHeader("Interrupted unary fusion")
    
    Profiler.checkpoint()
    
    // Don't override the environment variable for other tests.
    var previousProfilingEncoding = false
    defaultPluggableDevice.sync {
      previousProfilingEncoding = MTLPluggableDevice.profilingEncoding
      MTLPluggableDevice.profilingEncoding = true
    }
    defer {
      defaultPluggableDevice.sync {
        MTLPluggableDevice.profilingEncoding = previousProfilingEncoding
      }
    }
    
    for _ in 0..<2 {
      let fusion1_part1 = Tensor<Float>(repeating: 101, shape: [2])
      let fusion1_part2 = fusion1_part1.incremented()
      let fusion1_part3 = fusion1_part2.incremented()
      let fusion1_part4 = fusion1_part3.incremented()
      
      let fusion2_part1 = Tensor<Float>(repeating: 201, shape: [2])
      let fusion2_part2 = fusion2_part1.incremented()
      let fusion2_part3 = fusion2_part2.incremented()
      let fusion2_part4 = fusion2_part3.incremented()
      
      let fusion1_part5 = fusion1_part4.incremented()
      let fusion1_part6 = fusion1_part5.incremented()
      let fusion1_part7 = fusion1_part6.incremented()
      XCTAssertEqual(fusion1_part7.scalars, [107, 107])
      XCTAssertEqual(fusion2_part4.scalars, [204, 204])
    }
    
    Profiler.log("Interrupted unary fusion")
  }
  
  func testDivergentUnaryFusion() throws {
    testHeader()
    
    for _ in 0..<2 {
      _ = Tensor<Float>(repeating: 4, shape: [2]).incremented()
      
      let fusion_part1 = Tensor<Float>(repeating: 11, shape: [2])
      let fusion_part2 = fusion_part1.incremented()
      let fusion_part3 = fusion_part2.incremented()
      
      let fusion_divergence = -fusion_part2
      XCTAssertEqual(fusion_part3.scalars, [13, 13])
      XCTAssertEqual(fusion_divergence.scalars, [-12, -12])
    }
  }
  
  func testZombieOperationRemoval() throws {
    testHeader()
    
    func basicTestNothing(checking: Bool) {
      for _ in 0..<2 {
        func doNothing<T: Numeric>(_ input: Tensor<T>) {
          let unused1 = input.incremented()
          let unused2 = unused1.incremented()
          _ = unused2.incremented()
        }
        
        let input = Tensor<Float>(repeating: 8, shape: [2])
        doNothing(input)
        if checking {
          XCTAssertEqual(input.scalars, [8, 8])
        }
      }
    }
    
    basicTestNothing(checking: false)
    
    for _ in 0..<2 {
      _ = Tensor<Float>(repeating: 4, shape: [2]).incremented()
      
      func doNothingDivergent<T: SignedNumeric>(_ input: Tensor<T>) -> Tensor<T> {
        let unused1 = input.incremented()
        let unused2 = unused1.incremented()
        let unused3 = unused2.incremented()
        _ = unused3.incremented()
        
        // `unused2` still materializes because the minus operation below holds its reference. The
        // operations that increment `unused2` and `unused3` are aborted.
        return -unused2
      }
      
      let input = Tensor<Float>(repeating: 8, shape: [2])
      let output = doNothingDivergent(input)
      XCTAssertEqual(input.scalars, [8, 8])
      XCTAssertEqual(output.scalars, [-10, -10])
    }
    
    basicTestNothing(checking: true)
  }
  
  func testModifyingRead() {
    testHeader()
    
    // Test a pluggable device with storage mode opposite to the system default.
    let mtlDevice = MTLCreateSystemDefaultDevice()!
    let descriptor = MTLPluggableDeviceDescriptor()
    descriptor.storageMode = mtlDevice.hasUnifiedMemory ? .private : .shared
    let altDevice = mtlDevice.makePluggableDevice(descriptor: descriptor)!
    
    func testModifyingRead(on device: MTLPluggableDevice) {
      withDevice(device) {
        let currentDevice = _ExecutionContext.global.currentDevice
        XCTAssertIdentical(device, currentDevice)
        
        let tensor1 = Tensor<Float>([8, 8])
        let tensor2 = Tensor<Float>([9, 9])
        let tensor3 = tensor1 + tensor2
        device.readTensor(tensor3._rawTensorHandle, true) { buffer in
          let reboundBuffer = buffer.assumingMemoryBound(to: Float.self)
          XCTAssertEqual(Array(reboundBuffer), [17, 17])
          reboundBuffer[0] = 5
          reboundBuffer[1] = 6
        }
        XCTAssertEqual(tensor3.scalars, [5, 6])
        
        let tensor4 = Tensor<Float>([10, 10])
        device.readTensor(tensor4._rawTensorHandle, true) { buffer in
          let reboundBuffer = buffer.assumingMemoryBound(to: Float.self)
          XCTAssertEqual(Array(reboundBuffer), [10, 10])
          reboundBuffer[0] = 4
          reboundBuffer[1] = 4
        }
        XCTAssertEqual(tensor4.scalars, [4, 4])
      }
    }
    
    testModifyingRead(on: MTLPluggableDevice.default)
    testModifyingRead(on: altDevice)
  }
  
  func testConstantFolding() {
    testHeader()
    
    // TODO: Add dedicated constant folding checks to `TensorUnaryOperationTests` utility functions.
    
    // Unary constant folding
    do {
      let tensor1 = Tensor<Float>([2])
      let tensor2 = tensor1.incremented() // 3.0
      let tensor3 = Tensor<Int32>(tensor2) // 3
      let tensor4 = square(tensor3) // 9
      XCTAssertEqual(tensor4.scalars, [9])
    }
    
    // Binary constant folding
    do {
      let tensor1 = Tensor<Float>([2])
      let tensor2 = Tensor<Float>([4])
      let tensor3 = Tensor<Int32>([7])
      
      let tensor4 = tensor1 + tensor2 // 6.0
      let tensor5 = Tensor<Int32>(tensor4) // 6
      let tensor6 = tensor3 * tensor5 // 42
      XCTAssertEqual(tensor6.scalars, [42])
    }
    
    // TODO: Make this test more involved and have interplay with stuff that runs on the GPU.
    // Accomplish this by broadcasting a scalar to interact with a large vector.
  }
}
