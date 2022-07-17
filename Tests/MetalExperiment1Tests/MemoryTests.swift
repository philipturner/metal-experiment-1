import XCTest
@testable import MetalExperiment1

fileprivate func allocate(capacity: Int) -> UInt64 {
  withUnsafeTemporaryAllocation(of: Int.self, capacity: 1) { shape in
    shape[0] = capacity
    let (id, _) = Context.allocateTensor(Float.self, UnsafeBufferPointer(shape))
    return id
  }
}

final class MemoryTests: XCTestCase {
  func testSimpleAllocation() throws {
    testHeader("Simple memory allocation")
    HeapAllocator.global._releaseCachedBufferBlocks()
    
    do {
      let firstID = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
      let secondID = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
      Context.deleteTensor(firstID)
      Context.deleteTensor(secondID)
    }
    
    do {
      Profiler.checkpoint()
      let numIds = 100
      for _ in 0..<numIds {
        let id = allocate(capacity: 1000 / MemoryLayout<Float>.stride)
        Context.deleteTensor(id)
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(numIds)
      print("Unused ID creation throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    do {
      let id = allocate(capacity: 4000 / MemoryLayout<Float>.stride)
      defer { Context.deleteTensor(id) }
      
      Context.initializeTensor(id) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        ptr.initialize(repeating: 2.5)
      }
      var wereEqual = false
      Context.readTensor(id) { bufferPointer in
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
    HeapAllocator.global._releaseCachedBufferBlocks()
    
    func allocateDeallocate(bufferSize: Int, numBuffers: Int) {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = allocate(capacity: bufferSize / MemoryLayout<Float>.stride)
        ids.append(id)
      }
      for id in ids {
        Context.initializeTensor(id) { _ in }
      }
      for id in ids {
        Context.deleteTensor(id)
      }
    }
    func fakeAllocateDeallocate(numBuffers: Int) {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = Context.withDispatchQueue {
          return UInt64(2)
        }
        ids.append(id)
      }
      for id in ids {
        Context.withDispatchQueue {
          _ = id
        }
      }
      for id in ids {
        Context.withDispatchQueue {
          _ = id
        }
      }
    }
    func emptyAllocateDeallocate(bufferSize: Int, numBuffers: Int) {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = allocate(capacity: bufferSize / MemoryLayout<Float>.stride)
        ids.append(id)
      }
      for id in ids {
        Context.withDispatchQueue {
          _ = id
        }
      }
      for id in ids {
        Context.deleteTensor(id)
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
    let idCycleTime = Profiler.checkpoint()
    
    let totalThroughput = Double(totalTime) / 20
    print("Memory recycling throughput: \(totalThroughput) \(Profiler.timeUnit)")
    let nonGCDThroughput = Double(totalTime - gcdTime) / 20
    print("Time excluding GCD: \(nonGCDThroughput) \(Profiler.timeUnit)")
    let allocationThroughput = Double(totalTime - idCycleTime) / 20
    print("Time inside HeapAllocator: \(allocationThroughput) \(Profiler.timeUnit)")
  }
  
  func testComplexAllocation() throws {
    testHeader("Complex memory allocation")
    HeapAllocator.global._releaseCachedBufferBlocks()
    
    
    func allocate(byteCount: Int) -> UInt64 {
      // The compiler mistakes this for `allocate(byteCount:)`.
      let _avoidNameCollision = allocate(capacity:)
      let id = _avoidNameCollision(byteCount / MemoryLayout<Float>.stride)
      Context.initializeTensor(id) { _ in }
      return id
    }
    func deallocate(id: UInt64) {
      Context.deleteTensor(id)
    }
    
    let id1 = allocate(byteCount: 8_000_000)
    let id2 = allocate(byteCount: 12_000_000)
    let id3 = allocate(byteCount: 12_000_000)
    deallocate(id: id1)
    deallocate(id: id2)
    deallocate(id: id3)
    
    let id4 = allocate(byteCount: 999_000)
    deallocate(id: id4)
    
    let id5 = allocate(byteCount: 2_000_000)
    deallocate(id: id5)
    
    // Test mechanism for dealing with excessive memory allocation.
    
    do {
      HeapAllocator.global._releaseCachedBufferBlocks()
      let smallBufferID1 = allocate(byteCount: 1_000)
      defer { deallocate(id: smallBufferID1) }
      Context.withDispatchQueue {
        Context.global.permitExceedingSystemRAM = true
      }
      
      let largeBufferSize = Context.global.device.maxBufferLength
      let largeBufferID1 = allocate(byteCount: largeBufferSize)
      defer { deallocate(id: largeBufferID1) }
      Context.withDispatchQueue {
        XCTAssertTrue(Context.global.permitExceedingSystemRAM)
      }
      
      let smallBufferID2 = allocate(byteCount: 1_000)
      defer { deallocate(id: smallBufferID2) }
      Context.withDispatchQueue {
        XCTAssertTrue(Context.global.permitExceedingSystemRAM)
      }
    }
    Context.withDispatchQueue {
      XCTAssertTrue(Context.global.permitExceedingSystemRAM)
    }
    
    do {
      let smallBufferID3 = allocate(byteCount: 1_000)
      defer { deallocate(id: smallBufferID3) }
      Context.withDispatchQueue {
        XCTAssertTrue(Context.global.permitExceedingSystemRAM)
      }
      
      HeapAllocator.global._releaseCachedBufferBlocks()
      let smallBufferID4 = allocate(byteCount: 1_000)
      defer { deallocate(id: smallBufferID4) }
      Context.withDispatchQueue {
        XCTAssertFalse(Context.global.permitExceedingSystemRAM)
      }
    }
  }
  
  func testTensorHandleLifetime() throws {
    testHeader("Tensor handle lifetime")
    Context.withDispatchQueue {
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
    
    Context.withDispatchQueue {
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
    HeapAllocator.global._releaseCachedBufferBlocks()
    defer {
      HeapAllocator.global._releaseCachedBufferBlocks()
    }
    
    let device = Context.global.device
    #if os(macOS)
    let maxWorkingSize = Int(device.recommendedMaxWorkingSetSize)
    #else
    let maxWorkingSize = device.maxBufferLength
    #endif
    let maxBufferLength = device.maxBufferLength
    
    let bufferSize1 = maxBufferLength - 16384
    let bufferSize2 = (maxWorkingSize - maxBufferLength) + 16384
    let bufferCount3 = 16384 / MemoryLayout<Float>.stride
    
    let bufferID1 = allocate(capacity: bufferSize1  / MemoryLayout<Float>.stride)
    let bufferID2 = allocate(capacity: bufferSize2 / MemoryLayout<Float>.stride)
    defer {
      Context.deleteTensor(bufferID1)
      Context.deleteTensor(bufferID2)
    }
    
    var tensor3: Tensor<Float>?
    Context.withDispatchQueue {
      Context.global.permitExceedingSystemRAM = true
      Context.initializeTensor(bufferID1) { _ in }
      
      Context.global.permitExceedingSystemRAM = true
      Context.initializeTensor(bufferID2) { _ in }
      
      Context.global.permitExceedingSystemRAM = true
      tensor3 = Tensor<Float>(repeating: 0, shape: [bufferCount3])
      Context.global.permitExceedingSystemRAM = false
    }
    guard let tensor3 = tensor3 else {
      fatalError("This should never happen.")
    }
    
    withExtendedLifetime(tensor3) {
      let tensor4 = tensor3.incremented()
      
      let scalars3 = tensor3.scalars
      let scalars4 = tensor4.scalars
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
    Context.withDispatchQueue {
      previousProfilingEncoding = Context.profilingEncoding
      Context.profilingEncoding = true
    }
    defer {
      Context.withDispatchQueue {
        Context.profilingEncoding = previousProfilingEncoding
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
}
