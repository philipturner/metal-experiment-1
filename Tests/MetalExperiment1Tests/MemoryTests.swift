import XCTest
@testable import MetalExperiment1

final class MemoryTests: XCTestCase {
  func testSimpleAllocation() throws {
    testHeader("Simple memory allocation")
    HeapAllocator.global._releaseCachedBufferBlocks()
    
    do {
      let firstID = Context.generateID(allocationSize: 4000)
      let secondID = Context.generateID(allocationSize: 4000)
      
      try! Context.deallocate(id: firstID)
      try! Context.deallocate(id: secondID)
    }
    
    do {
      Profiler.checkpoint()
      let numIds = 100
      for _ in 0..<numIds {
        let id = Context.generateID(allocationSize: 4000)
        try! Context.deallocate(id: id)
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(numIds)
      print("Unused ID creation throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    func assertErrorMessage(_ body: @autoclosure () throws -> Void, _ message: String) {
      var threwError = false
      do {
        try body()
      } catch let error as AllocationError {
        threwError = true
        XCTAssertEqual(error.message, message)
      } catch {
        XCTFail(error.localizedDescription)
      }
      if !threwError {
        XCTFail("Should have thrown an error here.")
      }
    }
    
    do {
      let id = Context.generateID(allocationSize: 4000)
      defer { try! Context.deallocate(id: id) }
      assertErrorMessage(
        try Context.read(id: id) { _ in }, "Read from memory with a null underlying `MTLBuffer`.")
    }
    
    do {
      let id = Context.generateID(allocationSize: 4000)
      try! Context.initialize(id: id) { mutableBufferPointer in
        let ptr = mutableBufferPointer.assumingMemoryBound(to: Float.self)
        ptr.initialize(repeating: 2.5)
      }
      assertErrorMessage(
        try Context.initialize(id: id) { _ in }, "Cannot initialize something twice.")
      
      var wereEqual = false
      try! Context.read(id: id) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        let comparisonSequence = [Float](repeating: 2.5, count: 1000)
        wereEqual = ptr.elementsEqual(comparisonSequence)
      }
      XCTAssert(wereEqual)
      
      // Try accessing the buffer after it's deallocated.
      try! Context.deallocate(id: id)
      assertErrorMessage(
        try Context.initialize(id: id) { _ in }, "Tried to initialize memory that was deallocated.")
      assertErrorMessage(
        try Context.read(id: id) { _ in }, "Tried to read from memory that was deallocated.")
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
    
    print("Debug info enabled: \(HeapAllocator.debugInfoEnabled)")
  }
  
  func testRecyclingThroughput() throws {
    testHeader("Memory recycling throughput")
    HeapAllocator.global._releaseCachedBufferBlocks()
    
    func allocateDeallocate(bufferSize: Int, numBuffers: Int) throws {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = Context.generateID(allocationSize: bufferSize)
        ids.append(id)
      }
      for id in ids {
        try Context.initialize(id: id) { _ in }
      }
      for id in ids {
        try Context.deallocate(id: id)
      }
    }
    func fakeAllocateDeallocate(numBuffers: Int) throws {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = Context.dispatchQueue.sync {
          return UInt64(2)
        }
        ids.append(id)
      }
      for id in ids {
        Context.dispatchQueue.sync {
          _ = id
        }
      }
      for id in ids {
        Context.dispatchQueue.sync {
          _ = id
        }
      }
    }
    func emptyAllocateDeallocate(bufferSize: Int, numBuffers: Int) throws {
      var ids: [UInt64] = []
      for _ in 0..<numBuffers {
        let id = Context.generateID(allocationSize: bufferSize)
        ids.append(id)
      }
      for id in ids {
        Context.dispatchQueue.sync {
          _ = id
        }
      }
      for id in ids {
        try Context.deallocate(id: id)
      }
    }
    
    try! allocateDeallocate(bufferSize: 4000, numBuffers: 5)
    Profiler.checkpoint()
    try! allocateDeallocate(bufferSize: 1000, numBuffers: 5)
    try! allocateDeallocate(bufferSize: 2000, numBuffers: 5)
    try! allocateDeallocate(bufferSize: 3000, numBuffers: 5)
    try! allocateDeallocate(bufferSize: 4095, numBuffers: 5)
    let totalTime = Profiler.checkpoint()
    
    Profiler.checkpoint()
    try! fakeAllocateDeallocate(numBuffers: 5)
    try! fakeAllocateDeallocate(numBuffers: 5)
    try! fakeAllocateDeallocate(numBuffers: 5)
    try! fakeAllocateDeallocate(numBuffers: 5)
    let gcdTime = Profiler.checkpoint()
    
    Profiler.checkpoint()
    try! emptyAllocateDeallocate(bufferSize: 1000, numBuffers: 5)
    try! emptyAllocateDeallocate(bufferSize: 2000, numBuffers: 5)
    try! emptyAllocateDeallocate(bufferSize: 3000, numBuffers: 5)
    try! emptyAllocateDeallocate(bufferSize: 4095, numBuffers: 5)
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
    
    func allocate(size: Int) -> UInt64 {
      let id = Context.generateID(allocationSize: size)
      try! Context.initialize(id: id) { _ in }
      return id
    }
    func deallocate(id: UInt64) {
      try! Context.deallocate(id: id)
    }
    
    let id1 = allocate(size: 8_000_000)
    let id2 = allocate(size: 12_000_000)
    let id3 = allocate(size: 12_000_000)
    deallocate(id: id1)
    deallocate(id: id2)
    deallocate(id: id3)
    
    let id4 = allocate(size: 999_000)
    deallocate(id: id4)
    
    let id5 = allocate(size: 2_000_000)
    deallocate(id: id5)
    
    // Test mechanism for dealing with excessive memory allocation.
    
    do {
      HeapAllocator.global._releaseCachedBufferBlocks()
      let smallBufferID1 = allocate(size: 1_000)
      defer { deallocate(id: smallBufferID1) }
      Context.dispatchQueue.sync {
        Context.global.permitExceedingSystemRAM = true
      }
      
      let largeBufferSize = Context.global.device.maxBufferLength
      let largeBufferID1 = allocate(size: largeBufferSize)
      defer { deallocate(id: largeBufferID1) }
      Context.dispatchQueue.sync {
        XCTAssert(Context.global.permitExceedingSystemRAM)
      }
      
      let smallBufferID2 = allocate(size: 1_000)
      defer { deallocate(id: smallBufferID2) }
      Context.dispatchQueue.sync {
        XCTAssert(Context.global.permitExceedingSystemRAM)
      }
    }
    Context.dispatchQueue.sync {
      XCTAssert(Context.global.permitExceedingSystemRAM)
    }
    
    do {
      let smallBufferID3 = allocate(size: 1_000)
      defer { deallocate(id: smallBufferID3) }
      Context.dispatchQueue.sync {
        XCTAssert(Context.global.permitExceedingSystemRAM)
      }
      
      HeapAllocator.global._releaseCachedBufferBlocks()
      let smallBufferID4 = allocate(size: 1_000)
      defer { deallocate(id: smallBufferID4) }
      Context.dispatchQueue.sync {
        XCTAssertFalse(Context.global.permitExceedingSystemRAM)
      }
    }
  }
}