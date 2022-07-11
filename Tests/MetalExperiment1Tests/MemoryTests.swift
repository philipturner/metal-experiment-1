import XCTest
@testable import MetalExperiment1

final class MemoryTests: XCTestCase {
  func testSimpleAllocation() throws {
    testHeader("Simple memory allocation")
    
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
}
