import XCTest
@testable import MetalExperiment1

func testHeader(_ message: String, disableBarrier: Bool = false) {
  if !disableBarrier {
    Context.barrier()
  }
  print()
  print("=== \(message) ===")
}

final class MetalExperiment1Tests: XCTestCase {
  // Force this to execute first.
  func testA() throws {
    testHeader("Initialize context", disableBarrier: true)
    Profiler.checkpoint()
    _ = Context.global
    Profiler.log("Initialization time")
  }
  
  func testB() throws {
    testHeader("Dispatch queue latency")
    
    for _ in 0..<2 {
      Profiler.checkpoint()
      _ = Context.dispatchQueue.sync {
        Bool.random()
      }
      Profiler.log("Dispatch queue latency")
    }
    
    Profiler.checkpoint()
    let iterations = 100
    for _ in 0..<iterations {
      _ = Context.dispatchQueue.sync {
        Bool.random()
      }
    }
    let totalTime = Profiler.checkpoint()
    let throughput = Double(totalTime) / Double(iterations)
    print("Dispatch queue throughput: \(throughput) \(Profiler.timeUnit)")
  }
  
  func testC() throws {
    testHeader("Streamed command buffer throughput")
    
    do {
      Profiler.checkpoint()
      Context.commitStreamedCommand()
      let latency = Profiler.checkpoint()
      
      Context.validate()
      let totalTime = latency + Profiler.checkpoint()
      print("""
        First batch latency: \(latency) \(Profiler.timeUnit), \
        Total time: \(totalTime) \(Profiler.timeUnit)
        """)
    }
    
    func profileStream(length: Int) {
      print("--- Stream size: \(length) ---")
      Profiler.checkpoint()
      for _ in 0..<length {
        Context.commitStreamedCommand()
      }
      let latency = Profiler.checkpoint()
      Context.validate()
      
      let totalTime = latency + Profiler.checkpoint()
      print("""
        Average CPU-side latency: \(latency / length) \(Profiler.timeUnit), \
        Amortized sequential throughput: \(totalTime / length) \(Profiler.timeUnit), \
        Total time: \(totalTime) \(Profiler.timeUnit)
        """)
    }
    
    for _ in 0..<5 {
      profileStream(length: Context.maxCommandsPerBatch * 4)
    }
  }
  
  private struct ARCEvent: Equatable {
    var id: Int
    var wasAllocating: Bool
  }
  private static var history: [ARCEvent] = []
  
  // Enables automatic fusion of unary ops, with no extra work from the frontend. Deallocated
  // (invalid) tensors become placeholders for operator fusion.
  func testD() throws {
    testHeader("Automatic ARC deallocation")
    
    Self.history = []
    
    class DummyTensorHandle {
      var id: Int
      
      init(id: Int) {
        self.id = id
        MetalExperiment1Tests.history.append(ARCEvent(id: id, wasAllocating: true))
      }
      
      deinit {
        MetalExperiment1Tests.history.append(ARCEvent(id: id, wasAllocating: false))
      }
    }
    
    struct DummyTensor {
      private var handle: DummyTensorHandle
      
      init(id: Int) {
        self.handle = DummyTensorHandle(id: id)
      }
    }
    
    // Test whether inlining changes things.
    
    @inline(never)
    func f1(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 1)
    }
    
    func f2(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 2)
    }
    
    @inline(__always)
    func f3(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 3)
    }
    
    func f4(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 4)
    }
    
    _ = f4(f3(f2(f1(DummyTensor(id: 0)))))
    
    let expectedHistory: [ARCEvent] = [
      // Explicitly declared input to `function1`.
      .init(id: 0, wasAllocating: true),
      
      // Call into `f1`.
      .init(id: 1, wasAllocating: true),
      .init(id: 0, wasAllocating: false),
      
      // Call into `f2`.
      .init(id: 2, wasAllocating: true),
      .init(id: 1, wasAllocating: false),
      
      // Call into `f3`.
      .init(id: 3, wasAllocating: true),
      .init(id: 2, wasAllocating: false),
      
      // Call into `f3`.
      .init(id: 4, wasAllocating: true),
      .init(id: 3, wasAllocating: false),
      
      // Discarded result of `f4`.
      .init(id: 4, wasAllocating: false)
    ]
    XCTAssertEqual(Self.history, expectedHistory)
  }
  
  func testE() throws {
    testHeader("Memory allocation")
    
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
      
      let blocks = customSet.blocks
      XCTAssertEqual(blocks[0].size, 4)
      XCTAssertEqual(blocks[0].size, 6)
      XCTAssertEqual(blocks[0].size, 6)
      XCTAssertEqual(blocks[0].size, 6)
      XCTAssertEqual(blocks[0].size, 8)
    }
  }
}


