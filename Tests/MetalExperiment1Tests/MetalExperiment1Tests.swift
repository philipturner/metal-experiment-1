import XCTest
@testable import MetalExperiment1

func testHeader(_ message: String? = nil) {
  Profiler.checkpoint()
  Context.withDispatchQueue {
    _ = Context.global
  }
  let startupTime = Profiler.checkpoint()
  if startupTime > 1000 {
    print("=== Initialize context ===")
    print("Initialization time: \(startupTime) \(Profiler.timeUnit)")
  }
  
  if let message = message {
    print()
    print("=== \(message) ===")
  }
  
  // Stop messages about references from flooding the console. You can re-activate this inside a
  // test function if you want.
  Context.withDispatchQueue {
    Allocation.debugInfoEnabled = false
  }
  Context.barrier()
}

final class MetalExperiment1Tests: XCTestCase {
  func testGCDLatency() throws {
    testHeader("Dispatch queue latency")
    
    for _ in 0..<2 {
      Profiler.checkpoint()
      _ = Context.withDispatchQueue {
        Bool.random()
      }
      Profiler.log("Dispatch queue latency")
    }
    
    do {
      Profiler.checkpoint()
      let iterations = 100
      for _ in 0..<iterations {
        _ = Context.withDispatchQueue {
          Bool.random()
        }
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("Dispatch queue throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    do {
      Profiler.checkpoint()
      let iterations = 1000
      for _ in 0..<iterations {
        Context.testSynchronizationOverhead()
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("Synchronization throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
  }
  
  func testStreamedBatchThroughput() throws {
    testHeader("Streamed command buffer throughput")
    
    func validate(_ tensorHandle: TensorHandle, value: Float) {
      XCTAssertEqual(tensorHandle.makeHostCopy()[0], value)
    }
    
    func testWarmup(name: String) {
      print("--- Stream size: 1")
      Profiler.checkpoint()
      let handle1 = TensorHandle(repeating: 0, count: 10)
      let creationTime = Profiler.checkpoint()
      
      let handle2 = handle1.incremented()
      let latency = Profiler.checkpoint()
      validate(handle2, value: 1.0)
      
      let totalTime = latency + Profiler.checkpoint()
      print("""
        Creation time: \(creationTime) \(Profiler.timeUnit)
        First batch latency: \(latency) \(Profiler.timeUnit)
        Total time: \(totalTime) \(Profiler.timeUnit)
        """)
    }
    testWarmup(name: "First")
    testWarmup(name: "Second")
    
    func profileStream(length: Int) {
      print("--- Stream size: \(length)")
      Profiler.checkpoint()
      let handle = Context.withDispatchQueue {
        var handle = TensorHandle(repeating: 0, count: 10)
        for _ in 0..<length {
          handle = handle.incremented()
        }
        return handle
      }
      let latency = Profiler.checkpoint()
      validate(handle, value: Float(length))
      
      let totalTime = latency + Profiler.checkpoint()
      let latencyAverage = Double(latency * 10 / length) / 10
      let throughputAverage = Double(totalTime * 10 / length) / 10
      print("""
        Average CPU-side latency: \(latencyAverage) \(Profiler.timeUnit), \
        Amortized sequential throughput: \(throughputAverage) \(Profiler.timeUnit), \
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
  func testARCDeallocation() throws {
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
  
  func testOperationTypeList() throws {
    testHeader("OperationTypeList")
    
    enum TestOperationType: UInt8, CaseIterable {
      case type1
      case type2
      case type3
      case type4
    }
    
    var list1 = OperationTypeList2<TestOperationType>()
    list1.append(.type2)
    XCTAssertEqual(list1.count, 1)
    XCTAssertEqual(list1[0], .type2)
    
    list1.append(.type3)
    list1.append(.type4)
    XCTAssertEqual(list1.count, 3)
    XCTAssertEqual(list1[0], .type2)
    XCTAssertEqual(list1[1], .type3)
    XCTAssertEqual(list1[2], .type4)
    
    // Would crash with a regular Swift array, but is acceptable here.
    let list2 = OperationTypeList16<TestOperationType>()
    XCTAssertEqual(list2.count, 0)
    XCTAssertEqual(TestOperationType.type1.rawValue, 0)
    XCTAssertEqual(list2[0], .type1)
    XCTAssertEqual(list2[1], .type1)
  }
}
