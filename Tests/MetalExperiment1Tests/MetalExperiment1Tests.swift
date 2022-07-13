import XCTest
@testable import MetalExperiment1

func testHeader(_ message: String) {
  Profiler.checkpoint()
  _ = Context.global
  let startupTime = Profiler.checkpoint()
  if startupTime > 1000 {
    print("=== Initialize context ===")
    print("Initialization time: \(startupTime) \(Profiler.timeUnit)")
  }
  
  print()
  print("=== \(message) ===")
  
  // Stop messages about references from flooding the console. You can re-activate this inside a
  // test function if you want.
  Allocation.debugInfoEnabled = false
}

final class MetalExperiment1Tests: XCTestCase {
  func testGCDLatency() throws {
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
  
  func testStreamedBatchThroughput() throws {
    testHeader("Streamed command buffer throughput")
    Context.validate()
    
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
  
  // Alternative throughput test that validates differently.
  func testStreamedBatchThroughput2() throws {
    testHeader("Streamed command buffer throughput 2")
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
}
