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
    let throughput = totalTime / iterations
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
}
