import XCTest
@testable import MetalExperiment1

func testHeader(_ message: String, disableBarrier: Bool = false) {
  if !disableBarrier {
    Context.global.barrier()
  }
  print()
  print("=== \(message) ===")
}

final class MetalExperiment1Tests: XCTestCase {
  // Force this to execute first.
  func testA() throws {
    testHeader("Initialize context", disableBarrier: true)
    _ = Context.global
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
