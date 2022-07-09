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
    let ctx = Context.global
    
    Profiler.checkpoint()
    ctx.commitStreamedCommand()
    Profiler.log("First command buffer")
    ctx.validate()
    
    func profileStream(length: Int) {
      print("--- Stream size: \(length) ---")
      Profiler.checkpoint()
      for _ in 0..<length {
        ctx.commitStreamedCommand()
      }
      let executionTime = Profiler.checkpoint()
      ctx.validate()
      
      let trueDuration = Profiler.checkpoint() + executionTime
      print("""
        Average CPU-side latency: \(executionTime / length) \(Profiler.timeUnit), \
        Amortized sequential throughput: \(trueDuration / length) \(Profiler.timeUnit), \
        Total time: \(trueDuration) \(Profiler.timeUnit)
        """)
    }
    
    for _ in 0..<5 {
      profileStream(length: Context.maxCommandsPerBatch * 8)
    }
  }
}
