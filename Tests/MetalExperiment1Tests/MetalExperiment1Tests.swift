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
    testHeader("Streamed cmdbuf throughput")
    let ctx = Context.global
    
    Profiler.checkpoint()
    ctx.commitStreamedCommand(profilingEncoding: true)
    Profiler.log("First cmdbuf")
    ctx.validate()
    
    func profileStream(length: Int, message: String, profilingEncoding: Bool = false) {
      print("--- Stream size: \(length) ---")
      Profiler.checkpoint()
      for _ in 0..<length {
        ctx.commitStreamedCommand(profilingEncoding: profilingEncoding)
      }
      let executionTime = Profiler.checkpoint()
      ctx.validate()
      
      let trueDuration = Profiler.checkpoint() + executionTime
      print("""
        \(message): \(executionTime / length) \(Profiler.timeUnit), \
        Amortized sequential throughput: \(trueDuration / length) \(Profiler.timeUnit), \
        Total time: \(trueDuration) \(Profiler.timeUnit)
        """)
    }
    
    profileStream(length: 5, message: "Next 5 cmdbufs")
    profileStream(length: Context.maxCommandsPerCmdbuf * 4, message: "Average CPU-side latency")
    profileStream(length: Context.maxCommandsPerCmdbuf * 4, message: "Average CPU-side latency")
    profileStream(length: Context.maxCommandsPerCmdbuf * 4, message: "Average CPU-side latency")
    profileStream(length: Context.maxCommandsPerCmdbuf * 4, message: "Average CPU-side latency")
    profileStream(length: Context.maxCommandsPerCmdbuf * 2, message: "Average CPU-side latency", profilingEncoding: false)
    profileStream(length: 5, message: "Average CPU-side latency", profilingEncoding: true)
    
    Profiler.withLogging("Query active cmdbufs 10 times") {
      for _ in 0..<10 {
        _ = ctx.queryActiveCommandBuffers()
      }
    }
  }
}
