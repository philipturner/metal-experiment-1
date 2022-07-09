import XCTest
@testable import MetalExperiment1

func testHeader(_ message: String, disableBarrier: Bool = false) {
  if !disableBarrier {
    Context.global.barrier()
  }
  print("=== \(message) ===")
}

final class MetalExperiment1Tests: XCTestCase {
  // Force this to execute first.
  func testA() throws {
    testHeader("Initialize context", disableBarrier: true)
    _ = Context.global
  }
  
  func testB() throws {
    testHeader("Empty cmdbuf throughput")
    let ctx = Context.global
    
    Profiler.checkpoint()
    ctx.commitEmptyCommandBuffer()
    Profiler.log("First cmdbuf")
    ctx.barrier(showingStats: true)
    
    let numCommandBuffers = 20
    
    Profiler.checkpoint()
    for _ in 0..<numCommandBuffers {
      ctx.commitEmptyCommandBuffer()
    }
    let executionTime = Profiler.checkpoint()
    ctx.barrier()
    
    print("Average overhead: \(executionTime / UInt64(numCommandBuffers)) \(Profiler.timeUnit)")
  }
}
