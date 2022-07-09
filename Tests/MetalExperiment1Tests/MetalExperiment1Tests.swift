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
    ctx.commitComputeCommand()
    Profiler.log("First cmdbuf")
    ctx.validate(withBarrier: true, showingStats: true)
    
    func profileStream(length: Int, message: String) {
      print("--- Stream size: \(length) ---")
      Profiler.checkpoint()
      for _ in 0..<length {
        ctx.commitComputeCommand()
      }
      let executionTime = Profiler.checkpoint()
      ctx.validate(withBarrier: true, showingStats: false)
      
      print("\(message): \(executionTime / length) \(Profiler.timeUnit)")
    }
    
    profileStream(length: 5, message: "Next 5 cmdbufs")
    profileStream(length: Context.maxCommandBuffers, message: "Average overhead")
    profileStream(length: Context.maxCommandBuffers / 2, message: "Average overhead")
    profileStream(length: Context.maxCommandBuffers / 2, message: "Average overhead")
    profileStream(length: Context.maxCommandBuffers / 2, message: "Average overhead")
    profileStream(length: Context.maxCommandBuffers, message: "Average overhead")
    profileStream(length: Context.maxCommandBuffers / 2, message: "Average overhead")
  }
}
