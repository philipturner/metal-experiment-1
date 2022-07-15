import XCTest
@testable import MetalExperiment1

final class MetalExperiment1Tests: XCTestCase {
  func testReadPerformance() {
    _ = Context.global
    HeapAllocator.debugInfoEnabled = true
    
    let id1 = Context.generateID(allocationSize: 4)
    let id2 = Context.generateID(allocationSize: 4)
    #if true
    Context.commitIncrement(inputID: id1, outputID: id2)
    #else
    
    Context.withDispatchQueue {
      let ctx = Context.global
      let input = ctx._compilerFetchAllocation(id: id1)
      let output = ctx._compilerFetchAllocation(id: id2)
      output.initialized = true

      let cmdbuf = ctx.commandQueue.makeCommandBuffer()!
      ctx.commandBufferDictionary[0] = cmdbuf
      try! input.materialize()
      try! output.materialize()
//      output.lastModifiedCommandBufferID = 0

      let encoder = cmdbuf.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(ctx.unaryComputePipeline)
      encoder.setBuffer(input.mtlBuffer!, offset: 0, index: 0)
      encoder.setBuffer(output.mtlBuffer!, offset: 0, index: 1)

      var bytes: Float = 1
      encoder.setBytes(&bytes, length: MemoryLayout.stride(ofValue: bytes), index: 2)
      encoder.dispatchThreadgroups(1, threadsPerThreadgroup: 1)
      encoder.endEncoding()
      cmdbuf.addCompletedHandler { _ in
        Context._dispatchQueue.async {
//          ctx.commandBufferDictionary[0] = nil

          _ = input
          _ = output

        }
      }
      cmdbuf.commit()
//      cmdbuf.waitUntilCompleted()

    }

//    Context.global.commandBufferDictionary[0]!.waitUntilCompleted()
    #endif
    
    try! Context.release(id: id1)
    try! Context.release(id: id2)
    
    do {
      let id1 = Context.generateID(allocationSize: 4)
      let id2 = Context.generateID(allocationSize: 4)
      try! Context.initialize(id: id1) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        ptr[0] = 5.1
      }
      
      Context.withDispatchQueue {
        let ctx = Context.global
        let input = ctx._compilerFetchAllocation(id: id1)
        let output = ctx._compilerFetchAllocation(id: id2)
//        output.initialized = true
        
        try! input.materialize()
        try! output.materialize()
        
        usleep(1030)
      }
      
//      try! Context.read(id: id2) { _ in }
      try! Context.read(id: id1) { bufferPointer in
        let source = bufferPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(source[0], 5.1)
      }
      
      try! Context.release(id: id2)
      try! Context.release(id: id1)
    }
  }
}
