import XCTest
@testable import MetalExperiment1

final class MetalExperiment1Tests: XCTestCase {
  func testReadPerformance() {
    _ = Context.global
    HeapAllocator.debugInfoEnabled = true
    
    let id1 = Context.generateID(allocationSize: 4)
    let id2 = Context.generateID(allocationSize: 4)
    Context.commitIncrement(inputID: id1, outputID: id2)
    
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
        
//        try! input.materialize()
        try! output.materialize()
        
        usleep(1030)
      }
      
      try! Context.read(id: id1) { bufferPointer in
        let source = bufferPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(source[0], 5.1)
      }
      
      try! Context.release(id: id2)
      try! Context.release(id: id1)
    }
  }
}
