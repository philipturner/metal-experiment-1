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
    
    let id3 = Context.generateID(allocationSize: 4)
    Context.withDispatchQueue {
      let allocation3 = Context.global._compilerFetchAllocation(id: id3)
      try! allocation3.materialize()
      
      let contents = allocation3.mtlBuffer!.contents()
      let bufferPointer = UnsafeMutableRawBufferPointer(start: contents, count: 4)
      let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
      ptr[0] = 5.1
      allocation3.initialized = true
    }
    
    usleep(1030)
    
    try! Context.read(id: id3) { bufferPointer in
      let source = bufferPointer.assumingMemoryBound(to: Float.self)
      XCTAssertEqual(source[0], 5.1)
    }
  }
}
