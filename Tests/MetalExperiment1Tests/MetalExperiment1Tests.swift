import XCTest
@testable import MetalExperiment1

final class MetalExperiment1Tests: XCTestCase {
  func testReadPerformance() {
    _ = Context.global
    let allocationSize = 8
    do {
      let id_ = Context.generateID(allocationSize: allocationSize)
      try! Context.initialize(id: id_) { bufferPointer in
        let ptr = bufferPointer.assumingMemoryBound(to: Float.self)
        ptr.initialize(repeating: 4)
      }
      try! Context.release(id: id_)
    }
    
    do {
      let handle1 = TensorHandle(repeating: 5, count: 2)
      let handle2 = handle1.incremented()
      
      let id3 = Context.generateID(allocationSize: allocationSize)
      try! Context.release(id: id3)
    }
    
    do {
      let handle1 = TensorHandle(repeating: 5.1, count: 2)
      
      let handle2 = handle1.incremented()
      _ = handle2.copyScalars()
      
      let handle3 = handle1.incremented()
      XCTAssertEqual(handle1.copyScalars(), [5.1, 5.1])
    }
  }
}
