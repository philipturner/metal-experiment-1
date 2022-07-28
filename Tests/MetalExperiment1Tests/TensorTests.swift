import XCTest
@testable import MetalExperiment1

final class TensorTests: XCTestCase {
  func testTensor() throws {
    testHeader("Tensor")
    
    // Warm up the backend.
    do {
      _ = Tensor<Float>(repeating: 1, shape: [100]).incremented().scalars
      _ = Tensor<Float>(repeating: 1, shape: [100]).incremented().scalars
    }
    
    Profiler.checkpoint()
    let tensor3 = Tensor<Float>(repeating: 5, shape: [2, 2])
    XCTAssertEqual(tensor3.shape, [2, 2])
    XCTAssertEqual(tensor3.scalars, [5, 5, 5, 5])
    
    let tensor4 = tensor3.incremented()
    XCTAssertEqual(tensor4.shape, [2, 2])
    XCTAssertEqual(tensor4.scalars, [6, 6, 6, 6])
    Profiler.log("Generic tensor operation execution time")
  }
  
  func testTensorShape() throws {
    testHeader("Tensor shape")
    
    // Warm up the backend.
    do {
      _ = Tensor<Float>(repeating: 1, shape: [100]).incremented().scalars
      _ = Tensor<Float>(repeating: 1, shape: [100]).incremented().scalars
    }
    
    Profiler.checkpoint()
    let tensor = Tensor<Float>(repeating: 7, shape: [4])
    XCTAssertEqual(tensor.scalars, [7, 7, 7, 7])
    
    var shape = tensor.shape
    XCTAssertEqual(String(describing: shape), "[4]")
    
    shape.append(contentsOf: [1])
    XCTAssertEqual(String(describing: shape), "[4, 1]")
    
    shape.removeLast(2)
    XCTAssertEqual(String(describing: shape), "[]")
    
    let tensor2 = Tensor<Float>(repeating: 2, shape: [2])
    var shape2 = tensor2.shape
    shape2.append(3)
    shape2.append(4)
    shape2.append(5)
    shape2.append(6)
    let shape3 = TensorShape(2, 3, 4, 5, 6)
    XCTAssertEqual(shape2, shape3)
    XCTAssertEqual(shape2.dimensions, shape3.dimensions)
    Profiler.log("Tensor shape test execution time")
  }
  
  func testFusion() throws {
    testHeader("Tensor operation fusion")
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    typealias SmallFloat = Float
    #else
    typealias SmallFloat = Float16
    #endif
    
    do {
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 5.005, shape: [2])
        let tensor2 = square(tensor1) // 25.050
        let tensor3 = Tensor<Int8>(tensor2) // 25
        let tensor4 = Tensor<SmallFloat>(tensor3) // 5.0
        let tensor5 = sqrt(tensor4) // 5.0
        // Fusion break
        return tensor5
      }
      XCTAssertEqual(getOutput().scalars, [5.0, 5.0])
    }
    
    // Fusion is interrupted by replacing `Int8` with `UInt64`.
    do {
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 5.005, shape: [2])
        let tensor2 = square(tensor1) // 25.050
        // Fusion break
        let tensor3 = Tensor<Int64>(tensor2) // 25
        let tensor4 = Tensor<SmallFloat>(tensor3) // 5.0
        // Fusion break
        let tensor5 = sqrt(tensor4) // 5.0
        // Fusion break
        return tensor5
      }
      XCTAssertEqual(getOutput().scalars, [5.0, 5.0])
    }
    
    // Binary operation fusion
    do {
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 5.005, shape: [2])
        let tensor2 = square(tensor1) // 25.050
        // Fusion break
        let tensor3 = Tensor<Int64>(tensor2) // 25
        let tensor4 = Tensor<SmallFloat>(tensor3) // 5.0
        // Fusion break
        let tensor5 = sqrt(tensor4) // 5.0
        let tensor6 = min(tensor5, .init(repeating: 4.9, shape: [2])) // 4.9
        let tensor7 = max(tensor6, .init(repeating: 5.1, shape: [2])) // 5.1
        let tensor8 = -tensor7 // -5.1
        // Fusion break
        let tensor9 = max(tensor8, .init(repeating: -3.0, shape: [2])) // -3.0
        return tensor9
      }
      XCTAssertEqual(getOutput().scalars, [-3.0, -3.0])
    }
    
    // Binary operation fusion (broken)
    do {
      // First fusion is 4 ops
      // Second fusion is 2 ops
      // Third fusion is 3 ops
      //
      // What second fusion actually is:
      // reg1 = READ input1
      // reg2 = READ input2
      // reg1 = max(reg1, reg2)
      // reg1 = -reg1
      // WRITE reg1
      //
      // What third fusion actually is:
      // 
      //
      // What both could be:
      //
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 25, shape: [2])
        let tensor2 = sqrt(tensor1) // 5.0
        let tensor3 = min(tensor2, .init(repeating: 4.9, shape: [2])) // 4.9
        let tensor4 = max(tensor3, .init(repeating: 5.1, shape: [2])) // 5.1
        let tensor5 = -tensor4 // 5.1
        // Fusion break
        let tensor6 = max(tensor5, .init(repeating: -3.0, shape: [2])) // -3.0
        let tensor7 = -tensor6 // 3.0
        // Fusion break
        let tensor8 = -Tensor<Float>(repeating: -4, shape: [2])
        let tensor9 = max(tensor7, tensor8)
        let tensor10 = square(tensor9) // 16.0
        return tensor10
      }
      XCTAssertEqual(getOutput().scalars, [16.0, 16.0])
    }
  }
}
