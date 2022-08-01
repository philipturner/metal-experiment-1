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
    
    let tensor5 = Tensor<Float>([1, 2, 3, 4, 5])
    let tensor6 = Tensor<UInt8>([1, 2, 3, 4, 5])
    let tensor7 = Tensor<Bool>([false, true, false])
    let tensor8 = Tensor<Int64>([1, 2, 3, 4, 5])
    XCTAssertEqual(tensor5.scalars, [1, 2, 3, 4, 5])
    XCTAssertEqual(tensor6.scalars, [1, 2, 3, 4, 5])
    XCTAssertEqual(tensor7.scalars, [false, true, false])
    XCTAssertEqual(tensor8.scalars, [1, 2, 3, 4, 5])
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
    
    // Enable this to check the dump. Compilation is non-deterministic, so never assert that a
    // fusion produces any specific output. The comments below show what should happen in the most
    // optimal situation.
    #if false
    defaultPluggableDevice.barrier()
    Instruction.Elementwise.enableDump = true
    defer {
      defaultPluggableDevice.barrier()
      Instruction.Elementwise.enableDump = false
    }
    let showMarkers = true
    #else
    // Suppress warning "Will never be executed".
//    let showMarkers = false
    let showMarkers = Bool.random() && false
    #endif
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    typealias SmallFloat = Float16
    #else
    typealias SmallFloat = Float
    #endif
    
    do {
      if showMarkers {
        print("MARKER 1")
      }
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 5.005, shape: [2])
        let tensor2 = square(tensor1) // 25.050
        let tensor3 = Tensor<Int8>(tensor2) // 25
        let tensor4 = Tensor<SmallFloat>(tensor3) // 5.0
        let tensor5 = sqrt(tensor4) // 5.0
        return tensor5
      }
      XCTAssertEqual(getOutput().scalars, [5.0, 5.0])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      reg1 = square_f32(reg1)
      reg1 = cast_f32_to_i32(reg1)
      reg1 = cast_i32_to_f16(reg1)
      reg1 = sqrt_f32(reg1)
      output[i] = reg1
      #endif
    }
    
    // Fusion is interrupted by replacing `Int8` with `UInt64`.
    do {
      if showMarkers {
        print("MARKER 2")
      }
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<Float>(repeating: 5.005, shape: [2])
        let tensor2 = square(tensor1) // 25.050
        // Fusion break
        let tensor3 = Tensor<Int64>(tensor2) // 25
        let tensor4 = Tensor<SmallFloat>(tensor3) // 5.0
        // Fusion break
        let tensor5 = sqrt(tensor4) // 5.0
        return tensor5
      }
      XCTAssertEqual(getOutput().scalars, [5.0, 5.0])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      reg1 = square_f32(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      reg1 = cast_f32_to_i64(reg1)
      reg1 = cast_i64_to_f16(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      reg1 = sqrt_f32(reg1)
      output[i] = reg1
      #endif
    }
    
    // Binary operation fusion
    do {
      if showMarkers {
        print("MARKER 3")
      }
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
        let tensor9 = max(tensor8, .init(repeating: -3.0, shape: [2])) // -3.0
        return tensor9
      }
      XCTAssertEqual(getOutput().scalars, [-3.0, -3.0])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      reg1 = square_f32(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      reg1 = cast_f32_to_i64(reg1)
      reg1 = cast_i64_to_f16(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      var reg4 = input4[i]
      reg1 = sqrt_f32(reg1)
      reg1 = minimum_f32(reg1, reg2)
      swap(&reg2, &reg3)
      reg1 = maximum_f32(reg1, reg2)
      reg1 = neg_f32(reg1)
      swap(&reg2, &reg4)
      reg1 = maximum_f32(reg1, reg2)
      output[i] = reg1
      #endif
    }
    
    // Binary operation fusion (non-adjacent)
    do {
      if showMarkers {
        print("MARKER 4")
      }
      func getOutput() -> Tensor<Float> {
        let tensor1 = Tensor<Float>(repeating: 25, shape: [2])
        let tensor2 = sqrt(tensor1) // 5.0
        let tensor3 = min(tensor2, .init(repeating: 4.9, shape: [2])) // 4.9
        let tensor4 = max(tensor3, .init(repeating: 5.1, shape: [2])) // 5.1
        let tensor5 = max(tensor4, .init(repeating: 5.1, shape: [2])) // 5.1
        let tensor6 = -tensor5 // 5.1
        // Fusion break
        let tensor7 = max(tensor6, .init(repeating: -3.0, shape: [2])) // -3.0
        let tensor8 = -tensor7 // 3.0
        // Fusion break
        _ = -Tensor<Float>(repeating: -4, shape: [2])
        // Fusion break + context switch
        let tensor9 = max(tensor8, .init(repeating: 4, shape: [2])) // 4.0
        let tensor10 = square(tensor9) // 16.0
        return tensor10
      }
      XCTAssertEqual(getOutput().scalars, [16.0, 16.0])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      var reg4 = input4[i]
      reg1 = sqrt_f32(reg1)
      reg1 = minimum_f32(reg1, reg2)
      swap(&reg2, &reg3)
      reg1 = maximum_f32(reg1, reg2)
      swap(&reg2, &reg4)
      reg1 = maximum_f32(reg1, reg2)
      reg1 = neg_f32(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      reg1 = maximum_f32(reg1, reg2)
      reg1 = neg_f32(reg1)
      swap(&reg2, &reg3)
      reg1 = maximum_f32(reg1, reg2)
      reg1 = square_f32(reg1)
      output[i] = reg1
      #endif
    }
    
    // Ternary operation fusion
    do {
      if showMarkers {
        print("MARKER 5")
      }
      func getOutput() -> Tensor<SmallFloat> {
        let tensor1 = Tensor<SmallFloat>([25, 25])
        let tensor2 = sqrt(tensor1) // 5.0
        let tensor3 = tensor2 + Tensor<SmallFloat>([7, 7]) // 12.0
        let tensor4 = tensor3.clipped(min: Tensor([6]), max: Tensor([9])) // 9.0
        // Fusion break
        let mask = Tensor([false, true])
        let tensor5 = tensor4.replacing(with: Tensor([10, 11]), where: mask) // [9.0, 11.0]
        let tensor6 = pow(tensor5, Tensor([3.0, 3.0])) // [729.0, 1331.0]
        let tensor7 = -tensor6 // [-729.0, -1331.0]
        return tensor7
      }
      XCTAssertEqual(getOutput().scalars, [-729.0, -1331.0])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      var reg4 = input4[i]
      reg1 = sqrt_f32(reg1)
      reg1 = add_f32(reg1, reg2)
      swap(&reg2, &reg3)
      swap(&reg3, &reg4)
      reg1 = clip_by_value_f32(reg1, reg2, reg3)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      var reg4 = input4[i]
      reg1 = select_f32_i32(reg1, reg2, reg3)
      swap(&reg2, &reg4)
      reg1 = pow_f32(reg1, reg2)
      reg1 = neg_f32(reg1)
      output[i] = reg1
      #endif
    }
    
    // Ternary operation fusion (non-adjacent)
    do {
      if showMarkers {
        print("MARKER 6")
      }
      func getOutput() -> Tensor<Int32> {
        let tensor1 = sqrt(Tensor<Float>([25, 25])) // 5.0
        // Fusion break
        let tensor2 = sqrt(Tensor<Float>([36, 36])) // 6.0
        // Fusion break
        let tensor3 = Tensor<Float>([8, 8]) + Tensor<Float>([9, 9]) // 17.0
        let tensor4 = Tensor<Int32>(tensor3) // 17
        // Fusion break + context switch
        let tensor5 = Tensor<Float>([-4, 10]).clipped(min: tensor1, max: tensor2) // [5.0, 6.0]
        let tensor6 = Tensor<Int32>(tensor5) // [5, 6]
        let tensor7 = tensor4 + tensor6 // [22, 23]
        return tensor7
      }
      XCTAssertEqual(getOutput().scalars, [22, 23])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      reg1 = sqrt_f32(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      reg1 = add_f32(reg1, reg2)
      reg1 = cast_f32_to_i32(reg1)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      var reg3 = input3[i]
      var reg4 = input4[i]
      reg1 = sqrt_f32(reg1)
      swap(&reg1, &reg2)
      reg1 = clip_by_value_f32(reg1, reg2, reg3)
      reg1 = cast_f32_to_i32(reg1)
      swap(&reg2, &reg4)
      swap(&reg1, &reg2)
      reg1 = add_i32(reg1, reg2)
      output[i] = reg1
      #endif
    }
    
    // Non-adjacent fusion resulting in a zombie.
    do {
      if showMarkers {
        print("MARKER 7")
      }
      func getOutput() -> Tensor<Float> {
        let tensor1 = sqrt(Tensor<Float>([25, 25, 25])) // 5.0
        let tensor2 = Tensor<Float>([2, 2, 2])
        let tensor3 = Tensor<Float>([3, 3, 3])
        let tensor4 = max(tensor2, tensor3) // 3.0
        // Fusion break
        let tensor5 = Tensor<Float>([4, 4, 4])
        _ = -tensor5 // Zombie tensor
        // Fusion break + context switch
        _ = Tensor<Int32>(tensor4) // Zombie tensor
        return tensor1
      }
      XCTAssertEqual(getOutput().scalars, [5, 5, 5])
      
      #if false // Dumped instructions
      var reg1 = input1[i]
      reg1 = sqrt_f32(reg1)
      output[i] = reg1
      #endif
    }
    
    // Binary operation fusion (non-adjacent, favored context switch)
    do {
      if showMarkers {
        print("MARKER 8")
      }
      // highlight the chosen fusion
      func getOutput() -> Tensor<Float> {
        let tensor1 = sqrt(Tensor<Float>([25, 25])) // 5.0
        let tensor2 = max(tensor1, Tensor<Float>([6])) // 6.0; scalar broadcasting
        // Fusion break; disfavored for re-entry below
        let tensor3 = -Tensor<Float>([7, 7]) // -7.0
        // Fusion break; favored for re-entry below
        _ = Tensor<Float>([8, 8]).incremented() // Zombie tensor
        // Fusion break + context switch
        let tensor4 = tensor2 + tensor3 // -1.0
        return tensor4
      }
      XCTAssertEqual(getOutput().scalars, [-1, -1])
      
      #if false
      var reg1 = input1[i]
      var reg2 = input2[i]
      reg1 = sqrt_f32(reg1)
      reg1 = maximum_f32(reg1, reg2)
      output[i] = reg1
      
      var reg1 = input1[i]
      var reg2 = input2[i]
      reg1 = neg_f32(reg1)
      swap(&reg1, &reg2)
      reg1 = add_f32(reg1, reg2)
      output[i] = reg1
      #endif
    }
  }
}

