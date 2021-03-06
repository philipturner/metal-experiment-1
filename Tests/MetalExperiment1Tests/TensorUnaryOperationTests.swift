import XCTest
@testable import MetalExperiment1

func test<T: Equatable>(_ function: (Tensor<T>) -> () -> Tensor<T>, input: T, expected: T) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)()
  XCTAssertEqual(transformedTensor.scalars, [T](repeating: expected, count: 5))
}

func test<T: Equatable>(_ function: (Tensor<T>) -> Tensor<T>, input: T, expected: T) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)
  XCTAssertEqual(transformedTensor.scalars, [T](repeating: expected, count: 5))
}

func test4<T0: Equatable, T1: Equatable, T2: Equatable, T3: Equatable>(
  _ function1: (Tensor<T0>) -> () -> Tensor<T0>, input1: T0, expected1: T0,
  _ function2: (Tensor<T1>) -> () -> Tensor<T1>, input2: T1, expected2: T1,
  _ function3: (Tensor<T2>) -> () -> Tensor<T2>, input3: T2, expected3: T2,
  _ function4: (Tensor<T3>) -> () -> Tensor<T3>, input4: T3, expected4: T3
) {
  // Performs multiple operations before waiting, reducing test execution time.
  #if true
  let tensor1 = function1(Tensor(repeating: input1, shape: [5]))()
  let tensor2 = function2(Tensor(repeating: input2, shape: [5]))()
  let tensor3 = function3(Tensor(repeating: input3, shape: [5]))()
  let tensor4 = function4(Tensor(repeating: input4, shape: [5]))()
  _ = tensor4.scalars
  
  XCTAssertEqual(tensor1.scalars, [T0](repeating: expected1, count: 5))
  XCTAssertEqual(tensor2.scalars, [T1](repeating: expected2, count: 5))
  XCTAssertEqual(tensor3.scalars, [T2](repeating: expected3, count: 5))
  XCTAssertEqual(tensor4.scalars, [T3](repeating: expected4, count: 5))
  #else
  test(function1, input: input1, expected: expected1)
  test(function2, input: input2, expected: expected2)
  test(function3, input: input3, expected: expected3)
  test(function4, input: input4, expected: expected4)
  #endif
}

func test4<T0: Equatable, T1: Equatable, T2: Equatable, T3: Equatable>(
  _ function1: (Tensor<T0>) -> Tensor<T0>, input1: T0, expected1: T0,
  _ function2: (Tensor<T1>) -> Tensor<T1>, input2: T1, expected2: T1,
  _ function3: (Tensor<T2>) -> Tensor<T2>, input3: T2, expected3: T2,
  _ function4: (Tensor<T3>) -> Tensor<T3>, input4: T3, expected4: T3
) {
  // Performs multiple operations before waiting, reducing test execution time.
  #if true
  let tensor1 = function1(Tensor(repeating: input1, shape: [5]))
  let tensor2 = function2(Tensor(repeating: input2, shape: [5]))
  let tensor3 = function3(Tensor(repeating: input3, shape: [5]))
  let tensor4 = function4(Tensor(repeating: input4, shape: [5]))
  _ = tensor4.scalars
  
  XCTAssertEqual(tensor1.scalars, [T0](repeating: expected1, count: 5))
  XCTAssertEqual(tensor2.scalars, [T1](repeating: expected2, count: 5))
  XCTAssertEqual(tensor3.scalars, [T2](repeating: expected3, count: 5))
  XCTAssertEqual(tensor4.scalars, [T3](repeating: expected4, count: 5))
  #else
  test(function1, input: input1, expected: expected1)
  test(function2, input: input2, expected: expected2)
  test(function3, input: input3, expected: expected3)
  test(function4, input: input4, expected: expected4)
  #endif
}

func test<T, U: Equatable>(
  _ function: (Tensor<T>) -> Tensor<U>,
  input: T,
  expected: U
) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)
  XCTAssertEqual(transformedTensor.scalars, [U](repeating: expected, count: 5))
}

func test<T: FloatingPoint>(
  _ function: (Tensor<T>) -> Tensor<T>,
  input: T,
  expected: T,
  accuracy: T
) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)
  let scalars = transformedTensor.scalars
  XCTAssertEqual(scalars.count, 5)
  for scalar in scalars {
    XCTAssertEqual(scalar, expected, accuracy: accuracy)
  }
}

func testFloat(
  _ function: (Tensor<Float>) -> Tensor<Float>,
  _ swiftFunction: (Float) -> Float,
  input: Float,
  accuracy: Float = 0
) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)
  let scalars = transformedTensor.scalars
  XCTAssertEqual(scalars.count, 5)
  let expected = swiftFunction(input)
  for scalar in scalars {
    XCTAssertEqual(scalar, expected, accuracy: accuracy)
  }
}

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
func testFloat16(
  _ function: (Tensor<Float16>) -> Tensor<Float16>,
  _ swiftFunction: (Float) -> Float,
  input: Float16,
  accuracy: Float16 = 0
) {
  let tensor = Tensor(repeating: input, shape: [5])
  let transformedTensor = function(tensor)
  let scalars = transformedTensor.scalars
  XCTAssertEqual(scalars.count, 5)
  let expected = Float16(swiftFunction(Float(input)))
  for scalar in scalars {
    XCTAssertEqual(scalar, expected, accuracy: accuracy)
  }
}
#endif

fileprivate var profilingEncoding = false

func tensorOperationHeader(_ message: String? = nil) {
  Profiler.checkpoint()
  _ = defaultPluggableDevice
  let startupTime = Profiler.checkpoint()
  if startupTime > 1000 {
    print("=== Initialize pluggable device ===")
    print("Initialization time: \(startupTime) \(Profiler.timeUnit)")
  }
  
  if let message = message {
    print()
    print("=== \(message) ===")
  }
  
  // Stop messages about references from flooding the console. You can re-activate this inside a
  // test function if you want.
  defaultPluggableDevice.sync {
    Allocation.debugInfoEnabled = false
    profilingEncoding = MTLPluggableDevice.profilingEncoding
    MTLPluggableDevice.profilingEncoding = false
  }
  defaultPluggableDevice.barrier()
}

func tensorOperationFooter() {
  defaultPluggableDevice.sync {
    MTLPluggableDevice.profilingEncoding = profilingEncoding
  }
}

final class TensorUnaryOperationTests: XCTestCase {
  func testIncrement() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor<Float16>.incremented, input: 42, expected: 43)
    #endif
    test(Tensor<Float>.incremented, input: 42, expected: 43)
    test4(
      Tensor<Int8>.incremented, input1: 42, expected1: 43,
      Tensor<Int16>.incremented, input2: 42, expected2: 43,
      Tensor<Int32>.incremented, input3: 42, expected3: 43,
      Tensor<Int64>.incremented, input4: 42, expected4: 43)
    test4(
      Tensor<UInt8>.incremented, input1: 42, expected1: 43,
      Tensor<UInt16>.incremented, input2: 42, expected2: 43,
      Tensor<UInt32>.incremented, input3: 42, expected3: 43,
      Tensor<UInt64>.incremented, input4: 42, expected4: 43)
    
    // Test integer overflow.
    test4(
      Tensor<Int8>.incremented, input1: 127, expected1: -128,
      Tensor<Int16>.incremented, input2: .max, expected2: .min,
      Tensor<Int32>.incremented, input3: .max, expected3: .min,
      Tensor<Int64>.incremented, input4: .max, expected4: .min)
    test4(
      Tensor<UInt8>.incremented, input1: 255, expected1: 0,
      Tensor<UInt16>.incremented, input2: .max, expected2: .min,
      Tensor<UInt32>.incremented, input3: .max, expected3: .min,
      Tensor<UInt64>.incremented, input4: .max, expected4: .min)
  }
  
  // Integer overflow demonstrates the behavior described here:
  // https://stackoverflow.com/questions/35251410/c-safely-taking-absolute-value-of-integer
  func testAbs() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(abs, input: Float16(-42), expected: 42)
    #endif
    test(abs, input: Float(-42), expected: 42)
    test4(
      abs, input1: Int8(-127), expected1: 127,
      abs, input2: -Int16.max, expected2: Int16.max,
      abs, input3: -Int32.max, expected3: Int32.max,
      abs, input4: -Int64.max, expected4: Int64.max)
    
    // Test integer overflow.
    test4(
      abs, input1: Int8(-128), expected1: Int8(-128),
      abs, input2: Int16.min, expected2: Int16.min,
      abs, input3: Int32.min, expected3: Int32.min,
      abs, input4: Int64.min, expected4: Int64.min)
  }
  
  func testInverseTrigonometric() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    testFloat16(acos, acos, input: 0.42, accuracy: 0)
    testFloat16(acosh, acosh, input: 1.42, accuracy: 0)
    testFloat16(asin, asin, input: 0.42, accuracy: 0)
    testFloat16(asinh, asinh, input: 1.42, accuracy: 0)
    testFloat16(atan, atan, input: 0.42, accuracy: 0)
    testFloat16(atanh, atanh, input: 0.42, accuracy: 0)
    #endif
    testFloat(acos, acos, input: 0.42, accuracy: 1e-5)
    testFloat(acosh, acosh, input: 1.42, accuracy: 1e-5)
    testFloat(asin, asin, input: 0.42, accuracy: 1e-5)
    testFloat(asinh, asinh, input: 1.42, accuracy: 1e-5)
    testFloat(atan, atan, input: 0.42, accuracy: 1e-5)
    testFloat(atanh, atanh, input: 0.42, accuracy: 1e-5)
  }
  
  func testCast() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    // Same-type casting.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor.init, input: Float16(-42), expected: Float16(-42))
    #endif
    test(Tensor.init, input: Float(-42), expected: Float(-42))
    test(Tensor.unitTestCastBool, input: Bool(true), expected: Bool(true))
    test(Tensor.unitTestCastBool, input: Bool(false), expected: Bool(false))
    test4(
      Tensor.init, input1: Int8(-42), expected1: Int8(-42),
      Tensor.init, input2: Int16(-42), expected2: Int16(-42),
      Tensor.init, input3: Int32(-42), expected3: Int32(-42),
      Tensor.init, input4: Int64(-42), expected4: Int64(-42))
    test4(
      Tensor.init, input1: UInt8(42), expected1: UInt8(42),
      Tensor.init, input2: UInt16(42), expected2: UInt16(42),
      Tensor.init, input3: UInt32(42), expected3: UInt32(42),
      Tensor.init, input4: UInt64(42), expected4: UInt64(42))
    
    // Casting to/from boolean.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor.init, input: true, expected: Float16(1))
    test(Tensor.init, input: false, expected: Float16(0))
    test(Tensor.unitTestCastBool, input: Float16(0), expected: false)
    test(Tensor.unitTestCastBool, input: Float16(-1), expected: true)
    #endif
    test(Tensor.init, input: true, expected: Float(1))
    test(Tensor.init, input: false, expected: Float(0))
    test(Tensor.init, input: true, expected: UInt8(1))
    test(Tensor.init, input: false, expected: UInt8(0))
    test(Tensor.init, input: true, expected: Int32(1))
    test(Tensor.init, input: false, expected: Int32(0))
    test(Tensor.unitTestCastBool, input: Float(0), expected: false)
    test(Tensor.unitTestCastBool, input: Float(-1), expected: true)
    test(Tensor.unitTestCastBool, input: UInt8(0), expected: false)
    test(Tensor.unitTestCastBool, input: Int8(-1), expected: true)
    test(Tensor.unitTestCastBool, input: Int32(0), expected: false)
    test(Tensor.unitTestCastBool, input: Int32(-1), expected: true)
    test(Tensor.unitTestCastBool, input: Int64(0), expected: false)
    test(Tensor.unitTestCastBool, input: Int64(-1), expected: true)
    test(Tensor.unitTestCastBool, input: UInt64(1), expected: true)
    
    // Casting between floating-point types.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor.init, input: Float(0.5), expected: Float16(0.5))
    test(Tensor.init, input: Float16(0.5), expected: Float(0.5))
    test(Tensor.init, input: Float.infinity, expected: Float16.infinity)
    test(Tensor.init, input: Float16.infinity, expected: Float.infinity)
    test(Tensor.init, input: Float(65534), expected: Float16.infinity)
    #endif
    
    // Casting between floating-point types and integers.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor.init, input: Float16(7), expected: UInt8(7))
    test(Tensor.init, input: Float16(7), expected: Int32(7))
    test(Tensor.init, input: UInt8(7), expected: Float16(7))
    test(Tensor.init, input: Int32(7), expected: Float16(7))
    test(Tensor.init, input: UInt16(65534), expected: Float16.infinity)
    test(Tensor.init, input: Int64(70_000_000), expected: Float16.infinity)
    test(Tensor.init, input: UInt64(70_000_000), expected: Float16.infinity)
    test(Tensor.init, input: Float16.infinity, expected: UInt16.max)
    test(Tensor.init, input: -Float16.infinity, expected: UInt16.min)
    test(Tensor.init, input: -Float16.infinity, expected: Int16.min)
    #endif
    test(Tensor.init, input: Float(7), expected: UInt8(7))
    test(Tensor.init, input: Float(7), expected: Int32(7))
    test(Tensor.init, input: UInt8(7), expected: Float(7))
    test(Tensor.init, input: Int32(7), expected: Float(7))
    test(Tensor.init, input: UInt16(65534), expected: Float(65534))
    test(Tensor.init, input: Int64(70_000_000), expected: Float(70_000_000))
    test(Tensor.init, input: UInt64(70_000_000), expected: Float(70_000_000))
    test(Tensor.init, input: Float.infinity, expected: UInt16.max)
    test(Tensor.init, input: -Float.infinity, expected: UInt16.min)
    test(Tensor.init, input: -Float.infinity, expected: Int16.min)
    test(Tensor.init, input: Float.nan, expected: Int32(0))
    
    func testIntCombo<T: TensorFlowSignedNumeric & FixedWidthInteger, U: TensorFlowInteger>(
      _ type1: T.Type,
      _ type2: U.Type
    ) {
      // Performs multiple operations before waiting, reducing test execution time.
      #if true
      let tensor1 = Tensor<U>(Tensor(repeating: T.max, shape: [5]))
      let tensor2 = Tensor<U>(Tensor(repeating: T.max / 2, shape: [5]))
      let tensor3 = Tensor<U>(Tensor(repeating: -T.max, shape: [5]))
      let tensor4 = Tensor<U>(Tensor(repeating: -T.max / 2, shape: [5]))
      _ = tensor4.scalars
      
      let scalar1 = U(truncatingIfNeeded: T.max)
      let scalar2 = U(truncatingIfNeeded: T.max / 2)
      let scalar3 = U(truncatingIfNeeded: -T.max)
      let scalar4 = U(truncatingIfNeeded: -T.max / 2)
      XCTAssertEqual(tensor1.scalars, [U](repeating: scalar1, count: 5))
      XCTAssertEqual(tensor2.scalars, [U](repeating: scalar2, count: 5))
      XCTAssertEqual(tensor3.scalars, [U](repeating: scalar3, count: 5))
      XCTAssertEqual(tensor4.scalars, [U](repeating: scalar4, count: 5))
      #else
      test(Tensor.init, input: T.max, expected: U(truncatingIfNeeded: T.max))
      test(Tensor.init, input: T.max / 2, expected: U(truncatingIfNeeded: T.max / 2))
      test(Tensor.init, input: -T.max, expected: U(truncatingIfNeeded: -T.max))
      test(Tensor.init, input: -T.max / 2, expected: U(truncatingIfNeeded: -T.max / 2))
      #endif
    }
    
    // Casting between integral types (larger -> smaller).
    testIntCombo(Int16.self, Int8.self)
    test(Tensor.init, input: UInt16.max, expected: Int8(truncatingIfNeeded: UInt16.max))
    test(Tensor.init, input: UInt16.max / 2, expected: Int8(truncatingIfNeeded: UInt16.max / 2))
    test(Tensor.init, input: UInt16.max, expected: Int16(truncatingIfNeeded: UInt16.max))
    test(Tensor.init, input: UInt16.max / 2, expected: Int16(truncatingIfNeeded: UInt16.max / 2))
    testIntCombo(Int32.self, UInt8.self)
    testIntCombo(Int32.self, UInt16.self)
    
    // Casting between integral types (same size).
    testIntCombo(Int8.self, UInt8.self)
    testIntCombo(Int16.self, UInt16.self)
    testIntCombo(Int32.self, UInt32.self)
    testIntCombo(Int64.self, UInt64.self)
    
    // Casting between integral types (smaller -> larger).
    test(Tensor.init, input: UInt8.max, expected: UInt32(truncatingIfNeeded: UInt8.max))
    test(Tensor.init, input: UInt8.max / 2, expected: UInt32(truncatingIfNeeded: UInt8.max / 2))
    test(Tensor.init, input: UInt8.max, expected: Int64(truncatingIfNeeded: UInt8.max))
    test(Tensor.init, input: UInt8.max / 2, expected: Int64(truncatingIfNeeded: UInt8.max / 2))
    testIntCombo(Int8.self, Int16.self)
    testIntCombo(Int8.self, Int32.self)
    testIntCombo(Int8.self, UInt32.self)
    testIntCombo(Int8.self, Int64.self)
    testIntCombo(Int16.self, Int32.self)
    testIntCombo(Int32.self, UInt64.self)
    
    // Regression test for `Int16` -> `UInt64` conversion.
    testIntCombo(Int16.self, Int64.self)
    testIntCombo(Int16.self, UInt64.self)
    test(Tensor.init, input: UInt16.max, expected: Int64(truncatingIfNeeded: UInt16.max))
    test(Tensor.init, input: UInt16.max / 2, expected: Int64(truncatingIfNeeded: UInt16.max / 2))
    test(Tensor.init, input: UInt16.max, expected: UInt64(truncatingIfNeeded: UInt16.max))
    test(Tensor.init, input: UInt16.max / 2, expected: UInt64(truncatingIfNeeded: UInt16.max / 2))
    testIntCombo(Int8.self, Int64.self)
  }
  
  // Operations with codes 20 - 26
  func testOperations20Series() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    func swift_elu(_ x: Float) -> Float {
      if x < 0 {
        return exp(x) - 1
      } else {
        return x
      }
    }
    
    for input in [Float(-0.42), 0.42] {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let input_f16 = Float16(input)
      testFloat16(ceil, ceil, input: input_f16)
      testFloat16(cos, cos, input: input_f16)
      testFloat16(cosh, cosh, input: input_f16)
      testFloat16(elu, swift_elu, input: input_f16)
      testFloat16(exp, exp, input: input_f16)
      testFloat16(expm1, expm1, input: input_f16)
      testFloat16(floor, floor, input: input_f16)
      #endif
      testFloat(ceil, ceil, input: input)
      testFloat(cos, cos, input: input)
      testFloat(cosh, cosh, input: input, accuracy: 1e-5)
      testFloat(elu, swift_elu, input: input, accuracy: 1e-5)
      testFloat(exp, exp, input: input, accuracy: 1e-5)
      testFloat(expm1, expm1, input: input, accuracy: 1e-5)
      testFloat(floor, floor, input: input)
    }
  }
  
  func testRelational() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    func test<T: TensorFlowFloatingPoint>(_ type: T.Type) {
      let tensor1 = Tensor<T>(repeating: T(2), shape: [5])
      XCTAssertEqual(tensor1.isFinite.scalars, [Bool](repeating: true, count: 5))
      XCTAssertEqual(tensor1.isInfinite.scalars, [Bool](repeating: false, count: 5))
      XCTAssertEqual(tensor1.isNaN.scalars, [Bool](repeating: false, count: 5))
      
      let tensor2 = Tensor<T>(repeating: -T.infinity, shape: [5])
      XCTAssertEqual(tensor2.isFinite.scalars, [Bool](repeating: false, count: 5))
      XCTAssertEqual(tensor2.isInfinite.scalars, [Bool](repeating: true, count: 5))
      XCTAssertEqual(tensor2.isNaN.scalars, [Bool](repeating: false, count: 5))
      
      let tensor3 = Tensor<T>(repeating: T.signalingNaN, shape: [5])
      XCTAssertEqual(tensor3.isFinite.scalars, [Bool](repeating: false, count: 5))
      XCTAssertEqual(tensor3.isInfinite.scalars, [Bool](repeating: false, count: 5))
      XCTAssertEqual(tensor3.isNaN.scalars, [Bool](repeating: true, count: 5))
    }
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Float16.self)
    #endif
    test(Float.self)
  }
  
  // Operations with codes 40 - 48
  func testOperations40Series() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    func gpu_leakyReluWrapper<T: TensorFlowFloatingPoint>(
      alpha: Double
    ) -> (Tensor<T>) -> Tensor<T> {
      { leakyRelu($0, alpha: alpha) }
    }
    
    func swift_leakyReluWrapper(
      alpha: Double
    ) -> (Float) -> Float {
      { max($0, $0 * Float(alpha)) }
    }
    
    func swift_relu(_ x: Float) -> Float {
      max(x, 0)
    }
    func swift_relu6(_ x: Float) -> Float {
      min(max(x, 0), 6)
    }
    
    for input in [Float(-0.42), 0.42] {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let input_f16 = Float16(input)
      testFloat16(
        gpu_leakyReluWrapper(alpha: 0.2), swift_leakyReluWrapper(alpha: 0.2), input: input_f16)
      testFloat16(
        gpu_leakyReluWrapper(alpha: 0.7), swift_leakyReluWrapper(alpha: 0.7), input: input_f16)
      if input > 0 {
        testFloat16(log, log, input: input_f16)
      }
      testFloat16(log1p, log1p, input: input_f16)
      testFloat16(-, -, input: input_f16)
      testFloat16(relu, swift_relu, input: input_f16)
      testFloat16(relu6, swift_relu6, input: input_f16)
      testFloat16(round, rint, input: input_f16)
      #endif
      testFloat(gpu_leakyReluWrapper(alpha: 0.2), swift_leakyReluWrapper(alpha: 0.2), input: input)
      testFloat(gpu_leakyReluWrapper(alpha: 0.7), swift_leakyReluWrapper(alpha: 0.7), input: input)
      if input > 0 {
        testFloat(log, log, input: input, accuracy: 1e-5)
      }
      testFloat(log1p, log1p, input: input, accuracy: 1e-5)
      testFloat(-, -, input: input)
      testFloat(relu, swift_relu, input: input)
      testFloat(relu6, swift_relu6, input: input)
      testFloat(round, rint, input: input)
    }
    
    test(Tensor<Bool>.elementsLogicalNot, input: true, expected: false)
    test(Tensor<Bool>.elementsLogicalNot, input: false, expected: true)
    test(-, input: Int8(127), expected: Int8(-127))
    test(-, input: Int16.max, expected: -Int16.max)
    test(-, input: Int32.max, expected: -Int32.max)
    
    // Test integer overflow.
    test(-, input: Int8(-128), expected: Int8(-128))
    test(-, input: Int16.min, expected: Int16.min)
    test(-, input: Int32.min, expected: Int32.min)
    test(_Raw.neg, input: UInt8(0), expected: UInt8(0))
    test(_Raw.neg, input: UInt8(1), expected: UInt8(255))
    test(_Raw.neg, input: UInt16(65535), expected: UInt16(1))
    
    // The `round` operator in Swift does not round to nearest even.
    for input in [Float(-1.5), -0.5, 0.5, 1.5] {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      testFloat16(round, rint, input: Float16(input))
      #endif
      testFloat(round, rint, input: input)
    }
  }
  
  func testOperations50Series() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    // Avoiding the `simd_rsqrt` from Apple's "simd" library because that is not reproducible in the
    // OpenCL backend.
    func swift_rsqrt(_ x: Float) -> Float {
      1 / sqrt(x)
    }
    
    func swift_selu(_ x: Float) -> Float {
      let alpha: Float = 1.6732632423543772848170429916717
      let scale: Float = 1.0507009873554804934193349852946
      if x < 0 {
        return scale * alpha * (exp(x) - 1)
      } else {
        return scale * x
      }
    }
    
    func swift_sigmoid(_ x: Float) -> Float {
      1 / (1 + exp(-x))
    }
    
    func swift_sign(_ x: Float) -> Float {
      if x < 0 {
        return -1
      } else if x == 0 {
        return 0
      } else if x > 0 {
        return 1
      } else {
        fatalError("This should never happen.")
      }
    }
    
    func swift_softplus(_ x: Float) -> Float {
      log(exp(x) + 1)
    }
    
    // Inputs include zero to test `rsqrt` and `sign`.
    for input in [Float(-0.42), 0, 0.42] {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let input_f16 = Float16(input)
      if input >= 0 {
        testFloat16(rsqrt, swift_rsqrt, input: input_f16)
      }
      testFloat16(selu, swift_selu, input: input_f16)
      testFloat16(sigmoid, swift_sigmoid, input: input_f16)
      testFloat16(sign, swift_sign, input: input_f16)
      testFloat16(sin, sin, input: input_f16)
      testFloat16(sinh, sinh, input: input_f16)
      testFloat16(softplus, swift_softplus, input: input_f16)
      #endif
      if input >= 0 {
        testFloat(rsqrt, swift_rsqrt, input: input)
      }
      testFloat(selu, swift_selu, input: input, accuracy: 1e-5)
      testFloat(sigmoid, swift_sigmoid, input: input, accuracy: 1e-5)
      testFloat(sign, swift_sign, input: input)
      testFloat(sin, sin, input: input, accuracy: 1e-5)
      testFloat(sinh, sinh, input: input, accuracy: 1e-5)
      testFloat(softplus, swift_softplus, input: input, accuracy: 1e-5)
    }
    
    test(sign, input: Float(-0.0), expected: 0.0)
    test(sign, input: Float(-0.0), expected: -0.0)
    test4(
      sign, input1: Int8(-1), expected1: -1,
      sign, input2: Int16(-1), expected2: -1,
      sign, input3: Int32(-1), expected3: -1,
      sign, input4: Int64(-1), expected4: -1)
    
    test4(
      sign, input1: Int8(0), expected1: 0,
      sign, input2: Int16(0), expected2: 0,
      sign, input3: Int32(0), expected3: 0,
      sign, input4: Int64(0), expected4: 0)
    test4(
      sign, input1: UInt8(0), expected1: 0,
      sign, input2: UInt16(0), expected2: 0,
      sign, input3: UInt32(0), expected3: 0,
      sign, input4: UInt64(0), expected4: 0)
    
    test4(
      sign, input1: Int8(1), expected1: 1,
      sign, input2: Int16(1), expected2: 1,
      sign, input3: Int32(1), expected3: 1,
      sign, input4: Int64(1), expected4: 1)
    test4(
      sign, input1: UInt8(1), expected1: 1,
      sign, input2: UInt16(1), expected2: 1,
      sign, input3: UInt32(1), expected3: 1,
      sign, input4: UInt64(1), expected4: 1)
  }
  
  // Operations with codes 60 - 65
  func testOperations60Series() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    func swift_softsign(_ x: Float) -> Float {
      x / (abs(x) + 1)
    }
    
    func swift_square(_ x: Float) -> Float {
      x * x
    }
    
    for input in [Float(-0.42), 0.42] {
      #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
      let input_f16 = Float16(input)
      testFloat16(softsign, swift_softsign, input: input_f16)
      if input > 0 {
        testFloat16(sqrt, sqrt, input: input_f16)
      }
      testFloat16(square, swift_square, input: input_f16)
      testFloat16(tan, tan, input: input_f16)
      testFloat16(tanh, tanh, input: input_f16)
      #endif
      testFloat(softsign, swift_softsign, input: input)
      if input > 0 {
        testFloat(sqrt, sqrt, input: input)
      }
      testFloat(square, swift_square, input: input)
      testFloat(tan, tan, input: input, accuracy: 1e-5)
      testFloat(tanh, tanh, input: input)
    }
    
    test4(
      square, input1: Int8(-11), expected1: Int8(121),
      square, input2: Int16(-181), expected2: Int16(32761),
      square, input3: Int32(-46340), expected3: Int32(2_147_395_600),
      square, input4: Int64(-3_037_000_499), expected4: Int64(-3_037_000_499) * (-3_037_000_499))
    test4(
      square, input1: UInt8(15), expected1: UInt8(225),
      square, input2: UInt16(255), expected2: UInt16(255 * 255),
      square, input3: UInt32(65535), expected3: UInt32(65535 * 65535),
      square, input4: UInt64(4_294_967_295), expected4: UInt64(4_294_967_295) * (4_294_967_295))
    
    // Test integer overflow.
    test4(
      square, input1: Int8(16), expected1: Int8(0),
      square, input2: Int16(256), expected2: Int16(0),
      square, input3: Int32(65536), expected3: Int32(0),
      square, input4: Int64(1 << 32), expected4: Int64(0))
    test4(
      square, input1: UInt8(16), expected1: UInt8(0),
      square, input2: UInt16(256), expected2: UInt16(0),
      square, input3: UInt32(65536), expected3: UInt32(0),
      square, input4: UInt64(1 << 32), expected4: UInt64(0))
  }
}
