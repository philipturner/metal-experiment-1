import XCTest
@testable import MetalExperiment1

func test<T: TensorFlowScalar, U: Equatable>(
  _ function: (Tensor<T>) -> (Tensor<T>) -> Tensor<U>,
  lhs: T,
  rhs: T,
  expected: U
) {
  let lhsTensor = Tensor(repeating: lhs, shape: [5])
  let rhsTensor = Tensor(repeating: rhs, shape: [5])
  let transformedTensor = function(lhsTensor)(rhsTensor)
  XCTAssertEqual(transformedTensor.scalars, [U](repeating: expected, count: 5))
}

func test<T: TensorFlowScalar, U: Equatable>(
  _ function: (Tensor<T>, Tensor<T>) -> Tensor<U>,
  lhs: T,
  rhs: T,
  expected: U
) {
  let lhsTensor = Tensor(repeating: lhs, shape: [5])
  let rhsTensor = Tensor(repeating: rhs, shape: [5])
  let transformedTensor = function(lhsTensor, rhsTensor)
  XCTAssertEqual(transformedTensor.scalars, [U](repeating: expected, count: 5))
}

func test4<
  T0: TensorFlowScalar,
  T1: TensorFlowScalar,
  T2: TensorFlowScalar,
  T3: TensorFlowScalar,
  U0: Equatable,
  U1: Equatable,
  U2: Equatable,
  U3: Equatable
>(
  _ f1: (Tensor<T0>) -> (Tensor<T0>) -> Tensor<U0>, lhs1: T0, rhs1: T0, expected1: U0,
  _ f2: (Tensor<T1>) -> (Tensor<T1>) -> Tensor<U1>, lhs2: T1, rhs2: T1, expected2: U1,
  _ f3: (Tensor<T2>) -> (Tensor<T2>) -> Tensor<U2>, lhs3: T2, rhs3: T2, expected3: U2,
  _ f4: (Tensor<T3>) -> (Tensor<T3>) -> Tensor<U3>, lhs4: T3, rhs4: T3, expected4: U3
) {
  // Performs multiple operations before waiting, reducing test execution time.
  #if true
  let tensor1 = f1(Tensor(repeating: lhs1, shape: [5]))(Tensor(repeating: rhs1, shape: [5]))
  let tensor2 = f2(Tensor(repeating: lhs2, shape: [5]))(Tensor(repeating: rhs2, shape: [5]))
  let tensor3 = f3(Tensor(repeating: lhs3, shape: [5]))(Tensor(repeating: rhs3, shape: [5]))
  let tensor4 = f4(Tensor(repeating: lhs4, shape: [5]))(Tensor(repeating: rhs4, shape: [5]))
  _ = tensor4.scalars
  
  XCTAssertEqual(tensor1.scalars, [U0](repeating: expected1, count: 5))
  XCTAssertEqual(tensor2.scalars, [U1](repeating: expected2, count: 5))
  XCTAssertEqual(tensor3.scalars, [U2](repeating: expected3, count: 5))
  XCTAssertEqual(tensor4.scalars, [U3](repeating: expected4, count: 5))
  #else
  test(f1, lhs: lhs1, rhs: rhs1, expected: expected1)
  test(f2, lhs: lhs2, rhs: rhs2, expected: expected2)
  test(f3, lhs: lhs3, rhs: rhs3, expected: expected3)
  test(f4, lhs: lhs4, rhs: rhs4, expected: expected4)
  #endif
}

func test4<
  T0: TensorFlowScalar,
  T1: TensorFlowScalar,
  T2: TensorFlowScalar,
  T3: TensorFlowScalar,
  U0: Equatable,
  U1: Equatable,
  U2: Equatable,
  U3: Equatable
>(
  _ f1: (Tensor<T0>, Tensor<T0>) -> Tensor<U0>, lhs1: T0, rhs1: T0, expected1: U0,
  _ f2: (Tensor<T1>, Tensor<T1>) -> Tensor<U1>, lhs2: T1, rhs2: T1, expected2: U1,
  _ f3: (Tensor<T2>, Tensor<T2>) -> Tensor<U2>, lhs3: T2, rhs3: T2, expected3: U2,
  _ f4: (Tensor<T3>, Tensor<T3>) -> Tensor<U3>, lhs4: T3, rhs4: T3, expected4: U3
) {
  // Performs multiple operations before waiting, reducing test execution time.
  #if true
  let tensor1 = f1(Tensor(repeating: lhs1, shape: [5]), Tensor(repeating: rhs1, shape: [5]))
  let tensor2 = f2(Tensor(repeating: lhs2, shape: [5]), Tensor(repeating: rhs2, shape: [5]))
  let tensor3 = f3(Tensor(repeating: lhs3, shape: [5]), Tensor(repeating: rhs3, shape: [5]))
  let tensor4 = f4(Tensor(repeating: lhs4, shape: [5]), Tensor(repeating: rhs4, shape: [5]))
  _ = tensor4.scalars
  
  XCTAssertEqual(tensor1.scalars, [U0](repeating: expected1, count: 5))
  XCTAssertEqual(tensor2.scalars, [U1](repeating: expected2, count: 5))
  XCTAssertEqual(tensor3.scalars, [U2](repeating: expected3, count: 5))
  XCTAssertEqual(tensor4.scalars, [U3](repeating: expected4, count: 5))
  #else
  test(f1, lhs: lhs1, rhs: rhs1, expected: expected1)
  test(f2, lhs: lhs2, rhs: rhs2, expected: expected2)
  test(f3, lhs: lhs3, rhs: rhs3, expected: expected3)
  test(f4, lhs: lhs4, rhs: rhs4, expected: expected4)
  #endif
}

internal func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: [T], _ y: [T], accuracy: T, _ message: String = "",
  file: StaticString = #file, line: UInt = #line
) {
  for (x, y) in zip(x, y) {
    if x.isNaN || y.isNaN {
      XCTAssertTrue(
        x.isNaN && y.isNaN,
        "\(x) is not equal to \(y) - \(message)",
        file: file, line: line)
      continue
    }
    XCTAssertEqual(x, y, accuracy: accuracy, message, file: file, line: line)
  }
}

internal func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>, _ y: Tensor<T>, accuracy: T, _ message: String = "",
  file: StaticString = #file, line: UInt = #line
) {
  assertEqual(x.scalars, y.scalars, accuracy: accuracy, message, file: file, line: line)
}

final class TensorBinaryOperationTests: XCTestCase {
  // Add, Sub, Mul, Div
  func testScalarizedOperations() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    _ = Tensor<Float>.zero
    _ = Tensor<Int8>.zero
    _ = Tensor<UInt8>.zero
    _ = Tensor<Int64>.zero
    _ = Tensor<UInt64>.zero
    
    // Can't perform CPU-side division inside the function, so pass in `ratio`.
    func nestedTest<T: TensorFlowNumeric>(small: T, large: T, ratio: T?) {
      let smallTensor = Tensor<T>(repeating: small, shape: [5])
      let largeTensor = Tensor<T>(repeating: large, shape: [5])
      let addTest1 = smallTensor + largeTensor
      let addTest2 = smallTensor + large
      let addTest3 = small + largeTensor
      let addTestArray = [T](repeating: small + large, count: 5)
      XCTAssertEqual(addTest1.scalars, addTestArray)
      XCTAssertEqual(addTest2.scalars, addTestArray)
      XCTAssertEqual(addTest3.scalars, addTestArray)
      
      let subTest1 = largeTensor - smallTensor
      let subTest2 = largeTensor - small
      if T.self is any SignedNumeric {
        let subInvTest1 = smallTensor - largeTensor
        let subInvTest2 = smallTensor - large
        let subInvTestArray = [T](repeating: small - large, count: 5)
        XCTAssertEqual(subInvTest1.scalars, subInvTestArray)
        XCTAssertEqual(subInvTest2.scalars, subInvTestArray)
      }
      let subTestArray = [T](repeating: large - small, count: 5)
      XCTAssertEqual(subTest1.scalars, subTestArray)
      XCTAssertEqual(subTest2.scalars, subTestArray)
      
      let mulTest1 = smallTensor * largeTensor
      let mulTest2 = smallTensor * large
      let mulTest3 = small * largeTensor
      let mulTestArray = [T](repeating: small * large, count: 5)
      XCTAssertEqual(mulTest1.scalars, mulTestArray)
      XCTAssertEqual(mulTest2.scalars, mulTestArray)
      XCTAssertEqual(mulTest3.scalars, mulTestArray)
      
      if let ratio = ratio {
        let divTest1 = smallTensor / largeTensor
        let divTest2 = smallTensor / large
        let divTest3 = small / largeTensor
        let divTestArray = [T](repeating: ratio, count: 5)
        XCTAssertEqual(divTest1.scalars, divTestArray)
        XCTAssertEqual(divTest2.scalars, divTestArray)
        XCTAssertEqual(divTest3.scalars, divTestArray)
      } else {
        // Avoid dividing by zero.
      }
    }
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    nestedTest(small: Float16(5), large: 7, ratio: 5 / 7)
    #endif
    nestedTest(small: Float(5), large: 7, ratio: 5 / 7)
    nestedTest(small: Int8(5), large: 7, ratio: 5 / 7)
    nestedTest(small: UInt8(5), large: 7, ratio: 5 / 7)
    nestedTest(small: Int64(5), large: 7, ratio: 5 / 7)
    nestedTest(small: UInt64(5), large: 7, ratio: 5 / 7)
    
    // Test no-ops
    nestedTest(small: Int32(-5), large: 0, ratio: nil)
    nestedTest(small: Int32(0), large: -5, ratio: 0 / -5)
    nestedTest(small: Int32(-5), large: 1, ratio: -5 / 1)
    nestedTest(small: Int32(1), large: -5, ratio: 1 / -5)
  }
  
  func testComparison() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    typealias SmallFloat = Float16
    #else
    typealias SmallFloat = Float
    #endif
    
    func almostEqual<T: TensorFlowFloatingPoint>(
      _ tolerance: T
    ) -> (Tensor<T>, Tensor<T>) -> Tensor<Bool> {
      { $0.elementsAlmostEqual($1, tolerance: tolerance) }
    }
    
    test4(
      almostEqual(0.3), lhs1: SmallFloat(0.1), rhs1: 0.5, expected1: false,
      almostEqual(0.7), lhs2: SmallFloat(0.1), rhs2: 0.5, expected2: true,
      almostEqual(0.3), lhs3: Float(0.1), rhs3: 0.5, expected3: false,
      almostEqual(0.7), lhs4: Float(0.1), rhs4: 0.5, expected4: true)
    
    // Regression test: comparison not working correctly with booleans.
    test4(
      .==, lhs1: true, rhs1: true, expected1: true,
      .==, lhs2: true, rhs2: false, expected2: false,
      .==, lhs3: false, rhs3: false, expected3: true,
      .==, lhs4: false, rhs4: true, expected4: false)
    test4(
      .!=, lhs1: true, rhs1: true, expected1: false,
      .!=, lhs2: true, rhs2: false, expected2: true,
      .!=, lhs3: false, rhs3: false, expected3: false,
      .!=, lhs4: false, rhs4: true, expected4: true)
    
    test4(
      .==, lhs1: Float(8), rhs1: Float(8), expected1: true,
      .==, lhs2: Int8(8), rhs2: Int8(8), expected2: true,
      .==, lhs3: UInt32(8), rhs3: UInt32(8), expected3: true,
      .==, lhs4: Int64(8), rhs4: Int64(8), expected4: true)
    test4(
      .!=, lhs1: Float(8), rhs1: Float(6), expected1: true,
      .!=, lhs2: Int8(8), rhs2: Int8(6), expected2: true,
      .!=, lhs3: UInt32(8), rhs3: UInt32(6), expected3: true,
      .!=, lhs4: Int64(8), rhs4: Int64(6), expected4: true)
    
    test4(
      .<, lhs1: Int8(-6), rhs1: Int8(-8), expected1: false,
      .>, lhs2: Int8(-6), rhs2: Int8(-8), expected2: true,
      .<=, lhs3: Int8(-6), rhs3: Int8(-8), expected3: false,
      .>=, lhs4: Int8(-6), rhs4: Int8(-8), expected4: true)
    test4(
      .<, lhs1: UInt32(8), rhs1: UInt32(6), expected1: false,
      .>, lhs2: UInt32(8), rhs2: UInt32(6), expected2: true,
      .<=, lhs3: UInt32(8), rhs3: UInt32(6), expected3: false,
      .>=, lhs4: UInt32(8), rhs4: UInt32(6), expected4: true)
    test4(
      .<, lhs1: Int64(-6), rhs1: Int64(-8), expected1: false,
      .>, lhs2: Int64(-6), rhs2: Int64(-8), expected2: true,
      .<=, lhs3: Int64(-6), rhs3: Int64(-8), expected3: false,
      .>=, lhs4: Int64(-6), rhs4: Int64(-8), expected4: true)
    test4(
      .<=, lhs1: Int8(-6), rhs1: Int8(-6), expected1: true,
      .>=, lhs2: UInt16(6), rhs2: UInt16(6), expected2: true,
      .<=, lhs3: Int64(-6), rhs3: Int64(-6), expected3: true,
      .>=, lhs4: UInt64(6), rhs4: UInt64(6), expected4: true)
    
    test4(
      .==, lhs1: SmallFloat.nan, rhs1: SmallFloat.nan, expected1: false,
      .==, lhs2: Float.nan, rhs2: Float.nan, expected2: false,
      .!=, lhs3: SmallFloat.nan, rhs3: SmallFloat.nan, expected3: true,
      .!=, lhs4: Float.nan, rhs4: Float.nan, expected4: true)
    test4(
      .<, lhs1: Int8.min, rhs1: Int8.max, expected1: true,
      .>, lhs2: UInt16.max, rhs2: UInt16.min, expected2: true,
      .<, lhs3: UInt32.min, rhs3: UInt32.max, expected3: true,
      .>, lhs4: Int64.max, rhs4: Int64.min, expected4: true)
  }
  
  func testGradientOperations() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    // func _vjpRelu6
    
    // func _vjpRelu
    
    // func _vjpRsqrt
    
    // func _vjpSelu
    
    // func _vjpSigmoid
    
    // func _vjpSoftplus
    
    // func _vjpSoftsign
    
    // RsqrtGrad
    do {
      let x = Tensor<Float>([1, 0.25, Float(1.0) / 9.0, 0.0625, 0.04])
      let target = Tensor<Float>([1, 2, 3, 4, 5])
      let gradTarget = Tensor<Float>([-0.5, -4.0, -13.5, -32.0, -62.5])
      let (value, pullback) = _vjpRsqrt(x)
      let grad = pullback(Tensor(repeating: 1, shape: [5]))
      XCTAssertEqual(value.scalars, target.scalars)
      XCTAssertEqual(grad.scalars, gradTarget.scalars)
    }
    
    // Sigmoid
    do {
      let x = Tensor<Float>([-1, 0, 1])
      let gradTarget = Tensor<Float>([0.1966119, 0.25, 0.1966119])
      let (_, pullback) = _vjpSigmoid(x)
      let grad = pullback(Tensor(repeating: 1, shape: [3]))
      assertEqual(grad, gradTarget, accuracy: 0.0001)
    }
  }
  
  func testMinMax() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    typealias SmallFloat = Float16
    #else
    typealias SmallFloat = Float
    #endif
    
    test4(
      max, lhs1: Float(5), rhs1: 6, expected1: 6,
      min, lhs2: Float(5), rhs2: 6, expected2: 5,
      max, lhs3: SmallFloat(6), rhs3: 5, expected3: 6,
      min, lhs4: SmallFloat(6), rhs4: 5, expected4: 5)
    
    test4(
      max, lhs1: Int8(-6), rhs1: -5, expected1: -5,
      max, lhs2: Int16(-6), rhs2: -5, expected2: -5,
      max, lhs3: Int32(-6), rhs3: -5, expected3: -5,
      max, lhs4: Int64(-6), rhs4: -5, expected4: -5)
    test4(
      max, lhs1: Int8(-6), rhs1: 5, expected1: 5,
      max, lhs2: Int16(-6), rhs2: 5, expected2: 5,
      max, lhs3: Int32(5), rhs3: -6, expected3: 5,
      max, lhs4: Int64(5), rhs4: -6, expected4: 5)
    test4(
      min, lhs1: Int8(-6), rhs1: -5, expected1: -6,
      min, lhs2: Int16(-6), rhs2: -5, expected2: -6,
      min, lhs3: Int32(-6), rhs3: -5, expected3: -6,
      min, lhs4: Int64(-6), rhs4: -5, expected4: -6)
    test4(
      min, lhs1: Int8(-6), rhs1: 5, expected1: -6,
      min, lhs2: Int16(-6), rhs2: 5, expected2: -6,
      min, lhs3: Int32(5), rhs3: -6, expected3: -6,
      min, lhs4: Int64(5), rhs4: -6, expected4: -6)
    
    test4(
      max, lhs1: UInt8(6), rhs1: 5, expected1: 6,
      max, lhs2: UInt16(5), rhs2: 6, expected2: 6,
      max, lhs3: UInt32(6), rhs3: 5, expected3: 6,
      max, lhs4: UInt64(5), rhs4: 6, expected4: 6)
    test4(
      min, lhs1: UInt8(6), rhs1: 5, expected1: 5,
      min, lhs2: UInt16(5), rhs2: 6, expected2: 5,
      min, lhs3: UInt32(6), rhs3: 5, expected3: 5,
      min, lhs4: UInt64(5), rhs4: 6, expected4: 5)
  }
  
  // Mod, Pow, SquaredDifference, Xdivy
  func testOther() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    typealias SmallFloat = Float16
    #else
    typealias SmallFloat = Float
    #endif
    
    func swift_mod<T: FloatingPoint>(_ x: T, _ y: T) -> T {
      x.truncatingRemainder(dividingBy: y)
    }
    
    func swift_pow<T: BinaryFloatingPoint>(_ x: T, _ y: T) -> T {
      T(Darwin.pow(Float(x), Float(y)))
    }
    
    func swift_sqr_diff_f32<T: BinaryFloatingPoint>(_ x: T, _ y: T) -> T {
      (x - y) * (x - y)
    }
    
    func swift_sqr_diff_i32<T: FixedWidthInteger>(_ x: T, _ y: T) -> T {
      let absdiff = (x >= y) ? (x - y) : (y - x)
      return absdiff * absdiff
    }
    
    func swift_xdivy<T: BinaryFloatingPoint>(_ x: T, _ y: T) -> T {
      if x == 0 {
        return 0
      } else {
        return x / y
      }
    }
    
    // Build times are getting excessively long because of exponential complexity of resolving stray
    // operators. Instead, wrap the operator `%` in another function.
    func tensor_mod<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
      x % y
    }
    
    func swift_integer_mod<T: BinaryInteger>(_ x: T, _ y: T) -> T {
      x % y
    }
    
    test4(
      tensor_mod, lhs1: Float(5), rhs1: -6, expected1: swift_mod(5, -6),
      tensor_mod, lhs2: Float(-6), rhs2: -5, expected2: swift_mod(-6, -5),
      tensor_mod, lhs3: SmallFloat(5), rhs3: -6, expected3: swift_mod(5, -6),
      tensor_mod, lhs4: SmallFloat(-6), rhs4: -5, expected4: swift_mod(-6, -5))
    test4(
      tensor_mod, lhs1: Int8(5), rhs1: -6, expected1: swift_integer_mod(5, -6),
      tensor_mod, lhs2: Int16(-6), rhs2: -5, expected2: swift_integer_mod(-6, -5),
      tensor_mod, lhs3: Int32(5), rhs3: -6, expected3: swift_integer_mod(5, -6),
      tensor_mod, lhs4: Int64(-6), rhs4: -5, expected4: swift_integer_mod(-6, -5))
    test4(
      tensor_mod, lhs1: UInt8(5), rhs1: 6, expected1: swift_integer_mod(5, 6),
      tensor_mod, lhs2: UInt16(6), rhs2: 5, expected2: swift_integer_mod(6, 5),
      tensor_mod, lhs3: UInt32(5), rhs3: 6, expected3: swift_integer_mod(5, 6),
      tensor_mod, lhs4: UInt64(6), rhs4: 5, expected4: swift_integer_mod(6, 5))
    
    func testPow<T: TensorFlowFloatingPoint>(_ type: T.Type, accuracy: T) {
      let lhs1 = Tensor<T>(repeating: 5, shape: [5])
      let rhs1 = Tensor<T>(repeating: -6, shape: [5])
      let lhs2 = Tensor<T>(repeating: -6, shape: [5])
      let rhs2 = Tensor<T>(repeating: -5, shape: [5])
      let tensor1 = pow(lhs1, rhs1)
      let tensor2 = pow(lhs2, rhs2)
      let scalars1 = tensor1.scalars
      let scalars2 = tensor2.scalars
      
      let expected1 = swift_pow(T(5), T(-6))
      let expected2 = swift_pow(T(-6), T(-5))
      for (scalars, expected) in [(scalars1, expected1), (scalars2, expected2)] {
        for scalar in scalars {
          XCTAssertEqual(scalar, expected, accuracy: accuracy)
        }
      }
    }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    let smallFloatAccuracy = Float16(0)
    #else
    let smallFloatAccuracy = Float(1e-5)
    #endif
    testPow(SmallFloat.self, accuracy: smallFloatAccuracy)
    testPow(Float.self, accuracy: 1e-5)
    
    test4(
      squaredDifference, lhs1: Float(5), rhs1: -6, expected1: swift_sqr_diff_f32(5, -6),
      squaredDifference, lhs2: Float(-6), rhs2: -5, expected2: swift_sqr_diff_f32(-6, -5),
      squaredDifference, lhs3: SmallFloat(5), rhs3: -6, expected3: swift_sqr_diff_f32(5, -6),
      squaredDifference, lhs4: SmallFloat(-6), rhs4: -5, expected4: swift_sqr_diff_f32(-6, -5))
    test4(
      squaredDifference, lhs1: Int8(5), rhs1: -6, expected1: swift_sqr_diff_i32(5, -6),
      squaredDifference, lhs2: Int16(-6), rhs2: -5, expected2: swift_sqr_diff_i32(-6, -5),
      squaredDifference, lhs3: Int32(5), rhs3: -6, expected3: swift_sqr_diff_i32(5, -6),
      squaredDifference, lhs4: Int64(-6), rhs4: -5, expected4: swift_sqr_diff_i32(-6, -5))
    test4(
      squaredDifference, lhs1: UInt8(5), rhs1: 6, expected1: swift_sqr_diff_i32(5, 6),
      squaredDifference, lhs2: UInt16(6), rhs2: 5, expected2: swift_sqr_diff_i32(6, 5),
      squaredDifference, lhs3: UInt32(5), rhs3: 6, expected3: swift_sqr_diff_i32(5, 6),
      squaredDifference, lhs4: UInt64(6), rhs4: 5, expected4: swift_sqr_diff_i32(6, 5))
    
    test4(
      _Raw.xdivy, lhs1: Float(1), rhs1: 1, expected1: swift_xdivy(1, 1),
      _Raw.xdivy, lhs2: Float(0), rhs2: 0, expected2: swift_xdivy(0, 0),
      _Raw.xdivy, lhs3: SmallFloat(0), rhs3: -6, expected3: swift_xdivy(0, -6),
      _Raw.xdivy, lhs4: SmallFloat(0), rhs4: 0, expected4: swift_xdivy(0, 0))
  }
}
