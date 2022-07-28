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

final class TensorBinaryOperationTests: XCTestCase {
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
      max, lhs3: SmallFloat(5), rhs3: 6, expected3: 6,
      min, lhs4: SmallFloat(5), rhs4: 6, expected4: 5)
    
    test4(
      max, lhs1: Int8(-6), rhs1: -5, expected1: -5,
      max, lhs2: Int16(-6), rhs2: -5, expected2: -5,
      max, lhs3: Int32(-6), rhs3: -5, expected3: -5,
      max, lhs4: Int64(-6), rhs4: -5, expected4: -5)
    test4(
      max, lhs1: Int8(-6), rhs1: 5, expected1: 5,
      max, lhs2: Int16(-6), rhs2: 5, expected2: 5,
      max, lhs3: Int32(-6), rhs3: 5, expected3: 5,
      max, lhs4: Int64(-6), rhs4: 5, expected4: 5)
    test4(
      min, lhs1: Int8(-6), rhs1: -5, expected1: -6,
      min, lhs2: Int16(-6), rhs2: -5, expected2: -6,
      min, lhs3: Int32(-6), rhs3: -5, expected3: -6,
      min, lhs4: Int64(-6), rhs4: -5, expected4: -6)
    test4(
      min, lhs1: Int8(-6), rhs1: 5, expected1: -6,
      min, lhs2: Int16(-6), rhs2: 5, expected2: -6,
      min, lhs3: Int32(-6), rhs3: 5, expected3: -6,
      min, lhs4: Int64(-6), rhs4: 5, expected4: -6)
    
    test4(
      min, lhs1: UInt8(5), rhs1: 6, expected1: 5,
      min, lhs2: UInt16(5), rhs2: 6, expected2: 5,
      min, lhs3: UInt32(5), rhs3: 6, expected3: 5,
      min, lhs4: UInt64(5), rhs4: 6, expected4: 5)
  }
}
