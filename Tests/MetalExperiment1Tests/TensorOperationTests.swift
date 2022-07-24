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
  Context.global.sync {
    _ = Context.global
  }
  let startupTime = Profiler.checkpoint()
  if startupTime > 1000 {
    print("=== Initialize context ===")
    print("Initialization time: \(startupTime) \(Profiler.timeUnit)")
  }
  
  if let message = message {
    print()
    print("=== \(message) ===")
  }
  
  // Stop messages about references from flooding the console. You can re-activate this inside a
  // test function if you want.
  Context.global.sync {
    Allocation.debugInfoEnabled = false
    profilingEncoding = Context.profilingEncoding
    Context.profilingEncoding = false
  }
  Context.barrier()
}

func tensorOperationFooter() {
  Context.global.sync {
    Context.profilingEncoding = profilingEncoding
  }
}

final class TensorOperationTests: XCTestCase {
  func testIncrement() throws {
    tensorOperationHeader()
    defer { tensorOperationFooter() }
    
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    test(Tensor<Float16>.incremented, input: 42, expected: 43)
    #endif
    test(Tensor<Float>.incremented, input: 42, expected: 43)
    test(Tensor<Int8>.incremented, input: 42, expected: 43)
    test(Tensor<Int16>.incremented, input: 42, expected: 43)
    test(Tensor<Int32>.incremented, input: 42, expected: 43)
    test(Tensor<UInt8>.incremented, input: 42, expected: 43)
    test(Tensor<UInt16>.incremented, input: 42, expected: 43)
    
    // Test integer overflow.
    test(Tensor<Int8>.incremented, input: 127, expected: -128)
    test(Tensor<Int16>.incremented, input: .max, expected: .min)
    test(Tensor<Int32>.incremented, input: .max, expected: .min)
    test(Tensor<UInt8>.incremented, input: 255, expected: 0)
    test(Tensor<UInt16>.incremented, input: .max, expected: .min)
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
    test(abs, input: Int8(-127), expected: 127)
    test(abs, input: -Int16.max, expected: Int16.max)
    
    // Test integer overflow.
    test(abs, input: Int8(-128), expected: Int8(-128))
    test(abs, input: Int16.min, expected: Int16.min)
    test(abs, input: Int32.min, expected: Int32.min)
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
    test(sign, input: Int8(-1), expected: -1)
    test(sign, input: Int16(-1), expected: -1)
    test(sign, input: Int32(-1), expected: -1)
    
    test(sign, input: Int8(0), expected: 0)
    test(sign, input: Int16(0), expected: 0)
    test(sign, input: Int32(0), expected: 0)
    test(sign, input: UInt8(0), expected: 0)
    test(sign, input: UInt16(0), expected: 0)
    
    test(sign, input: Int8(1), expected: 1)
    test(sign, input: Int16(1), expected: 1)
    test(sign, input: Int32(1), expected: 1)
    test(sign, input: UInt8(1), expected: 1)
    test(sign, input: UInt16(1), expected: 1)
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
    
    test(square, input: Int8(-11), expected: Int8(121))
    test(square, input: Int16(-181), expected: Int16(32761))
    test(square, input: Int32(-46340), expected: Int32(2_147_395_600))
    test(square, input: UInt8(15), expected: UInt8(225))
    test(square, input: UInt16(255), expected: UInt16(255 * 255))
    
    // Test integer overflow.
    test(square, input: Int8(16), expected: Int8(0))
    test(square, input: Int16(256), expected: Int16(0))
    test(square, input: Int32(65536), expected: Int32(0))
    test(square, input: UInt8(16), expected: UInt8(0))
    test(square, input: UInt16(256), expected: UInt16(0))
  }
}
