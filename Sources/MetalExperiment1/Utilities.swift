//
//  Utilities.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Metal

enum Profiler {
  // Microseconds.
  static let timeUnit = "\u{b5}s"
  
  private static var time: UInt64?
  
  @discardableResult
  static func checkpoint() -> Int {
    let lastTime = time
    let currentTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    time = currentTime
    if let lastTime = lastTime {
      return Int(currentTime - lastTime) / 1000
    } else {
      return 0
    }
  }
  
  static func log(_ message: String? = nil) {
    let duration = checkpoint()
    if let message = message {
      print("\(message): \(duration) \(timeUnit)")
    } else {
      print("\(duration) \(timeUnit)")
    }
  }
  
  @discardableResult
  static func withLogging<T>(
    _ message: String? = nil,
    _ body: () throws -> T
  ) rethrows -> T {
    checkpoint()
    let output = try body()
    let duration = checkpoint()
    if let message = message {
      print("\(message): \(duration) \(timeUnit)")
    } else {
      print("\(duration) \(timeUnit)")
    }
    return output
  }
}

// From https://github.com/philipturner/ARHeadsetUtil
extension MTLSize: ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
  @inline(__always)
  init(_ value: Int) {
    self = MTLSizeMake(value, 1, 1)
  }
  
  @inline(__always)
  public /*internal*/ init(integerLiteral value: Int) {
    self = MTLSizeMake(value, 1, 1)
  }
  
  @inline(__always)
  public /*internal*/ init(arrayLiteral elements: Int...) {
    switch elements.count {
    case 1:  self = MTLSizeMake(elements[0], 1, 1)
    case 2:  self = MTLSizeMake(elements[0], elements[1], 1)
    case 3:  self = MTLSizeMake(elements[0], elements[1], elements[2])
    default: fatalError("A MTLSize must not exceed three dimensions!")
    }
  }
}

extension MTLCommandBuffer {
  @inline(never)
  var errorMessage: String {
    guard let error = error as NSError? else {
      fatalError("Tried to retrieve 'MTLCommandBuffer' error message when there was no error")
    }
    
    var output: [String] = []
    for log in logs {
      output.append(log.description)
      
      let encoderLabel = log.encoderLabel ?? "Unknown label"
      output.append("Faulting encoder: \"\(encoderLabel)\"")
      
      guard let debugLocation = log.debugLocation,
            let functionName = debugLocation.functionName else {
        fatalError("'MTLCommandBuffer' log should have debug info")
      }
      output.append("""
        Faulting function: \(functionName) (line \(debugLocation.line), column \
        \(debugLocation.column))"
        """)
    }
    
    switch status {
    case .notEnqueued: output.append("Status: not enqueued")
    case .enqueued:    output.append("Status: enqueued")
    case .committed:   output.append("Status: committed")
    case .scheduled:   output.append("Status: scheduled")
    case .completed:   output.append("Status: completed")
    case .error:       output.append("Status: error")
    @unknown default: fatalError("This status is not possible!")
    }
    
    output.append("Error code: \(error.code)")
    output.append("Description: \(error.localizedDescription)")
    if let reason = error.localizedFailureReason {
      output.append("Failure reason: \(reason)")
    }
    if let options = error.localizedRecoveryOptions {
      for i in 0..<options.count {
        output.append("Recovery option \(i): \(options[i])")
      }
    }
    if let suggestion = error.localizedRecoverySuggestion {
      output.append("Recovery suggestion: \(suggestion)")
    }
    
    return output.joined(separator: "\n")
  }
}

@inline(__always)
func withUnsafeAddress<T: AnyObject, U>(
  of object: T,
  _ body: (UnsafeMutableRawPointer) throws -> U
) rethrows -> U {
  let ptr = Unmanaged<T>.passUnretained(object).toOpaque()
  return try body(ptr)
}

func fetchEnvironmentBoolean(_ name: String) -> Bool {
  if let value = getenv(name) {
    let string = String(cString: value)
    return Int(string) != 0
  }
  return false
}

// MARK: - OperationTypeList

// Similar to the `SmallVector<_, _>` C++ type in LLVM.
protocol OperationTypeList {
  associatedtype Element: CaseIterable & RawRepresentable
  where Element.RawValue: FixedWidthInteger & SIMDScalar
  
  associatedtype Vector: SIMD where Vector.Scalar == Element.RawValue
  var storage: OperationTypeListStorage<Vector> { get set }
}

struct OperationTypeListStorage<Vector: SIMD>
where Vector.Scalar: FixedWidthInteger & SIMDScalar {
  private var vector: Vector
  private(set) fileprivate var count: Int
  private var array: [Vector.Scalar]?
  
  @inline(__always)
  fileprivate init() {
    vector = .zero
    count = 0
    array = nil
  }
  
  @inline(__always)
  fileprivate mutating func append(_ newElement: Vector.Scalar) {
    if count < Vector.scalarCount {
      vector[count] = newElement
    } else {
      if _slowPath(array == nil) {
        array = Array(unsafeUninitializedCapacity: Vector.scalarCount &+ 1) { bufferPointer, count in
          count = self.count
          bufferPointer.withMemoryRebound(to: Vector.self) { ptr in
            ptr[0] = vector
          }
        }
      }
      array!.append(newElement)
    }
    count &+= 1
  }
  
  @inline(__always)
  fileprivate subscript(index: Int) -> Vector.Scalar {
    if count <= Vector.scalarCount {
      return vector[index]
    } else {
      return array.unsafelyUnwrapped[index]
    }
  }
}

extension OperationTypeList {
  @inline(__always)
  mutating func append(_ newElement: Element) {
    storage.append(newElement.rawValue)
  }
  
  @inline(__always)
  var count: Int {
    storage.count
  }
  
  @inline(__always)
  subscript(index: Int) -> Element {
    Element(rawValue: storage[index]).unsafelyUnwrapped
  }
}

// MARK: - OperationTypeList Implementations

struct OperationTypeList2<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD2<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

struct OperationTypeList4<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD4<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

struct OperationTypeList8<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD8<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

struct OperationTypeList16<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD16<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

struct OperationTypeList32<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD32<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

struct OperationTypeList64<Element: CaseIterable & RawRepresentable>: OperationTypeList
where Element.RawValue: FixedWidthInteger & SIMDScalar {
  typealias Vector = SIMD64<Element.RawValue>
  var storage: OperationTypeListStorage<Vector> = .init()
}

// MARK: - StringWrapper

// From the Swift standard library's conformance of `String` to `Hashable`.
struct StringWrapper: Hashable {
  var ptr: UnsafeRawBufferPointer
  
  @inline(__always)
  init(wrapping wrappedValue: UnsafeRawBufferPointer) {
    self.ptr = wrappedValue
  }
  
  @inline(__always)
  init(_ string: StaticString) {
    let start = string.utf8Start
    let count = string.utf8CodeUnitCount
    self.ptr = UnsafeRawBufferPointer(start: start, count: count)
  }
  
  @inline(__always)
  static func == (lhs: StringWrapper, rhs: StringWrapper) -> Bool {
    lhs.ptr.elementsEqual(rhs.ptr)
  }
  
  @inline(__always)
  func hash(into hasher: inout Hasher) {
    hasher.combine(bytes: ptr)
    hasher.combine(0xFF as UInt8) // terminator
  }
}

extension StringWrapper: ExpressibleByStringLiteral {
  @inline(__always)
  init(stringLiteral value: StaticString) {
    self.init(value)
  }
}
