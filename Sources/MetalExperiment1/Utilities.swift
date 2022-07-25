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
      fatalError("Tried to retrieve 'MTLCommandBuffer' error message when there was no error.")
    }
    
    var output: [String] = []
    for log in logs {
      output.append(log.description)
      
      let encoderLabel = log.encoderLabel ?? "Unknown label"
      output.append("Faulting encoder: \"\(encoderLabel)\"")
      
      guard let debugLocation = log.debugLocation,
            let functionName = debugLocation.functionName else {
        fatalError("'MTLCommandBuffer' log should have debug info.")
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

// MARK: - SmallVector

// Similar to the `SmallVector<_, _>` C++ type in LLVM.
struct SmallVector<Vector: SIMD>
where Vector.Scalar: FixedWidthInteger & SIMDScalar {
  typealias Scalar = Vector.Scalar
  
  private var vector: Vector
  private(set) var count: Int
  private var array: [Vector.Scalar]?
  
  @inline(__always)
  init() {
    vector = .zero
    count = 0
  }
  
  // Need `Collection` instead of `Sequence` to query the number of elements efficiently. The
  // restriction on `C.Index` makes cycling through indices faster and more concise.
  @inline(__always)
  init<C: Collection>(_ elements: C) where C.Element == Scalar, C.Index == Int {
    vector = .zero
    count = elements.count
    
    // `<=` because the count does not change.
    if _fastPath(count <= Vector.scalarCount) {
      for i in elements.indices {
        vector[i] = elements[i]
      }
    } else {
      array = Array(elements)
    }
  }
  
  @inline(__always)
  mutating func append(_ newElement: Scalar) {
    // Not `<=` because the count increases by one.
    if count < Vector.scalarCount {
      vector[count] = newElement
    } else {
      if _slowPath(array == nil) {
        array = Array(unsafeUninitializedCapacity: Vector.scalarCount &+ 1) { ptr, count in
          count = self.count
          ptr.withMemoryRebound(to: Vector.self) { ptr in
            ptr[0] = vector
          }
        }
      }
      array!.append(newElement)
    }
    count &+= 1
  }
  
  @inline(__always)
  subscript(index: Int) -> Scalar {
    if count <= Vector.scalarCount {
      return vector[index]
    } else {
      // Safely unwrapping the array could become a bottleneck when this is looped over several
      // times.
      return array.unsafelyUnwrapped[index]
    }
  }
  
  @inline(__always)
  func elementsEqual(_ other: Self) -> Bool {
    guard count == other.count else {
      return false
    }
    if _fastPath(count <= Vector.scalarCount) {
      // Unused lanes are zeroed out, so equal instances have equal vectors.
      return vector == other.vector
    } else {
      return array!.elementsEqual(other.array!)
    }
  }
  
  @inline(__always)
  func copy(into bufferPointer: UnsafeMutableBufferPointer<Scalar>) {
    let bufferSize = bufferPointer.count
    precondition(
      count == bufferSize,
      "Attempted to copy 'SmallVector' of size \(count) into buffer of size \(bufferSize).")
    let baseAddress = bufferPointer.baseAddress!
    if count <= Vector.scalarCount {
      for i in 0..<count {
        baseAddress[i] = vector[i]
      }
    } else {
      array!.withUnsafeBufferPointer {
        _ = bufferPointer.initialize(from: $0)
      }
    }
  }
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
  
  func makeString() -> String {
    String(unsafeUninitializedCapacity: ptr.count) { bufferPointer in
      _ = bufferPointer.initialize(from: ptr)
      return ptr.count
    }
  }
}

extension StringWrapper: ExpressibleByStringLiteral {
  @inline(__always)
  init(stringLiteral value: StaticString) {
    self.init(value)
  }
}

struct MovingAverage<Element: FixedWidthInteger> {
  private var data: [Element]
  private var sum: Element
  private(set) var average: Element
  private var index: Int
  private var count: Int { data.count }
  
  init(repeating repeatedValue: Element, count: Int) {
    precondition(count > 0, "'MovingAverage' cannot be empty.")
    data = Array(repeating: repeatedValue, count: count)
    sum = repeatedValue * Element(count)
    average = repeatedValue
    index = 0
  }
  
  mutating func append(_ newElement: Element) {
    let oldElement = data[index]
    data[index] = newElement
    sum -= oldElement
    sum += newElement
    average = sum / Element(count)
    
    index += 1
    if index == count {
      index = 0
    }
  }
}
