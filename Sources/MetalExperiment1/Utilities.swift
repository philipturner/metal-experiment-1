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
        array = Array(unsafeUninitializedCapacity: Vector.scalarCount + 1) { bufferPointer, count in
          count = self.count
          bufferPointer.withMemoryRebound(to: Vector.self) { ptr in
            ptr[0] = vector
          }
        }
      }
      array!.append(newElement)
    }
    count += 1
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
