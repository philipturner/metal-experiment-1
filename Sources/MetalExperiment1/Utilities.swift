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
func withUnsafeAddress<T: AnyObject>(
  of object: T,
  _ body: (UnsafeMutableRawPointer) throws -> Void
) rethrows {
  let ptr = Unmanaged<T>.passUnretained(object).toOpaque()
  try body(ptr)
}
