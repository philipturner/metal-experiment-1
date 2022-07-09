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

extension MTLSize: ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
  @inlinable @inline(__always)
  public init(integerLiteral value: Int) {
    self = [value, 1, 1]
  }
  
  @inlinable @inline(__always)
  public init(arrayLiteral elements: Int...) {
    switch elements.count {
    case 1:  self = MTLSizeMake(elements[0], 1, 1)
    case 2:  self = MTLSizeMake(elements[0], elements[1], 1)
    case 3:  self = MTLSizeMake(elements[0], elements[1], elements[2])
    default: fatalError("A MTLSize must not exceed three dimensions!")
    }
  }
}

extension MTLOrigin: ExpressibleByArrayLiteral {
  @inlinable @inline(__always)
  public init(arrayLiteral elements: Int...) {
    switch elements.count {
    case 2:  self = MTLOriginMake(elements[0], elements[1], 0)
    case 3:  self = MTLOriginMake(elements[0], elements[1], elements[2])
    default: fatalError("A MTLOrigin must have two or three dimensions!")
    }
  }
}

extension MTLSamplePosition: ExpressibleByArrayLiteral {
  @inlinable @inline(__always)
  public init(arrayLiteral elements: Float...) {
    switch elements.count {
    case 2:  self = .init(x: elements[0], y: elements[1])
    default: fatalError("A MTLSamplePosition must have two dimensions")
    }
  }
}
