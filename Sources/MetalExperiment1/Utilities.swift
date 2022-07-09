//
//  Utilities.swift
//  
//
//  Created by Philip Turner on 7/8/22.
//

import Foundation

enum Profiler {
  // Microseconds.
  static let timeUnit = "\u{b5}s"
  
  private static var time: UInt64?
  
  @discardableResult
  static func checkpoint() -> UInt64 {
    let lastTime = time
    let currentTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    time = currentTime
    if let lastTime = lastTime {
      return (currentTime - lastTime) / 1000
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
