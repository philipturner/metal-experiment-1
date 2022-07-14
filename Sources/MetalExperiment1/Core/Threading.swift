//
//  Threading.swift
//  
//
//  Created by Philip Turner on 7/14/22.
//

import Foundation

extension Context {
  private static let dispatchQueueIdentifier = DispatchSpecificKey<Bool>()
  static let _dispatchQueue: DispatchQueue = {
    let output = DispatchQueue(label: "com.s4tf.metal.Context.dispatchQueue")
    output.setSpecific(key: dispatchQueueIdentifier, value: true)
    return output
  }()
  
  @inline(__always)
  static func testSynchronizationOverhead() {
    _ = DispatchQueue.getSpecific(key: dispatchQueueIdentifier)
  }
  
  // A special `with` function the caller can use to make everything inside happen on the
  // dispatch queue. This eliminates the extra 0.3 µs overhead from calling into the dispatch queue
  // on each function call. The actual dispatch queue will still be opaque. This should be used
  // widely inside the _Raw namespace, but avoided anywhere else until there are definitive metrics
  // showing it improves performance.
  //
  // Should should be selectively applied to frontend functions showing a
  // very high overhead. In such functions, it often removes overhead from invisible tasks like
  // deallocating an allocation ID. It incurs a minimum overhead of 0.02 µs, so 20 redundant calls
  // to it equal the same overhead of calling into the dispatch queue. Because of this overhead,
  // calling into `withDispatchQueue` is avoided internally when possible.
  //
  // Also, once you enter synchronization in any situation, this wrapper could free deadlocks. But
  // wait to implement that feature so you can find thread synchronization bugs in the early stages
  // of development.
  public static func withDispatchQueue<T>(_ body: () throws -> T) rethrows -> T {
    if DispatchQueue.getSpecific(key: dispatchQueueIdentifier) == true {
      return try body()
    } else {
      return try _dispatchQueue.sync {
        let output = try body()
        return output
      }
    }
  }
}
