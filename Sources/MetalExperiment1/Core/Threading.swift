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
  // on each function call. The actual dispatch queue will still be opaque. This should be avoided
  // anywhere else until there are definitive metrics showing it improves performance.
  //
  // This blocks memory retained by a command buffer from deallocating. Unless the operations
  // contained by it are likely to fuse, this may degrade performance.
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
