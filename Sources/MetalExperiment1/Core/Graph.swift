//
//  Graph.swift
//  
//
//  Created by Philip Turner on 7/30/22.
//

struct Graph {
  private(set) var instructions: [Instruction?]
  
  // Keep the cache's object around between compiler passes, avoiding the overhead of allocatng a
  // Swift object every pass.
  private static var cache: [AllocationHandle: Int] = [:]
  
  init(eagerOperationCount: Int) {
    self.instructions = []
    self.instructions.reserveCapacity(eagerOperationCount)
    Self.cache.removeAll(keepingCapacity: true)
  }
}

extension Graph {
  mutating func append(_ elementwise: Instruction.Elementwise, tailReferenceCount: Int) {
    instructions.append(.elementwise(elementwise))
    if tailReferenceCount == 1 {
      // The cache should never already contain the tail. This check is costly, so only perform it
      // in debug mode.
      assert(Self.cache[elementwise.output.handle] == nil, "Cache already contained tail.")
      Self.cache[elementwise.output.handle] = instructions.count - 1
    }
  }
  
  mutating func append(_ explicitCopy: Instruction.ExplicitCopy) {
    instructions.append(.explicitCopy(explicitCopy))
  }
  
  var endsWithPlaceholder: Bool {
    instructions.count > 0 && instructions.last! == nil
  }
}

extension Graph {
  struct SearchKey {
    var handle: AllocationHandle
    var referenceCount: Int
  }
  
  @inline(__always)
  mutating func remove(
    matching key1: SearchKey,
    _ key2: SearchKey? = nil,
    _ key3: SearchKey? = nil
  ) -> Instruction.Elementwise? {
    if key1.referenceCount == 0,
       key2?.referenceCount == 0,
       key3?.referenceCount == 0 {
      return nil
    }
    if Self.cache.isEmpty {
      return nil
    }
    return removeSlowPath(matching: key1, key2, key3)
  }
  
  @inline(never)
  private mutating func removeSlowPath(
    matching key1: SearchKey,
    _ key2: SearchKey?,
    _ key3: SearchKey?
  ) -> Instruction.Elementwise? {
    nil
  }
  
  // func markZombie
}
