//
//  Compiler.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import Metal

extension Context {
  // Start off with a no-op compiler, which doesn't fuse any placeholders. Aggressively optimize the
  // function calls here for low CPU-side overhead.
  func compileEagerOperations() -> [CompiledOperation] {
    precondition(eagerOperations.count > 0)
    defer {
      eagerOperations.removeAll(keepingCapacity: true)
    }
    var compiledOperations: [CompiledOperation] = []
    compiledOperations.reserveCapacity(eagerOperations.count)
    
    for eagerOperation in eagerOperations {
      switch eagerOperation {
      case .unary(let unary):
        let types = [unary.type]
        let input = try! _unsafeFetchAllocation(id: unary.input, releasingZombies: true)!
        let output = try! _unsafeFetchAllocation(id: unary.output, releasingZombies: true)!
        // Allow this to become a zombie instead of deallocating normally.
        output.initialized = true
        try! input.materialize()
        try! output.materialize()
        let size = unary.size
        
        let multiUnary = CompiledOperation.MultiUnary(
          types: types, input: input, output: output, size: size)
        compiledOperations.append(.multiUnary(multiUnary))
      }
    }
    
    return compiledOperations
  }
}
