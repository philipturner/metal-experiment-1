//
//  Compiler.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

import Metal

extension Context {
  // Make specialized execution paths for having either 1 or 2 eager operations queued up. These
  // paths will have lower CPU-side latency.
  func compileEagerOperations() -> [CompiledOperation] {
    if _slowPath(Allocation.debugInfoEnabled) {
      print("Compiler pass starts with \(eagerOperations.count) operations.")
    }
    defer {
      if _slowPath(Allocation.debugInfoEnabled) {
        print("Compiler pass ends.")
      }
    }
    precondition(eagerOperations.count > 0)
    defer {
      eagerOperations.removeAll(keepingCapacity: true)
    }
    var compiledOperations: [CompiledOperation] = []
    compiledOperations.reserveCapacity(eagerOperations.count)
    
    var unaryFusionArray: OperationTypeList16<UnaryOperationType> = .init()
    var unaryFusionHead: Allocation?
    var unaryFusionTail: Allocation?
    var unaryFusionTailID: UInt64 = .max
    var unaryFusionSize = -1
    @inline(__always)
    func pendingFusionOperationExists() -> Bool {
      unaryFusionSize != -1
    }
    
    // Call this before encoding non-unary operations.
    @inline(never)
    func appendFusionOperation() {
      defer {
        unaryFusionArray = .init()
        unaryFusionHead = nil
        unaryFusionTail = nil
        unaryFusionTailID = .max
        unaryFusionSize = -1
      }
      guard let unaryFusionHead = unaryFusionHead,
            let unaryFusionTail = unaryFusionTail,
            unaryFusionArray.count > 0,
            unaryFusionSize >= 0 else {
        fatalError("This should never happen.")
      }
      
      // Make the unary fusion tail valid to read from.
      unaryFusionTail.initialized = true
      
      let multiUnary = CompiledOperation.MultiUnary(
        types: unaryFusionArray, input: unaryFusionHead, output: unaryFusionTail,
        size: unaryFusionSize)
      compiledOperations.append(.multiUnary(multiUnary))
      if _slowPath(Allocation.debugInfoEnabled || Context.profilingEncoding) {
        if unaryFusionArray.count >= 2 {
          print("*** Fused \(unaryFusionArray.count) unary operators ***")
        } else {
          print("Appended single unary operation")
        }
      }
    }
    
    for i in eagerOperations.indices {
      let eagerOperation = eagerOperations[i]
      switch eagerOperation {
      case .unary(let unary):
        var input: Allocation
        if unary.input == unaryFusionTailID {
          // In the middle of a unary fusion.
          input = unaryFusionTail!
          
          // The tail was the output of previous operation. Something that's initialized by the
          // frontend can't be the output of an operation, only the input.
          precondition(!input.initialized)
        } else {
          // At the start of a unary fusion.
          if pendingFusionOperationExists() {
            // Finish the previous unary fusion. When fusing non-adjacent operators, don't check
            // whether the output needs to be computed. It might make the JIT graph compiler take
            // longer, or make benchmarks difficult. TODO: Revisit this decision after more complex
            // fusions are implemented.
            appendFusionOperation()
          }
          input = _compilerFetchAllocation(id: unary.input)
          unaryFusionHead = input
          unaryFusionSize = unary.size
        }
        unaryFusionArray.append(unary.type)
        
        let output = _compilerFetchAllocation(id: unary.output)
        unaryFusionTail = output
        unaryFusionTailID = unary.output
        _compilerRelease(input)
        _compilerRelease(output)
      }
    }
    if pendingFusionOperationExists() {
      appendFusionOperation()
    }
    
    return compiledOperations
  }
}
