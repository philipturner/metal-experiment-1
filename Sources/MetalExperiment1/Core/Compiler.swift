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
    
    var fusionTypes: OperationTypeList16<UnaryOperationType> = .init()
    var fusionDataTypes: OperationTypeList4<DataType> = .init()
    var fusionHead: Allocation?
    var fusionTail: Allocation?
    var fusionTailID: UInt64 = .max
    var fusionSize = -1
    @inline(__always)
    func pendingFusionOperationExists() -> Bool {
      fusionSize != -1
    }
    
    // Call this before encoding operations that can't be fused.
    @inline(never)
    func appendFusionOperation() {
      defer {
        fusionTypes = .init()
        fusionDataTypes = .init()
        fusionHead = nil
        fusionTail = nil
        fusionTailID = .max
        fusionSize = -1
      }
      guard let fusionHead = fusionHead,
            let fusionTail = fusionTail,
            fusionTypes.count > 0,
            fusionDataTypes.count > 0,
            fusionSize >= 0 else {
        fatalError("Something went wrong while fusing operators")
      }
      
      // Make the fusion tail valid to read from.
      fusionTail.initialized = true
      
      let multiUnary = CompiledOperation.MultiUnary(
        types: fusionTypes, dataTypes: fusionDataTypes, input: fusionHead, output: fusionTail,
        size: fusionSize)
      compiledOperations.append(.multiUnary(multiUnary))
      if _slowPath(Allocation.debugInfoEnabled || Context.profilingEncoding) {
        if fusionTypes.count >= 2 {
          print("*** Fused \(fusionTypes.count) unary operators ***")
        } else {
          print("Appended single unary operation")
        }
      }
    }
    
    for i in eagerOperations.indices {
      let eagerOperation = eagerOperations[i]
      switch eagerOperation {
      case .explicitCopy(let explicitCopy):
        if pendingFusionOperationExists() {
          appendFusionOperation()
        }
        let input = _internalFetch(explicitCopy.input)
        let output = _internalFetch(explicitCopy.output)
        precondition(input.dataType == output.dataType)
        // TODO: Finish implementing this.
        
      case .unary(let unary):
        var input: Allocation
        var inputDataType: DataType
        if unary.input == fusionTailID {
          // In the middle of an operator fusion.
          input = fusionTail!
          inputDataType = input.dataType
          
          // The tail was the output of previous operation. Something that's initialized by the
          // frontend can't be the output of an operation, only the input.
          precondition(!input.initialized)
        } else {
          // At the start of an operator fusion.
          if pendingFusionOperationExists() {
            // Finish the previous operator fusion. When fusing non-adjacent operators, don't check
            // whether the output needs to be computed. It might make the JIT graph compiler take
            // longer, or make benchmarks difficult. TODO: Revisit this decision after more complex
            // fusions are implemented.
            appendFusionOperation()
          }
          input = _internalFetch(unary.input)
          inputDataType = input.dataType
          precondition(inputDataType == .float32, "Execution currently limited to 'Float'")
          fusionDataTypes.append(inputDataType)
          fusionHead = input
          fusionSize = inputDataType.contiguousSize(byteCount: input.byteCount)
        }
        fusionTypes.append(unary.type)
        
        let output = _internalFetch(unary.output)
        let outputDataType = output.dataType
        precondition(inputDataType == outputDataType, "Casting not yet supported")
        precondition(input.byteCount == output.byteCount)
        fusionTail = output
        fusionTailID = unary.output
        _internalRelease(input)
        _internalRelease(output)
      }
    }
    if pendingFusionOperationExists() {
      appendFusionOperation()
    }
    
    return compiledOperations
  }
}
