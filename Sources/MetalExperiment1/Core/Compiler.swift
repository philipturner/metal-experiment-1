//
//  Compiler.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

extension Context {
  // TODO's #1 and #2 have the same purpose. #1 might be the heuristic required for #2.
  //
  // TODO: Make specialized execution paths for having either 1 or 2 eager operations queued up.
  // These paths will have lower CPU-side latency.
  //
  // TODO: Make an option that encourages longer compile time when there's modest encoding
  // backpressure. This would enable complex fusions like non-adjacent unary operations, which
  // require either peeking ahead or running a second pass on the IR. When running two passes, it
  // won't mark allocations as `initialized` until after the non-adjacent fusions.
  //
  // TODO: Add a property to certain operations, which tells the encoder to perform them on the CPU.
  // This is how I will achieve "constant folding". Alternatively, the encoder can make that
  // decision by querying the allocations' size and whether they're in the special CPU-side storage
  // mode.
  func compileEagerOperations() -> [CompiledOperation] {
    if Allocation.debugInfoEnabled {
      print("Compiler pass starts with \(eagerOperations.count) operations.")
    }
    defer {
      if Allocation.debugInfoEnabled {
        print("Compiler pass ends.")
      }
    }
    precondition(eagerOperations.count > 0)
    defer {
      eagerOperations.removeAll(keepingCapacity: true)
    }
    var compiledOperations: [CompiledOperation] = []
    compiledOperations.reserveCapacity(eagerOperations.count)
    
    var fusionTypes: TypeList16<UnaryOperationType> = .init()
    var fusionDataTypes: TypeList4<DataType> = .init()
    var fusionHead: Allocation?
    var fusionTail: Allocation?
    var fusionTailID: UInt64 = .max
    var fusionSize = -1
    @inline(__always)
    func pendingOperationFusionExists() -> Bool {
      fusionSize != -1
    }
    
    // Call this before encoding operations that can't be fused. Avoid proactively peeking at the
    // next operation and seeing whether the fusion ends, because that's costly (0.03 - 0.04 µs).
    @inline(never)
    func appendOperationFusion() {
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
        fatalError("Something went wrong while fusing operations")
      }
      
      // Make the fusion tail valid to read from.
      fusionTail.initialized = true
      
      let multiUnary = CompiledOperation.MultiUnary(
        types: fusionTypes, dataTypes: fusionDataTypes, input: fusionHead, output: fusionTail,
        size: fusionSize)
      compiledOperations.append(.multiUnary(multiUnary))
      if _slowPath(Allocation.debugInfoEnabled || Context.profilingEncoding) {
        if fusionTypes.count >= 2 {
          print("*** Fused \(fusionTypes.count) unary operations ***")
        } else {
          print("Appended single unary operation")
        }
      }
    }
    
    // Separating each case with a newline to make this much easier to read.
    for i in eagerOperations.indices {
      let eagerOperation = eagerOperations[i]
      switch eagerOperation {
      case .unary(let unary):
        var input: Allocation
        var inputDataType: DataType
        if unary.input == fusionTailID {
          // In the middle of an operation fusion.
          input = fusionTail!
          inputDataType = input.dataType
          
          // The tail was the output of previous operation. Something that's initialized by the
          // frontend can't be the output of an operation, only the input.
          precondition(!input.initialized)
        } else {
          // At the start of an operation fusion.
          if pendingOperationFusionExists() {
            // Finish the previous operation fusion. When fusing non-adjacent operations, don't
            // check whether the output needs to be computed. It might make the JIT graph compiler
            // take longer, or make benchmarks difficult. TODO: Revisit this decision after more
            // complex fusions are implemented.
            appendOperationFusion()
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
        
      case .explicitCopy(let explicitCopy):
        if pendingOperationFusionExists() {
          appendOperationFusion()
        }
        
        let input = _internalFetch(explicitCopy.input)
        let output = _internalFetch(explicitCopy.output)
        precondition(input.dataType == output.dataType)
        let byteCount = input.byteCount
        precondition(byteCount == output.byteCount)
        
        output.initialized = true
        _internalRelease(input)
        _internalRelease(output)
        let explicitCopy = CompiledOperation.ExplicitCopy(
          input: input, output: output, byteCount: byteCount)
        compiledOperations.append(.explicitCopy(explicitCopy))
      }
    }
    
    // Finish compilation and return the compiled operations.
    if pendingOperationFusionExists() {
      appendOperationFusion()
    }
    return compiledOperations
  }
}
