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
  func compileEagerOperations() -> [Instruction] {
    if Allocation.debugInfoEnabled {
      print("Compiler pass starts with \(eagerOperations.count) operations.")
    }
    defer {
      if Allocation.debugInfoEnabled {
        print("Compiler pass ends.")
      }
    }
    precondition(eagerOperations.count > 0, "Compiled without any eager operations.")
    defer {
      eagerOperations.removeAll(keepingCapacity: true)
    }
    var instructions: [Instruction] = []
    instructions.reserveCapacity(eagerOperations.count)
    
    var fusionOperations: SmallVector<SIMD8<UInt16>> = .init()
    var fusionMetadata: SmallVector<SIMD2<UInt64>> = .init()
    var fusionDataGroup: DataGroup?
    var fusionHeadAllocation1: Allocation?
    var fusionHeadAllocation2: Allocation?
    var fusionHeadAllocation3: Allocation?
    var fusionTailReferenceCount: Int = -9999
    var fusionTail: AllocationHandle?
    var fusionSize: Int = -9999
    
    // Call this before encoding operations that can't be fused. Avoid proactively peeking at the
    // next operation and seeing whether the fusion ends, because that's costly (0.03 - 0.04 µs).
    @inline(never)
    func appendOperationFusion() {
      defer {
        fusionOperations = .init()
        fusionMetadata = .init()
        fusionDataGroup = nil
        fusionHeadAllocation1 = nil
        fusionHeadAllocation2 = nil
        fusionHeadAllocation3 = nil
        fusionTailReferenceCount = -9999
        fusionTail = nil
        fusionSize = -9999
      }
      guard let fusionHeadAllocation1 = fusionHeadAllocation1,
            fusionTailReferenceCount >= 0,
            let fusionTail = fusionTail,
            fusionSize >= 0,
            let fusionDataGroup = fusionDataGroup else {
        fatalError("Something went wrong while fusing operations.")
      }
      if fusionHeadAllocation2 == nil {
        guard fusionHeadAllocation3 == nil else {
          fatalError("Something went wrong while fusing operations.")
        }
      }
      
      // The frontend will never read this operation's results, so abort it. This may make
      // benchmarks difficult, because not every dispatched operation actually executes.
      //
      // When fusing non-adjacent operation chains, the reference count is no longer a 100% reliable
      // way to abort unused operations. Instead, analyze the graph and trace back unused ends. If
      // an "end" fusion chain ends with a zombie (zero-refcount) tensor, the zombie-ness transfers
      // to anything that fuses with it.
      if fusionTailReferenceCount == 0 {
        return
      }
      
      // Due to custom reference counting semantics, the fusion tail object might be deallocated if
      // its reference count is zero. Emphasis on "might", because the frontend could have released
      // it and would be waiting on the mutex to deallocate it. In that case, the allocation could
      // still exist when returning in the above statement. There is nothing wrong the allocation
      // existing there. If it didn't return early and the object was deallocated, then there would
      // be a problem.
      let fusionTailAllocation = fusionTail.reference!.takeUnretainedValue()
      
      // Make the fusion tail valid to read from. This does not prevent it from being optimized away
      // in a later compiler pass; that's the job of `referenceCount`.
      fusionTailAllocation.initialized = true
      
      let elementwise = Instruction.Elementwise(
        operations: fusionOperations,
        metadata: fusionMetadata,
        dataGroup: fusionDataGroup,
        input1: fusionHeadAllocation1,
        input2: fusionHeadAllocation2,
        input3: fusionHeadAllocation3,
        output: fusionTailAllocation,
        size: fusionSize)
      instructions.append(.elementwise(elementwise))
      if _slowPath(Allocation.debugInfoEnabled || Context.profilingEncoding) {
        if fusionOperations.count >= 2 {
          // This number does not include no-ops that were fused.
          print("*** Fused \(fusionOperations.count) unary operations ***")
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
        let (input, output) = (unary.input, unary.output)
        var restartingFusion: Bool
        if input == fusionTail {
          // In the middle of an operation fusion.
          if fusionDataGroup! == unary.dataGroup {
            restartingFusion = false
          } else {
            restartingFusion = true
          }
          
          // The tail was the output of previous operation. Something initialized by the frontend
          // can't be an operation's output; only an input. Since the check may incur ARC overhead,
          // only perform it in debug mode.
          assert(!input.reference!.takeUnretainedValue().initialized)
        } else {
          // At the start of an operation fusion. A previous fusion may already be in progress.
          restartingFusion = true
        }
        
        // Decrement input's reference count.
        let referenceCount = input.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        if Allocation.debugInfoEnabled {
          _internalReleaseSlowPath(input, referenceCount)
        }
        if !restartingFusion {
          restartingFusion = referenceCount > 0
        }
        
        // Restart operation fusion.
        if restartingFusion {
          if fusionDataGroup != nil {
            appendOperationFusion()
          }
          fusionHeadAllocation1 = input.reference!.takeUnretainedValue()
          fusionSize = input.dataType.contiguousSize(byteCount: input.byteCount)
          fusionDataGroup = unary.dataGroup
        }
        
        // Append operation.
        precondition(
          input.shape.elementsEqual(output.shape), "Input shape did not match output shape.")
        if !unary.isNoOp {
          precondition(
            unary.operation != UInt16.max, "'unary.operation' had the raw value of a no-op.")
          fusionOperations.append(unary.operation)
          if let metadata = unary.metadata {
            fusionMetadata.append(metadata)
          }
        }
        
        // Release input.
        if referenceCount == 0 {
          let reference = input.reference!
          input.reference = nil
          reference.release()
        }
        
        // Update fusion tail.
        fusionTailReferenceCount = _internalRelease(output)
        fusionTail = output
        
      case .explicitCopy(let explicitCopy):
        if fusionDataGroup != nil {
          appendOperationFusion()
        }
        
        let (input, output) = (explicitCopy.input, explicitCopy.output)
        let inputAllocation = input.reference!.takeUnretainedValue()
        let outputAllocation = output.reference!.takeUnretainedValue()
        _internalRelease(input)
        _internalRelease(output)
        precondition(input.dataType == output.dataType, "Data types did not match.")
        let byteCount = explicitCopy.input.byteCount
        precondition(byteCount == explicitCopy.output.byteCount, "Byte counts did not match.")
        
        outputAllocation.initialized = true
        let explicitCopy = Instruction.ExplicitCopy(
          input: inputAllocation, output: outputAllocation, byteCount: byteCount)
        instructions.append(.explicitCopy(explicitCopy))
      }
    }
    
    // Finish compilation and return the compiled operations.
    if fusionDataGroup != nil {
      appendOperationFusion()
    }
    return instructions
  }
}
