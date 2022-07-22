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
    
    var fusionOperations: SmallVector<SIMD8<UInt16>> = .init()
    var fusionMetadata: SmallVector<SIMD2<UInt64>> = .init()
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
        fusionOperations = .init()
        fusionMetadata = .init()
        fusionHead = nil
        fusionTail = nil
        fusionTailID = .max
        fusionSize = -1
      }
      guard let fusionHead = fusionHead,
            let fusionTail = fusionTail,
            fusionSize >= 0 else {
        fatalError("Something went wrong while fusing operations.")
      }
      
      // The frontend will never read this operation's results, so abort it. This may make
      // benchmarks difficult, because not every dispatched operation actually executes.
      //
      // When fusing non-adjacent operation chains, the reference count is no longer a 100% reliable
      // way to abort unused operations. Instead, analyze the graph and trace back unused ends. If
      // an "end" fusion chain ends with a zombie (zero-refcount) tensor, the zombie-ness transfers
      // to anything that fuses with it.
      if fusionTail.referenceCount == 0 {
        return
      }
      
      // Make the fusion tail valid to read from. This does not prevent it from being optimized away
      // in a later compiler pass; that's the job of `referenceCount`.
      fusionTail.initialized = true
      
      let elementwise = CompiledOperation.Elementwise(
        operations: fusionOperations, metadata: fusionMetadata, input: fusionHead,
        output: fusionTail, size: fusionSize)
      compiledOperations.append(.elementwise(elementwise))
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
        var input: Allocation
        var startingNewFusion = false
        
        if unary.input == fusionTailID {
          // In the middle of an operation fusion.
          precondition(pendingOperationFusionExists())
          input = fusionTail!
          
          // The tail was the output of previous operation. Something initialized by the frontend
          // can't be an operation's output; only an input.
          precondition(!input.initialized)
          startingNewFusion = input.referenceCount > 1
        } else {
          // At the start of an operation fusion. A previous fusion may already be in progress.
          input = _internalFetch(unary.input)
          
          precondition(input.referenceCount > 0)
          startingNewFusion = true
        }
        _internalRelease(input)
        
        if startingNewFusion {
          if pendingOperationFusionExists() {
            // Finish the previous operation fusion. When fusing non-adjacent operations, don't
            // check whether the output needs to be computed. It might make the JIT graph compiler
            // take longer, or make benchmarks difficult. TODO: Revisit this decision after more
            // complex fusions are implemented.
            appendOperationFusion()
          }
          
          fusionHead = input
          fusionSize = input.dataType.contiguousSize(byteCount: input.byteCount)
        } else if false {
          // `input.referenceCount` could be zero.
          precondition(!pendingOperationFusionExists())
          fusionHead = input
          fusionSize = input.dataType.contiguousSize(byteCount: input.byteCount)
        } else {
          // Waiting for tests to trigger this assertion failure. I don't know why this is always
          // true.
          precondition(input.referenceCount == 0)
        }
        
        if !unary.isNoOp {
          fusionOperations.append(unary.operation.rawValue)
          if let metadata = unary.metadata {
            fusionMetadata.append(metadata)
          }
        }
        
        let output = _internalFetch(unary.output)
        _internalRelease(output)
        precondition(input.shape.elementsEqual(output.shape))
        fusionTail = output
        fusionTailID = unary.output
        
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
