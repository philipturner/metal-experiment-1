//
//  Compilation.swift
//  
//
//  Created by Philip Turner on 7/13/22.
//

extension MTLPluggableDevice {
  func compileEagerOperations() -> [Instruction?] {
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
    
    var graph = Graph(
      eagerOperationCount: eagerOperations.count,
      showingDebugInfo: Allocation.debugInfoEnabled || MTLPluggableDevice.profilingEncoding)
    var fusion: Instruction.Elementwise = createBlankFusion()
    var fusionTailReferenceCount: Int = -9999
    var fusionTail: AllocationHandle?
    
    @inline(__always)
    func createBlankFusion() -> Instruction.Elementwise {
      .init(
        operations: .init(),
        metadata: .init(),
        dataGroup: .u32_i64_u64, // Override this, otherwise results are undefined.
        numFusedUnaryOperations: 0,
        numFusedNonUnaryOperations: 0,
        input1: nil,
        input2: nil,
        input3: nil,
        input4: nil,
        output: nil,
        size: -9999)
    }
    
    // Extracts matching instruction from history to continue working on. Changes the variables
    // above to reflect the extracted instruction. Returns whether context did switch.
    //
    // Call `switchContext` based on the following heuristic. If there were no previous instructions
    // and you just added one because of a forced graph break, don't search the history. There won't
    // be any matching instructions.
    //
    // Switching contexts is costly (+1.0 µs), but balanced out by a reduction in encoding time
    // (-1.0 µs).
    @inline(__always)
    func switchContext(
      _ key1: Graph.SearchKey,
      _ key2: Graph.SearchKey? = nil,
      _ key3: Graph.SearchKey? = nil,
      dataGroup: DataGroup,
      availableHeads: Int
    ) -> Bool {
      if graph.shouldRemove(matching: key1, key2, key3) {
        return switchContextSlowPath(key1, key2, key3, dataGroup, availableHeads)
      } else {
        return false
      }
    }
    
    @inline(never)
    func switchContextSlowPath(
      _ key1: Graph.SearchKey,
      _ key2: Graph.SearchKey?,
      _ key3: Graph.SearchKey?,
      _ dataGroup: DataGroup,
      _ availableHeads: Int
    ) -> Bool {
      if let elementwise = graph.remove(
        matching: key1, key2, key3, dataGroup: dataGroup, availableHeads: availableHeads)
      {
        fusion = elementwise
        fusionTailReferenceCount = 0
        fusionTail = fusion.output.handle
        
        if _slowPath(graph.showingDebugInfo) {
          print("""
              Context switch (\(fusion.numFusedUnaryOperations) unary, \
            \(fusion.numFusedNonUnaryOperations) non-unary)
            """)
        }
        return true
      } else {
        return false
      }
    }
    
    // Call this before encoding operations that can't be fused. Avoid proactively peeking at the
    // next operation and seeing whether the fusion ends, because that's costly (0.03 - 0.04 µs).
    @inline(never)
    func appendOperationFusion() {
      defer {
        fusion = createBlankFusion()
        fusionTailReferenceCount = -9999
        fusionTail = nil
      }
      func noMissingHeads() -> Bool {
        if fusion.input2 == nil {
          if fusion.input3 != nil {
            return false
          }
        } else if fusion.input3 == nil {
          if fusion.input4 != nil {
            return false
          }
        }
        return true
      }
      guard fusion.input1 != nil,
            let fusionTail = fusionTail,
            fusionTailReferenceCount >= 0,
            fusion.size >= 0,
            noMissingHeads() else {
        fatalError("Something went wrong while fusing operations.")
      }
      
      // The frontend will never read this operation's results, so abort it. The tail was
      // deallocated; dereferencing its handle creates a runtime segfault. This may make benchmarks
      // difficult, because not every dispatched operation actually executes.
      guard fusionTailReferenceCount > 0 else {
        return
      }
      
      // If the current `fusion` came from a context switch, the compiler guarantees at least one
      // operation was appended. The new tail never equals the previous tail.
      precondition(fusion.output?.handle != fusionTail)
      
      // Due to custom reference counting semantics, the fusion tail object might be deallocated if
      // its reference count is zero. Emphasis on "might", because the frontend could have released
      // it and would be waiting on the mutex to deallocate it. In that case, the allocation could
      // still exist when returning in the above statement. There is nothing wrong the allocation
      // existing there. If it didn't return early and the object was deallocated, then there would
      // be a problem.
      fusion.output = fusionTail.reference!.takeUnretainedValue()
      
      // The tail was the output of previous operation. Something initialized by the frontend can't
      // be an operation's output; only an input can.
      precondition(!fusion.output!.initialized, "Fusion tail should not be initialized yet.")
      
      // Make the fusion tail valid to read from. This does not prevent it from being optimized away
      // in a later compiler pass; that's the job of `referenceCount`.
      fusion.output!.initialized = true
      
      if _slowPath(graph.showingDebugInfo) {
        let numFusedOperations = fusion.numFusedUnaryOperations + fusion.numFusedNonUnaryOperations
        if numFusedOperations >= 2 {
          print("""
              Fused \(numFusedOperations) operations (\(fusion.numFusedUnaryOperations) unary, \
            \(fusion.numFusedNonUnaryOperations) non-unary)
            """)
        } else if fusion.numFusedUnaryOperations == 1 {
          precondition(fusion.numFusedNonUnaryOperations == 0)
          print("  Appended single unary operation")
        } else if fusion.numFusedNonUnaryOperations == 1 {
          precondition(fusion.numFusedUnaryOperations == 0)
          print("  Appended single non-unary operation")
        } else {
          print("  Appended copying operation")
        }
      }
      graph.append(fusion, tailReferenceCount: fusionTailReferenceCount)
    }
    
    // MARK: - Switch Statement
    
    // Separating each case with a newline to make this much easier to read.
    for i in eagerOperations.indices {
      let eagerOperation = eagerOperations[i]
      switch eagerOperation {
      case .unary(let unary):
        let (input, output) = (unary.input, unary.output)
        var restartingFusion = true
        if input == fusionTail {
          // In the middle of an operation fusion.
          if fusion.dataGroup == unary.dataGroup {
            restartingFusion = false
          }
        } else {
          // At the start of an operation fusion. A previous fusion may already be in progress.
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
          let shouldTryRemoval = graph.shouldTryRemoval
          if fusion.input1 != nil {
            appendOperationFusion()
          }
          if shouldTryRemoval && switchContext(
            .init(input, referenceCount), dataGroup: unary.dataGroup, availableHeads: 0) {
            // Switched context.
          } else {
            fusion.input1 = input.reference!.takeUnretainedValue()
            fusion.size = input.dataType.contiguousSize(byteCount: input.byteCount)
            fusion.dataGroup = unary.dataGroup
          }
        }
        
        // Append operation.
        precondition(
          input.shape.elementsEqual(output.shape), "Input shape did not match output shape.")
        if unary.operation == .max {
          // Skip no-ops.
        } else {
          fusion.operations.append(unary.operation)
          if let metadata = unary.metadata {
            fusion.metadata.append(metadata)
          }
        }
        fusion.numFusedUnaryOperations += 1
        
        // Release input.
        if referenceCount == 0 {
          let reference = input.reference!
          input.reference = nil
          reference.release()
        }
        
        // Update fusion tail.
        fusionTailReferenceCount = _internalRelease(output)
        fusionTail = output
        
      case .binary(let binary):
        let (input1, input2, output) = (binary.input1, binary.input2, binary.output)
        var restartingFusion = true
        if input1 == fusionTail || input2 == fusionTail {
          // In the middle of an operation fusion.
          if fusion.dataGroup == binary.dataGroup {
            // Before fusing, factor in whether enough heads are available.
            if fusion.input4 == nil {
              restartingFusion = false
            }
          }
        } else {
          // At the start of an operation fusion. A previous fusion may already be in progress.
        }
        
        // Decrement each input's reference count.
        let referenceCount1 = input1.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        let referenceCount2 = input2.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        if Allocation.debugInfoEnabled {
          _internalReleaseSlowPath(input1, referenceCount1)
          _internalReleaseSlowPath(input2, referenceCount2)
        }
        if !restartingFusion {
          if input1 == fusionTail {
            restartingFusion = referenceCount1 > 0
          } else if input2 == fusionTail {
            restartingFusion = referenceCount2 > 0
          } else {
            fatalError("This should never happen.")
          }
        }
        
        // Check shape.
        var firstShapeMatch: AllocationHandle?
        var numMatches = 0
        var numOnes = 0
        for i in 0..<2 {
          let input = (i == 0) ? input1 : input2
          if input.shape.elementsEqual(output.shape) {
            if firstShapeMatch == nil {
              firstShapeMatch = input
            }
            numMatches += 1
          } else if input.dataType.stride == input.byteCount {
            numOnes += 1
          }
        }
        guard let firstShapeMatch = firstShapeMatch,
              (numMatches + numOnes == 2) else {
          preconditionFailure("""
            Use explicit broadcast if one binary input needs broadcasting and is not a single \
            scalar.
            """)
        }
        
        // Restart operation fusion.
        if restartingFusion {
          let shouldTryRemoval = graph.shouldTryRemoval
          if fusion.input1 != nil {
            appendOperationFusion()
          }
          if shouldTryRemoval && switchContext(
            .init(input1, referenceCount1), .init(input2, referenceCount2),
            dataGroup: binary.dataGroup, availableHeads: 1) {
            // Switched context.
          } else {
            let dataType = firstShapeMatch.dataType
            let byteCount = firstShapeMatch.byteCount
            fusion.input1 = input1.reference!.takeUnretainedValue()
            fusion.size = dataType.contiguousSize(byteCount: byteCount)
            fusion.dataGroup = binary.dataGroup
          }
        }
        
        // Ensure operands are in correct registers.
        if fusionTail == nil {
          // No tail exists yet; beginning of operation fusion.
          fusion.input2 = input2.reference!.takeUnretainedValue()
        } else {
          // Take special care to avoid bugs when input1 == input2, or either input has already been
          // read. Previous tail is always register 1, fetch other input from device RAM.
          guard let fusionTail = fusionTail else {
            preconditionFailure("Fusion tail was absent when compiling binary operation.")
          }
          
          if input1 == fusionTail {
            let newHead = input2.reference!.takeUnretainedValue()
            if fusion.input2 == nil {
              fusion.input2 = newHead
            } else if fusion.input3 == nil {
              // RHS: reg3 -> reg2
              fusion.input3 = newHead
              fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            } else /*fusion.input4 == nil*/ {
              // RHS: reg4 -> reg2
              fusion.input4 = newHead
              fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_4.rawValue)
            }
          } else if input2 == fusionTail {
            let newHead = input1.reference!.takeUnretainedValue()
            if fusion.input2 == nil {
              fusion.input2 = newHead
            } else if fusion.input3 == nil {
              // LHS: reg3 -> reg2
              fusion.input3 = newHead
              fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            } else /*fusion.input4 == nil*/ {
              // LHS: reg3 -> reg2
              fusion.input4 = newHead
              fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_4.rawValue)
            }
            
            // LHS: reg2 -> reg1
            // RHS: reg1 -> reg2
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
          } else {
            fatalError("This should never happen.")
          }
        }
        
        // Append operation.
        fusion.operations.append(1000 + binary.operation)
        if let metadata = binary.metadata {
          fusion.metadata.append(metadata)
        }
        fusion.numFusedNonUnaryOperations += 1
        
        // Release inputs.
        if referenceCount1 == 0 {
          let reference = input1.reference!
          input1.reference = nil
          reference.release()
        }
        if referenceCount2 == 0 {
          let reference = input2.reference!
          input2.reference = nil
          reference.release()
        }
        
        // Update fusion tail.
        fusionTailReferenceCount = _internalRelease(output)
        fusionTail = output
        
      case .ternary(let ternary):
        let (input1, input2, input3) = (ternary.input1, ternary.input2, ternary.input3)
        let output = ternary.output
        var restartingFusion = true
        if input1 == fusionTail || input2 == fusionTail || input3 == fusionTail {
          // In the middle of an operation fusion.
          if fusion.dataGroup == ternary.dataGroup {
            // Before fusing, factor in whether enough heads are available.
            if fusion.input3 == nil {
              precondition(fusion.input4 == nil, "Input 3 is missing.")
              restartingFusion = false
            }
          } else {
            // At the start of an operation fusion. A previous fusion may already be in progress.
          }
        }
        
        // Decrement each input's reference count.
        let referenceCount1 = input1.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        let referenceCount2 = input2.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        let referenceCount3 = input3.referenceCount.wrappingDecrementThenLoad(ordering: .relaxed)
        if Allocation.debugInfoEnabled {
          _internalReleaseSlowPath(input1, referenceCount1)
          _internalReleaseSlowPath(input2, referenceCount2)
          _internalReleaseSlowPath(input3, referenceCount3)
        }
        if !restartingFusion {
          if input1 == fusionTail {
            restartingFusion = referenceCount1 > 0
          } else if input2 == fusionTail {
            restartingFusion = referenceCount2 > 0
          } else if input3 == fusionTail {
            restartingFusion = referenceCount3 > 0
          } else {
            fatalError("This should never happen.")
          }
        }
        
        var firstShapeMatch: AllocationHandle?
        var numMatches = 0
        var numOnes = 0
        for i in 0..<3 {
          var input: AllocationHandle
          if i == 0 {
            input = input1
          } else if i == 1 {
            input = input2
          } else /*i == 2*/ {
            input = input3
          }
          if input.shape.elementsEqual(output.shape) {
            if firstShapeMatch == nil {
              firstShapeMatch = input
            }
            numMatches += 1
          } else if input.dataType.stride == input.byteCount {
            numOnes += 1
          }
        }
        guard let firstShapeMatch = firstShapeMatch,
              (numMatches + numOnes == 3) else {
          preconditionFailure("""
            Use explicit broadcast if one ternary input needs broadcasting and is not a single \
            scalar.
            """)
        }
        // Restart operation fusion.
        if restartingFusion {
          let shouldTryRemoval = graph.shouldTryRemoval
          if fusion.input1 != nil {
            appendOperationFusion()
          }
          if shouldTryRemoval && switchContext(
            .init(input1, referenceCount1), .init(input2, referenceCount2),
            .init(input3, referenceCount3), dataGroup: ternary.dataGroup, availableHeads: 2) {
            // Switched context.
          } else {
            let dataType = firstShapeMatch.dataType
            let byteCount = firstShapeMatch.byteCount
            fusion.input1 = input1.reference!.takeUnretainedValue()
            fusion.size = dataType.contiguousSize(byteCount: byteCount)
            fusion.dataGroup = ternary.dataGroup
          }
        }
        
        // Ensure operands are in correct registers.
        if fusionTail == nil {
          // No tail exists yet; beginning of operation fusion.
          fusion.input2 = input2.reference!.takeUnretainedValue()
          fusion.input3 = input3.reference!.takeUnretainedValue()
        } else {
          // Take special care to avoid bugs when input1 == input2, or either input has already been
          // read. Previous tail is always register 1, fetch other input from device RAM.
          guard let fusionTail = fusionTail else {
            preconditionFailure("Fusion tail was absent when compiling binary operation.")
          }
          
          // The register that the current tail will transfer to.
          var tailRegister: Int
          var newHead1: Allocation
          var newHead2: Allocation
          if input1 == fusionTail {
            tailRegister = 1
            newHead1 = input2.reference!.takeUnretainedValue()
            newHead2 = input3.reference!.takeUnretainedValue()
          } else if input2 == fusionTail  {
            tailRegister = 2
            newHead1 = input1.reference!.takeUnretainedValue()
            newHead2 = input3.reference!.takeUnretainedValue()
          } else if input3 == fusionTail {
            tailRegister = 3
            newHead1 = input1.reference!.takeUnretainedValue()
            newHead2 = input2.reference!.takeUnretainedValue()
          } else {
            fatalError("This should never happen.")
          }
          
          if fusion.input2 == nil {
            precondition(fusion.input3 == nil, "Input 2 is missing.")
            fusion.input2 = newHead1
            fusion.input3 = newHead2
          } else {
            precondition(fusion.input3 == nil, "This should never happen.")
            precondition(fusion.input4 == nil, "Input 3 is missing.")
            fusion.input3 = newHead1
            fusion.input4 = newHead2
            
            // newHead1: reg3 -> reg2
            // undefined: reg2 -> reg3
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            
            // newHead2: reg4 -> reg3
            // undefined: reg3 -> reg4
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_3_4.rawValue)
          }
          
          if tailRegister == 1 {
            // Nothing to do.
          } else if tailRegister == 2 {
            // input1: reg2 -> reg1
            // input2: reg1 -> reg2
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
          } else /*tailRegister == 3*/ {
            // input1: reg2 -> reg1
            // input3: reg1 -> reg2
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
            
            // input2: reg3 -> reg2
            // input3: reg2 -> reg3
            fusion.operations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
          }
        }
        
        // Append operation.
        fusion.operations.append(2000 + ternary.operation)
        if let metadata = ternary.metadata {
          fusion.metadata.append(metadata)
        }
        fusion.numFusedNonUnaryOperations += 1
        
        // Release inputs.
        if referenceCount1 == 0 {
          let reference = input1.reference!
          input1.reference = nil
          reference.release()
        }
        if referenceCount2 == 0 {
          let reference = input2.reference!
          input2.reference = nil
          reference.release()
        }
        if referenceCount3 == 0 {
          let reference = input3.reference!
          input3.reference = nil
          reference.release()
        }
        
        // Update fusion tail.
        fusionTailReferenceCount = _internalRelease(output)
        fusionTail = output
        
      case .explicitCopy(let explicitCopy):
        if fusion.input1 != nil {
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
        graph.append(explicitCopy)
      }
    }
    
    // Finish compilation and return the compiled operations.
    if fusion.input1 != nil {
      appendOperationFusion()
    }
    return graph.finish()
    
    // Referring to the source code above:
    //
    // Here's how non-adjacent operation fusion works. When there is no source tail, look back at
    // the history of ops. Find a compatible one with a refcount=1 tail, then pull it out of the
    // list. This search can happens after releasing the operation's inputs, so the head's refcount
    // (0) does not match the tail's cached refcount (1).
    //
    // If the user deallocates a tensor while compiling, it could only monotonically decrement the
    // refcount. It could make a non-fusable tail fusable, but not the other way around. There is a
    // very low chance this happens during compilation. So, the tail's last-checked refcount may be
    // cached.
    //
    // Additionally, create a cache containing indices of all elements with 1-refcount tails. That
    // makes the search faster, although worst-case complexity may still be O(n^2).
    //
    // This idea is much simpler than alteratives. You don't have to rework the register swaps
    // because it's like you time-traveled to before the matching instruction ended. Append new
    // register allocations to what the matching fusion already allocated. However, I must ensure
    // fusions can be packed and unpacked efficiently. Replace the removed instruction with a
    // placeholder (`nil`), otherwise the array will rearrange its elements upon each removal
    // (O(n^2)). The instruction must be re-inserted at the end of the list to preserve temporal
    // order of execution; one of its heads might come from another instruction during this
    // compilation.
  }
}
