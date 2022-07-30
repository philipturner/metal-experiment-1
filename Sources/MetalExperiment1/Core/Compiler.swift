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
    var graph = Graph(eagerOperationCount: eagerOperations.count)
    
    // TODO: Function that searches the transient instruction list for a match, returning one if it
    // exists. Arguments are the current instruction's inputs.
    
    var fusionOperations: SmallVector<SIMD8<UInt16>> = .init()
    var fusionMetadata: SmallVector<SIMD2<UInt64>> = .init()
    var fusionDataGroup: DataGroup?
    var fusionHeadAllocation1: Allocation?
    var fusionHeadAllocation2: Allocation?
    var fusionHeadAllocation3: Allocation?
    var fusionHeadAllocation4: Allocation?
    var fusionTailReferenceCount: Int = -9999
    var fusionTail: AllocationHandle?
    var fusionSize: Int = -9999
    var numFusedUnaryOperations: Int = 0
    var numFusedNonUnaryOperations: Int = 0
    
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
        fusionHeadAllocation4 = nil
        fusionTailReferenceCount = -9999
        fusionTail = nil
        fusionSize = -9999
        numFusedUnaryOperations = 0
        numFusedNonUnaryOperations = 0
      }
      func noMissingHeads() -> Bool {
        if fusionHeadAllocation2 == nil {
          if fusionHeadAllocation3 != nil {
            return false
          }
        } else if fusionHeadAllocation3 == nil {
          if fusionHeadAllocation4 != nil {
            return false
          }
        }
        return true
      }
      guard let fusionHeadAllocation1 = fusionHeadAllocation1,
            let fusionTail = fusionTail,
            let fusionDataGroup = fusionDataGroup,
            fusionTailReferenceCount >= 0,
            fusionSize >= 0,
            noMissingHeads() else {
        fatalError("Something went wrong while fusing operations.")
      }
      
      // The frontend will never read this operation's results, so abort it. This may make
      // benchmarks difficult, because not every dispatched operation actually executes.
      //
      // When fusing non-adjacent operation chains, the reference count is no longer a 100% reliable
      // way to abort unused operations. Instead, analyze the graph and trace back unused ends. If
      // an "end" fusion chain ends with a zombie (zero-refcount) tensor, the zombie-ness transfers
      // to anything that fuses with it.
      if fusionTailReferenceCount == 0 {
        // TODO: If one of the heads matched another instruction's tail with cached refcount = 2,
        // decrement that reference count. It might now be available for fusion. This update will
        // not catch all cases where an instruction becomes fusable. For example, the fusion match
        // may encode before the zombie operation chain. To fuse, the zombie chain must go first.
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
        input4: fusionHeadAllocation4,
        output: fusionTailAllocation,
        size: fusionSize)
      graph.append(elementwise, tailReferenceCount: fusionTailReferenceCount)
      if _slowPath(Allocation.debugInfoEnabled || Context.profilingEncoding) {
        let numFusedOperations = numFusedUnaryOperations + numFusedNonUnaryOperations
        if numFusedOperations >= 2 {
          print("""
              Fused \(numFusedOperations) operations (\(numFusedUnaryOperations) unary, \
            \(numFusedNonUnaryOperations) non-unary)
            """)
        } else if numFusedUnaryOperations == 1 {
          precondition(numFusedNonUnaryOperations == 0)
          print("  Appended single unary operation")
        } else if numFusedNonUnaryOperations == 1 {
          precondition(numFusedUnaryOperations == 0)
          print("  Appended single non-unary operation")
        } else {
          print("  Appended copying operation")
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
        if unary.operation == .max {
          // Skip no-ops.
        } else {
          fusionOperations.append(unary.operation)
          if let metadata = unary.metadata {
            fusionMetadata.append(metadata)
          }
        }
        numFusedUnaryOperations += 1
        
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
          if fusionDataGroup! == binary.dataGroup {
            // Before fusing, factor in whether enough heads are available.
            if fusionHeadAllocation4 == nil {
              restartingFusion = false
            }
          }
          
          // The tail was the output of previous operation. Something initialized by the frontend
          // can't be an operation's output; only an input. Since the check may incur ARC overhead,
          // only perform it in debug mode.
          assert(!fusionTail!.reference!.takeUnretainedValue().initialized)
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
          if fusionDataGroup != nil {
            appendOperationFusion()
          }
          fusionHeadAllocation1 = input1.reference!.takeUnretainedValue()
          fusionSize = firstShapeMatch.dataType.contiguousSize(byteCount: firstShapeMatch.byteCount)
          fusionDataGroup = binary.dataGroup
        }
        
        // Ensure operands are in correct registers.
        if fusionTail == nil {
          // No tail exists yet; beginning of operation fusion.
          fusionHeadAllocation2 = input2.reference!.takeUnretainedValue()
        } else {
          // Take special care to avoid bugs when input1 == input2, or either input has already been
          // read. Previous tail is always register 1, fetch other input from device RAM.
          guard let fusionTail = fusionTail else {
            preconditionFailure("Fusion tail was absent when compiling binary operation.")
          }
          
          if input1 == fusionTail {
            let newHead = input2.reference!.takeUnretainedValue()
            if fusionHeadAllocation2 == nil {
              fusionHeadAllocation2 = newHead
            } else if fusionHeadAllocation3 == nil {
              // RHS: reg3 -> reg2
              fusionHeadAllocation3 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            } else {
              // RHS: reg4 -> reg2
              fusionHeadAllocation4 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_4.rawValue)
            }
          } else if input2 == fusionTail {
            let newHead = input1.reference!.takeUnretainedValue()
            if fusionHeadAllocation2 == nil {
              fusionHeadAllocation2 = newHead
            } else if fusionHeadAllocation3 == nil {
              // LHS: reg3 -> reg2
              fusionHeadAllocation3 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            } else {
              // LHS: reg3 -> reg2
              fusionHeadAllocation4 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_4.rawValue)
            }
            
            // LHS: reg2 -> reg1
            // RHS: reg1 -> reg2
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
          } else {
            fatalError("This should never happen.")
          }
        }
        
        // Append operation.
        fusionOperations.append(1000 + binary.operation)
        if let metadata = binary.metadata {
          fusionMetadata.append(metadata)
        }
        numFusedNonUnaryOperations += 1
        
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
          if fusionDataGroup! == ternary.dataGroup {
            // Before fusing, factor in whether enough heads are available.
            if fusionHeadAllocation3 == nil {
              precondition(fusionHeadAllocation4 == nil, "Input 3 is missing.")
              restartingFusion = false
            }
          }
          
          // The tail was the output of previous operation. Something initialized by the frontend
          // can't be an operation's output; only an input. Since the check may incur ARC overhead,
          // only perform it in debug mode.
          assert(!fusionTail!.reference!.takeUnretainedValue().initialized)
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
          if fusionDataGroup != nil {
            appendOperationFusion()
          }
          fusionHeadAllocation1 = input1.reference!.takeUnretainedValue()
          fusionSize = firstShapeMatch.dataType.contiguousSize(byteCount: firstShapeMatch.byteCount)
          fusionDataGroup = ternary.dataGroup
        }
        
        // Ensure operands are in correct registers.
        if fusionTail == nil {
          // No tail exists yet; beginning of operation fusion.
          fusionHeadAllocation2 = input2.reference!.takeUnretainedValue()
          fusionHeadAllocation3 = input3.reference!.takeUnretainedValue()
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
          
          if fusionHeadAllocation2 == nil {
            precondition(fusionHeadAllocation3 == nil, "Input 2 is missing.")
            fusionHeadAllocation2 = newHead1
            fusionHeadAllocation3 = newHead2
          } else {
            precondition(fusionHeadAllocation3 == nil, "This should never happen.")
            precondition(fusionHeadAllocation4 == nil, "Input 3 is missing.")
            fusionHeadAllocation3 = newHead1
            fusionHeadAllocation4 = newHead2
            
            // newHead1: reg3 -> reg2
            // undefined: reg2 -> reg3
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            
            // newHead2: reg4 -> reg3
            // undefined: reg3 -> reg4
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_3_4.rawValue)
          }
          
          if tailRegister == 1 {
            // Nothing to do.
          } else if tailRegister == 2 {
            // input1: reg2 -> reg1
            // input2: reg1 -> reg2
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
          } else /*tailRegister == 3*/ {
            // input1: reg2 -> reg1
            // input3: reg1 -> reg2
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_1_2.rawValue)
            
            // input2: reg3 -> reg2
            // input3: reg2 -> reg3
            fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
          }
        }
        
        // Append operation.
        fusionOperations.append(2000 + ternary.operation)
        if let metadata = ternary.metadata {
          fusionMetadata.append(metadata)
        }
        numFusedNonUnaryOperations += 1
        
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
        graph.append(explicitCopy)
      }
    }
    
    // Finish compilation and return the compiled operations.
    if fusionDataGroup != nil {
      appendOperationFusion()
    }
    
    // TODO: Implement non-adjacent operation fusion.
    // - Use the Swift `swap(_:_:)` function to potentally avoid ARC overhead when extracting an
    //   instruction from the list.
    // - Erase comments at the top of this function, erase bullet points on the decision to use AI
    //   in the command stream.
    
    // Referring to all the source code above:
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
    //
    // Do not compress the list before sending it off to the command stream. It is faster to scan
    // over placeholders and not encode them. This is why `compileEagerOperations` returns an array
    // of optional values.
    //
    // In the worst case, two fusable unary operation streams are interlaced. It switches contexts
    // after each operation. Assuming `maxCommandsPerBatch == 128`, it ends with 127 null values and
    // 2 non-null values. To reduce the overhead of lookup, periodically compress the list when:
    // - (1) At least 8 elements are null
    // - AND
    // - (2) Null elements make up at least 1/4 of the list.
    //
    // Thus, the compiler should keep track of how many null elements exist at a given moment.
    
    // TODO: This can actually happen if the last fusion is a zombie. Leave the precondition here
    // until I create a test for it, then transform into something that peels back the list's end.
    // The final product should either (a) end with a valid instruction or (b) have zero length.
    precondition(!graph.endsWithPlaceholder, """
      Last instruction should never be a placeholder. That breaks the command stream's iteration \
      mechanism.
      """)
    return graph.instructions
  }
}

extension Instruction.Elementwise {
  static var enableDump = false
  
  func dump() -> String {
    func dumpRead(register: Int) -> String {
      "var reg\(register) = input\(register)[i]"
    }
    
    func dumpWrite() -> String {
      "output[i] = reg1"
    }
    
    func dumpUnary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = UnaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = UnaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1)"
    }
    
    func dumpBinary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = BinaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = BinaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1, reg2)"
    }
    
    func dumpTernary(code: UInt16) -> String {
      var operationDesc: String
      if self.dataGroup == .f32_i32 {
        let unary = TernaryOperationType(rawValue: code)!
        operationDesc = String(describing: unary)
      } else {
        let unary = TernaryOperationType2(rawValue: code)!
        operationDesc = String(describing: unary)
      }
      return "reg1 = \(operationDesc)(reg1, reg2, reg3)"
    }
    
    func dumpSwap(code: UInt16) -> String {
      let swapType = RegisterSwapType(rawValue: code)!
      switch swapType {
      case .swap_registers_1_2:
        return "swap(&reg1, &reg2)"
      case .swap_registers_1_3:
        return "swap(&reg1, &reg3)"
      case .swap_registers_1_4:
        return "swap(&reg1, &reg4)"
      case .swap_registers_2_3:
        return "swap(&reg2, &reg3)"
      case .swap_registers_2_4:
        return "swap(&reg2, &reg4)"
      case .swap_registers_3_4:
        return "swap(&reg3, &reg4)"
      }
    }
    
    var output: [String] = []
    output.append(dumpRead(register: 1))
    if input2 != nil {
      output.append(dumpRead(register: 2))
    }
    if input3 != nil {
      output.append(dumpRead(register: 3))
    }
    if input4 != nil {
      output.append(dumpRead(register: 4))
    }
    
    for i in 0..<operations.count {
      let code = operations[i]
      if code < 1000 {
        output.append(dumpUnary(code: code - 0))
      } else if code < 2000 {
        output.append(dumpBinary(code: code - 1000))
      } else if code < 3000 {
        output.append(dumpTernary(code: code - 2000))
      } else if code < 4000 {
        output.append(dumpSwap(code: code - 3000))
      }
    }
    output.append(dumpWrite())
    return output.joined(separator: "\n")
  }
}

struct Graph {
  private(set) var instructions: [Instruction?]
  private var cache: [CacheElement]?
  
  @inline(__always)
  init(eagerOperationCount: Int) {
    self.instructions = []
    self.instructions.reserveCapacity(eagerOperationCount)
  }
  
  struct CacheElement {
    var handle: AllocationHandle
    var instructionIndex: Int
  }
  
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

extension Graph {
  @inline(__always)
  mutating func append(_ elementwise: Instruction.Elementwise, tailReferenceCount: Int) {
    instructions.append(.elementwise(elementwise))
  }
  
  @inline(__always)
  mutating func append(_ explicitCopy: Instruction.ExplicitCopy) {
    instructions.append(.explicitCopy(explicitCopy))
  }
  
  var endsWithPlaceholder: Bool {
    instructions.count > 0 && instructions.last! == nil
  }
}
