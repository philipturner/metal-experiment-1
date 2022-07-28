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
        fusionTailReferenceCount = -9999
        fusionTail = nil
        fusionSize = -9999
        numFusedUnaryOperations = 0
        numFusedNonUnaryOperations = 0
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
            if fusionHeadAllocation3 == nil {
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
            } else {
              // RHS: reg3 -> reg2
              fusionHeadAllocation3 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
            }
          } else if input2 == fusionTail {
            let newHead = input1.reference!.takeUnretainedValue()
            if fusionHeadAllocation2 == nil {
              fusionHeadAllocation2 = newHead
            } else {
              // LHS: reg3 -> reg2
              fusionHeadAllocation3 = newHead
              fusionOperations.append(3000 + RegisterSwapType.swap_registers_2_3.rawValue)
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
      case .swap_registers_2_3:
        return "swap(&reg2, &reg3)"
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
