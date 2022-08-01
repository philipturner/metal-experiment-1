//
//  Graph.swift
//  
//
//  Created by Philip Turner on 7/30/22.
//

struct Graph {
  private var instructions: [Instruction?]
  private var cache: [AllocationHandle: Int]
  private(set) var showingDebugInfo: Bool
  
  init(eagerOperationCount: Int, showingDebugInfo: Bool) {
    self.instructions = []
    self.instructions.reserveCapacity(eagerOperationCount)
    self.cache = [:]
    self.showingDebugInfo = showingDebugInfo
  }
}

extension Graph {
  mutating func append(_ elementwise: Instruction.Elementwise, tailReferenceCount: Int) {
    let handle = elementwise.output.handle
    instructions.append(.elementwise(elementwise))
    if tailReferenceCount == 1 {
      // The cache should never already contain the tail. This check is costly, so only perform it
      // in debug mode.
      assert(cache[handle] == nil, "Cache already contained tail.")
      cache[handle] = instructions.count - 1
    }
  }
  
  mutating func append(_ explicitCopy: Instruction.ExplicitCopy) {
    instructions.append(.explicitCopy(explicitCopy))
  }
  
  var shouldTryRemoval: Bool {
    cache.count > 0
  }
  
  var endsWithPlaceholder: Bool {
    instructions.count > 0 && instructions.last! == nil
  }
  
  mutating func finish() -> [Instruction?] {
    // The last instruction should never be a placeholder. That breaks the command stream's
    // iteration mechanism. I could pop placeholders off the end until the last element is valid,
    // but that would cause problems when the system runs out of memory. If any placeholder exists,
    // it could spawn a no-op command buffer when the command stream subdivides the list of
    // instructions.
    instructions.removeAll(where: { $0 == nil })
    return instructions
  }
}

extension Graph {
  struct SearchKey {
    var handle: AllocationHandle
    var referenceCount: Int
    
    init(_ handle: AllocationHandle, _ referenceCount: Int) {
      self.handle = handle
      self.referenceCount = referenceCount
    }
  }
  
  @inline(__always)
  func shouldRemove(
    matching key1: SearchKey,
    _ key2: SearchKey?,
    _ key3: SearchKey?
  ) -> Bool {
    if key1.referenceCount > 0,
       (key2?.referenceCount ?? 1) > 0,
       (key3?.referenceCount ?? 1) > 0 {
      return false
    }
    if cache.isEmpty {
      return false
    }
    return true
  }
  
  // Wrap this in an `@inline(never)` nested function.
  mutating func remove(
    matching key1: SearchKey,
    _ key2: SearchKey?,
    _ key3: SearchKey?,
    dataGroup: DataGroup,
    availableHeads: Int
  ) -> Instruction.Elementwise? {
    let showingDebugInfo = Allocation.debugInfoEnabled || Context.profilingEncoding
    var candidate: Instruction.Elementwise?
    var candidateIndex = -1
    var candidateNumHeads = availableHeads - 1
    for i in 0..<3 {
      var key: SearchKey
      switch i {
      case 0:
        key = key1
      case 1:
        guard let key2 = key2 else {
          continue
        }
        key = key2
      default: /*2*/
        guard let key3 = key3 else {
          continue
        }
        key = key3
      }
      
      guard key.referenceCount == 0,
            let match = cache[key.handle] else {
        continue
      }
      
      // Using `swap` might avoid ARC overhead of extracting the instruction from the list.
      var instruction: Instruction? = /*placeholder*/nil
      swap(&instructions[match], &instruction)
      guard case .elementwise(let elementwise) = instruction else {
        // Do not pollute this function with assembly instructions* from the slow path. Extract the
        // description generation to something captured within the `fatalError`'s autoclosure.
        //
        // *I'm referring to CPU instructions as in x86, arm64, RISC-V - not the `instructions`
        // property of `Graph`.
        func getDescription() -> String {
          if let instruction = instruction {
            return String(describing: instruction)
          } else {
            return "nil"
          }
        }
        fatalError("Found non-elementwise instruction at index '\(match)': '\(getDescription())'")
      }
      
      var canUseMatch = false
      if elementwise.dataGroup == dataGroup {
        var numHeads: Int
        if elementwise.input4 != nil {
          numHeads = 0
        } else if elementwise.input3 != nil {
          numHeads = 1
        } else if elementwise.input2 != nil {
          numHeads = 2
        } else {
          numHeads = 3
        }
        if numHeads > candidateNumHeads {
          canUseMatch = true
          candidateNumHeads = numHeads
        }
      }
      
      if canUseMatch {
        if _slowPath(showingDebugInfo) {
          if candidateIndex == -1 {
            print("""
                Establishing candidate at index \(match).
              """)
          } else {
            print("""
                Replacing candidate at index \(candidateIndex) with candidate at index \(match).
              """)
          }
        }
        if candidateIndex != -1 {
          // Avoid ARC overhead of creating an extra reference
          func getCandidate() -> Instruction.Elementwise {
            var elementwise: Instruction.Elementwise?
            swap(&elementwise, &candidate)
            return elementwise!
          }
          var instruction: Optional = Instruction.elementwise(getCandidate())
          swap(&instructions[candidateIndex], &instruction)
          precondition(instruction == nil, "Did not replace candidate with placeholder.")
        }
        candidate = elementwise
        candidateIndex = match
      } else {
        // Insert element back into list.
        swap(&instructions[match], &instruction)
      }
    }
    candidate?.output.initialized = false
    return candidate
  }
}
