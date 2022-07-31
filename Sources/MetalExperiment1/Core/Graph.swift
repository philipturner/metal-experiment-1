//
//  Graph.swift
//  
//
//  Created by Philip Turner on 7/30/22.
//

struct Graph {
  private(set) var instructions: [Instruction?]
  
  // Keep the cache's object around between compiler passes, avoiding the overhead of allocatng a
  // Swift object every pass.
  private static var cache: [AllocationHandle: Int] = [:]
  
  init(eagerOperationCount: Int) {
    self.instructions = []
    self.instructions.reserveCapacity(eagerOperationCount)
    Self.cache.removeAll(keepingCapacity: true)
  }
}

extension Graph {
  mutating func append(_ elementwise: Instruction.Elementwise, tailReferenceCount: Int) {
    instructions.append(.elementwise(elementwise))
    if tailReferenceCount == 1 {
      // The cache should never already contain the tail. This check is costly, so only perform it
      // in debug mode.
      assert(Self.cache[elementwise.output.handle] == nil, "Cache already contained tail.")
      Self.cache[elementwise.output.handle] = instructions.count - 1
    }
  }
  
  mutating func append(_ explicitCopy: Instruction.ExplicitCopy) {
    instructions.append(.explicitCopy(explicitCopy))
  }
  
  var endsWithPlaceholder: Bool {
    instructions.count > 0 && instructions.last! == nil
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
    if key1.referenceCount == 0,
       key2?.referenceCount == 0,
       key3?.referenceCount == 0 {
      return false
    }
    if Self.cache.isEmpty {
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
      case 2:
        guard let key3 = key3 else {
          continue
        }
        key = key3
      default:
        fatalError("This should never happen.")
      }
      
      guard key.referenceCount == 0,
            let match = Self.cache[key.handle] else {
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
      
      var canUseMatch: Bool
      if elementwise.dataGroup != dataGroup {
        canUseMatch = false
      } else if elementwise.input4 != nil {
        canUseMatch = availableHeads == 0
      } else if elementwise.input3 != nil {
        canUseMatch = availableHeads <= 1
      } else if elementwise.input2 != nil {
        canUseMatch = availableHeads <= 2
      } else {
        canUseMatch = availableHeads <= 3
      }
      
      if canUseMatch {
        // TODO: Mark the fusion tail as un-initialized.
        
        // TODO: Extract some of the removal logic into the enclosing function body. This lets me
        // unpack the elementwise operation into local variables inside an `@inline(never)`
        // function without either (a) duplicating many CPU instructions or (b) invoking two
        // function calls. Alternatively, the function call is permissible because the overhead is
        // dwarfed by calling into `removeSlowPath` and its calls to `Dictionary` subscript.
        //
        // Store the number of fused operations as a tiny piece of metadata somewhere inside the
        // elementwise instruction. It should be possible without changing the memory stride.
        return elementwise
      } else {
        // Insert element back into list.
        swap(&instructions[match], &instruction)
        return nil
      }
    }
    return nil
  }
}
