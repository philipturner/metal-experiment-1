//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

// Using `UInt8` instead of `UInt16` to fit as many operations as possible into a `TypeList16`.
enum UnaryOperationType: UInt8, CaseIterable {
  case increment_f32 = 70
  case increment_i32 = 71
}

// Ordered by relative frequency, minimizing the number of conditional checks during compilation and
// encoding.
enum EagerOperation {
  struct Unary {
    var operation: UnaryOperationType
    var input: UInt64
    var output: UInt64
  }
  case unary(Unary)
  
  struct ExplicitCopy {
    var input: UInt64
    var output: UInt64
  }
  case explicitCopy(ExplicitCopy)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct MultiUnary {
    // `metadata` much less vector capacity of `operations`. It doesn't need as much storage because
    // it's serialized efficiently. Metadata is only recorded after each operation that needs it.
    var operations: TypeList16<UnaryOperationType>
    
    // Warning: `SIMD2` does not mean 2 operations worth of metadata. It means the total capacity
    // for metadata is 16, which happens to be (2 operations) * (8 bytes/operation). The rationing
    // of metadata per operation is subject to change.
    var metadata: TypeListStorage<SIMD2<Int>>
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  case multiUnary(MultiUnary)
  
  struct ExplicitCopy {
    var input: Allocation
    var output: Allocation
    var byteCount: Int
  }
  case explicitCopy(ExplicitCopy)
}
