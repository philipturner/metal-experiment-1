//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

enum UnaryOperationType: UInt8, CaseIterable {
  case increment
}

// Ordered by relative frequency, minimizing the number of conditional checks during compilation and
// encoding.
enum EagerOperation {
  struct Unary {
    var type: UnaryOperationType
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
    // `dataTypes` has half the vector capacity of `types`. It doesn't need as much storage because
    // it's serialized efficiently. A new type is only recorded after each cast operation. When
    // encoding Metal commands, both lists expand to 2 bytes/element, mapping one-to-one with shader
    // loop iterations.
    var types: OperationTypeList16<UnaryOperationType>
    var dataTypes: OperationTypeList4<DataType>
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
