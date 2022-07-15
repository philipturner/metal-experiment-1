//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

enum UnaryOperationType: UInt8, CaseIterable {
  case increment
}

enum EagerOperation {
  struct Unary {
    var type: UnaryOperationType
    var input: UInt64
    var output: UInt64
    var size: Int
  }
  
  case unary(Unary)
}

// Instead of manually extracting references to the individual buffers, this keeps references to the
// compiled operations until finishing. It indirectly stores references to the buffers, making it
// easier to implement and more performant.
enum CompiledOperation {
  struct MultiUnary {
    var input: Allocation
    var output: Allocation
    var size: Int
  }
  
  case multiUnary(MultiUnary)
}
