//
//  elementwise_f32_i32.swift
//  
//
//  Created by Philip Turner on 8/3/22.
//

import Darwin

// Swift analogue of the `elementwise_f32_i32` ubershader.
struct Swift_elementwise_f32_i32 {
  enum MemoryCast: UInt16 {
    case f32_i32_native = 0
    case f16_as_f32 = 1
    case i8_as_i32 = 2
    case i16_as_i32 = 3
    case u8_as_i32 = 4
    case u16_as_i32 = 5
    // `bool` can be masked as either `i8` or `u8`.
  }
  
  var read_layouts: SIMD3<UInt16>
  var read_memory_casts: SIMD3<MemoryCast.RawValue>
  var num_inputs: Int
  var write_memory_cast: MemoryCast
  var unary_operation_type: UnaryOperationType?
  var binary_operation_type: BinaryOperationType?
  var ternary_operation_type: TernaryOperationType?
  var metadata: UInt64
  var input1: UnsafeRawPointer?
  var input2: UnsafeRawPointer?
  var input3: UnsafeRawPointer?
  var output: UnsafeMutableRawPointer
  
  mutating func execute() {
    var register1: UInt32
    var register2: UInt32
    var register3: UInt32
    for i in 0..<num_inputs {
      var input: UnsafeRawPointer
      switch i {
      case 0:
        input = input1.unsafelyUnwrapped
      case 1:
        input = input2.unsafelyUnwrapped
      default: /*2*/
        input = input3.unsafelyUnwrapped
      }
      
      var compressed: UInt32
      switch read_layouts[i] {
      case 1:
        let rebound = input.assumingMemoryBound(to: UInt8.self)
        compressed = UInt32(rebound.pointee)
      case 2:
        let rebound = input.assumingMemoryBound(to: UInt16.self)
        compressed = UInt32(rebound.pointee)
      default: /*4*/
        let rebound = input.assumingMemoryBound(to: UInt32.self)
        compressed = UInt32(rebound.pointee)
      }
      
      var expanded: UInt32
      switch MemoryCast(rawValue: read_memory_casts[i]).unsafelyUnwrapped {
      case .f32_i32_native:
        expanded = compressed
      case .f16_as_f32:
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let `in` = Float16(bitPattern: .init(truncatingIfNeeded: compressed))
        expanded = Float(`in`).bitPattern
        #else
        fatalError("'Float16' not supported yet.")
        #endif
      case .i8_as_i32:
        let `in` = Int8(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .i16_as_i32:
        let `in` = Int16(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .u8_as_i32:
        let `in` = UInt8(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      case .u16_as_i32:
        let `in` = UInt16(truncatingIfNeeded: compressed)
        expanded = UInt32(bitPattern: Int32(`in`))
      }
    }
  }
}
