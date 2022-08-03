//
//  elementwise_u32_i64_u64.swift
//  
//
//  Created by Philip Turner on 8/3/22.
//

import Darwin

// Swift analogue of the `elementwise_u32_i64_u64` ubershader.
struct Swift_elementwise_u32_i64_u64 {
  enum MemoryCast: UInt16, RawRepresentable {
    case i64_u64_native = 0
    case i32_as_i64 = 1
    case i16_as_i64 = 2
    case i8_as_i64 = 3
    
    case u32_as_i64 = 4
    case u16_as_i64 = 5
    case u8_as_i64 = 6
    case f32_padded = 7
    case f16_as_f32_padded = 8
    
    @inline(__always)
    init(dataTypeRawValue: UInt16) {
      let dataType = DataType(rawValue: dataTypeRawValue)
      switch dataType.unsafelyUnwrapped {
      case .float16:
        self = .f16_as_f32_padded
      case .float32:
        self = .f32_padded
      case .bool:
        self = .u8_as_i64
      case .int8:
        self = .i8_as_i64
      case .int16:
        self = .i16_as_i64
      case .int32:
        self = .i32_as_i64
      case .int64:
        self = .i64_u64_native
      case .uint8:
        self = .u8_as_i64
      case .uint16:
        self = .u16_as_i64
      case .uint32:
        self = .u32_as_i64
      case .uint64:
        self = .i64_u64_native
      }
    }
    
    @inline(__always)
    var readSize: UInt16 {
      switch self {
      case .i64_u64_native: return 8
      case .i32_as_i64, .u32_as_i64: return 4
      case .i16_as_i64, .u16_as_i64: return 2
      case .i8_as_i64, .u8_as_i64: return 1
      case .f32_padded: return 4
      case .f16_as_f32_padded: return 2
      }
    }
  }
  
  mutating func execute() {
    
  }
}
