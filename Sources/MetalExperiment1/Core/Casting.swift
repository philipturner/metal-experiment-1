//
//  Casting.swift
//  
//
//  Created by Philip Turner on 7/24/22.
//

extension UnaryOperationType {
  // Do not enter u32/i64/u64 as either argument. Returns `nil` for no-ops.
  init?(casting input: DataType, to output: DataType, metadata: inout UInt64?) {
    precondition(!input.requiresLargeRepresentation, "Input required large representation.")
    precondition(!output.requiresLargeRepresentation, "Output required large representation.")
    if input.isFloatingPoint {
      switch output {
      case .float16:
        if input == .float16 {
          return nil
        }
        self = .cast_f32_to_f16
      case .float32:
        return nil
      case .bool:
        self = .cast_f32_to_bool
      default:
        var bounds: SIMD2<Int32>
        switch output {
        case .int8:
          bounds = .init(truncatingIfNeeded: SIMD2<Int8>(.min, .max))
        case .int16:
          bounds = .init(truncatingIfNeeded: SIMD2<Int16>(.min, .max))
        case .int32:
          bounds = .init(truncatingIfNeeded: SIMD2<Int32>(.min, .max))
        case .uint8:
          bounds = .init(truncatingIfNeeded: SIMD2<UInt8>(.min, .max))
        case .uint16:
          bounds = .init(truncatingIfNeeded: SIMD2<UInt16>(.min, .max))
        default:
          fatalError("This should never happen.")
        }
        metadata = unsafeBitCast(bounds, to: UInt64.self)
        self = .cast_f32_to_i32
      }
    } else {
      if input == .bool {
        guard output.isFloatingPoint else {
          return nil
        }
      } else if input.isSignedInteger == output.isSignedInteger {
        if input.rawValue <= output.rawValue {
          return nil
        }
      }
      
      switch output {
      case .float16:
        self = .cast_i32_to_f16
      case .float32:
        self = .cast_i32_to_f32
      case .bool:
        self = .cast_f32_to_bool
      default:
        var masks: SIMD2<Int32>
        switch output {
        case .int8:
          masks = .init(1 << 8 - 1, 1 << 7)
        case .int16:
          masks = .init(1 << 16 - 1, 1 << 15)
        case .int32:
          // No sign extension occurs when casting bool/u8/u16 to i32.
          return nil
        case .uint8:
          masks = .init(1 << 8 - 1, 0)
        case .uint16:
          masks = .init(1 << 16 - 1, 0)
        default:
          fatalError("This should never happen.")
        }
        metadata = unsafeBitCast(masks, to: UInt64.self)
        self = .cast_i32_to_i32
      }
    }
  }
}

extension UnaryOperationType2 {
  // One argument must be u32/i64/u64. Returns `nil` for no-ops.
  init?(casting input: DataType, to output: DataType, metadata: inout UInt64?) {
    precondition(
      input.requiresLargeRepresentation || output.requiresLargeRepresentation,
      "Neither input nor output required large representation.")
    if input.isFloatingPoint {
      switch output {
      case .uint32:
        self = .cast_f32_to_u32
      case .int64:
        self = .cast_f32_to_i64
      case .uint64:
        self = .cast_f32_to_u64
      default:
        fatalError("This should never happen.")
      }
    } else {
      if input == .bool {
        // Output requires large representation.
        return nil
      } else if input.isSignedInteger == output.isSignedInteger {
        if input.rawValue <= output.rawValue {
          return nil
        }
      } else if (input == .int64 && output == .uint64) ||
                (input == .uint64 && output == .int64) {
        return nil
      }
      
      switch output {
      case .float16:
        if input.representableByInt64 {
          self = .cast_i64_to_f16
        } else {
          self = .cast_u64_to_f16
        }
      case .float32:
        if input.representableByInt64 {
          self = .cast_i64_to_f32
        } else {
          self = .cast_u64_to_f32
        }
      case .bool:
        self = .cast_i64_to_bool
      default:
        // Sign mask determined in shader with `(mask ^ (mask >> 1))`.
        var mask: UInt64
        if output.isSignedInteger {
          switch output {
          case .int8:
            mask = 1 << 8 - 1
          case .int16:
            mask = 1 << 16 - 1
          case .int32:
            mask = 1 << 32 - 1
          case .int64:
            // No sign extension occurs when casting bool/u8/u16/u32 to i64.
            return nil
          default:
            fatalError("This should never happen.")
          }
          self = .cast_u64_to_i32
        } else {
          switch output {
          case .uint8:
            mask = 1 << 8 - 1
          case .uint16:
            mask = 1 << 16 - 1
          case .uint32:
            mask = 1 << 32 - 1
          case .uint64:
            // Responsible for sign-extending casts of i8/i16/i32 to u64.
            mask = 1 << 64 - 1
          default:
            fatalError("This should never happen.")
          }
          self = .cast_u64_to_u32
        }
        metadata = mask
      }
    }
  }
}
