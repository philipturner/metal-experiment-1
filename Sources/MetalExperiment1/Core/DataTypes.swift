//
//  DataTypes.swift
//  
//
//  Created by Philip Turner on 7/16/22.
//

import MetalPerformanceShaders

// Use a custom enumeration instead of `MPSDataType`. GPU shaders read the raw value for dynamic
// typing. This has less cases than `MPSDataType` and is smaller (`UInt16` vs. `UInt32`), reducing
// GPU register usage and enabling other optimizations.
enum DataType: UInt16, CaseIterable {
  // Floating-point types
  case float16 = 0
  case float32 = 1
  
  // Integral types
  case bool = 2
  case int8 = 3
  case int16 = 4
  case int32 = 5
  case int64 = 6
  case uint8 = 7
  case uint16 = 8
  case uint32 = 9
  case uint64 = 10
  
  init(_ type: Any.Type) {
    // Check floating-point types first because they're the most common. This should reduce the
    // average number of comparisons performed.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    if type == Float16.self {
      self = .float16
      return
    }
    #endif
    if type == Float.self {
      self = .float32
      return
    }
    
    // Check integral types next.
    if type == Bool.self {
      self = .bool
      return
    }
    if type == Int8.self {
      self = .int8
      return
    }
    if type == Int16.self {
      self = .int16
      return
    }
    if type == Int32.self {
      self = .int32
      return
    }
    if type == Int64.self {
      self = .int64
      return
    }
    if type == UInt8.self {
      self = .uint8
      return
    }
    if type == UInt16.self {
      self = .uint16
      return
    }
    if type == UInt32.self {
      self = .uint32
      return
    }
    if type == UInt64.self {
      self = .uint64
      return
    }
    
    // Check `Float16` on Intel Macs last because the comparison is very costly. It creates and
    // deallocates a `String` object.
    #if (os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)
    if String(describing: type) == "Float16" {
      self = .float16
      return
    }
    #endif
    fatalError("Did not recognize data type '\(type)'.")
  }
  
  var mpsDataType: MPSDataType {
    switch self {
    case .float16:
      return .float16
    case .float32:
      return .float32
    case .bool:
      return .bool
    case .int8:
      return .int8
    case .int16:
      return .int16
    case .int32:
      return .int32
    case .int64:
      return .int64
    case .uint8:
      return .uInt8
    case .uint16:
      return .uInt16
    case .uint32:
      return .uInt32
    case .uint64:
      return .uInt64
    }
  }
}
