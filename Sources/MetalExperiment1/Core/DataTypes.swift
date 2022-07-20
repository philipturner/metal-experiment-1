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
enum DataType: UInt16, CaseIterable, CustomStringConvertible {
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
  
  // TODO: A way to initialize from `TF_DataType` raw values.
  
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
  
  // Used in multiple places; the getter is likely a function call. Inlining optimizations for some
  // other members of `DataType` assume `stride` is a function call.
  var stride: Int {
    switch self {
    case .float16:
      #if (os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64)
      return 2
      #else
      return MemoryLayout<Float16>.stride
      #endif
    case .float32:
      return MemoryLayout<Float>.stride
    case .bool:
      return MemoryLayout<Bool>.stride
    case .int8:
      return MemoryLayout<Int8>.stride
    case .int16:
      return MemoryLayout<Int16>.stride
    case .int32:
      return MemoryLayout<Int32>.stride
    case .int64:
      return MemoryLayout<Int64>.stride
    case .uint8:
      return MemoryLayout<UInt8>.stride
    case .uint16:
      return MemoryLayout<UInt16>.stride
    case .uint32:
      return MemoryLayout<UInt32>.stride
    case .uint64:
      return MemoryLayout<UInt64>.stride
    }
  }
  
  @inline(__always)
  var isFloatingPoint: Bool {
    rawValue <= 1
  }
  
  @inline(__always)
  var isSigned: Bool {
    rawValue >= 3 && rawValue <= 6
  }
  
  @inline(__always)
  var representableByInt32: Bool {
    (rawValue >= 2 && rawValue <= 5) ||
    (rawValue >= 7 && rawValue <= 8)
  }
  
  @inline(__always)
  func contiguousSize(byteCount: Int) -> Int {
    let stridePowerOf2 = self.stride.trailingZeroBitCount
    return byteCount >> stridePowerOf2
  }
  
  var description: String {
    switch self {
    case .float16:
      return "Float16"
    case .float32:
      return "Float"
    case .bool:
      return "Bool"
    case .int8:
      return "Int8"
    case .int16:
      return "Int16"
    case .int32:
      return "Int32"
    case .int64:
      return "Int64"
    case .uint8:
      return "UInt8"
    case .uint16:
      return "UInt16"
    case .uint32:
      return "UInt32"
    case .uint64:
      return "UInt64"
    }
  }
}

let TF_FLOAT: Int32 = 1
let TF_DOUBLE: Int32 = 2
let TF_INT32: Int32 = 3
let TF_UINT8: Int32 = 4
let TF_INT16: Int32 = 5
let TF_INT8: Int32 = 6
let TF_STRING: Int32 = 7
let TF_COMPLEX64: Int32 = 8
let TF_COMPLEX: Int32 = 8
let TF_INT64: Int32 = 9
let TF_BOOL: Int32 = 10
let TF_QINT8: Int32 = 11
let TF_QUINT8: Int32 = 12
let TF_QINT32: Int32 = 13
let TF_BFLOAT16: Int32 = 14
let TF_QINT16: Int32 = 15
let TF_QUINT16: Int32 = 16
let TF_UINT16: Int32 = 17
let TF_COMPLEX128: Int32 = 18
let TF_HALF: Int32 = 19
let TF_RESOURCE: Int32 = 20
let TF_VARIANT: Int32 = 21
let TF_UINT32: Int32 = 22
let TF_UINT64: Int32 = 23
