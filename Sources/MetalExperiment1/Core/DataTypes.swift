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
enum DataType: UInt16, CustomStringConvertible {
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
  
  init(tensorFlowDataType: TF_DataType) {
    switch tensorFlowDataType {
    case TF_HALF:
      self = .float16
    case TF_FLOAT:
      self = .float32
    case TF_BOOL:
      self = .bool
    case TF_INT8:
      self = .int8
    case TF_INT16:
      self = .int16
    case TF_INT32:
      self = .int32
    case TF_INT64:
      self = .int64
    case TF_UINT8:
      self = .uint8
    case TF_UINT16:
      self = .uint16
    case TF_UINT32:
      self = .uint32
    case TF_UINT64:
      self = .uint64
    default:
      fatalError("Did not recognize 'TF_DataType' with raw value '\(tensorFlowDataType)'.")
    }
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
  
  // Used in multiple places; the getter is likely a function call. Inlining optimizations for some
  // other members of `DataType` assume `stride` is a function call.
  //
  // TODO: Make this extremely fast by doing tricks with bits. Changes this from a function call to
  // a few inlined assembly instructions.
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
  var representableByInt64: Bool {
    rawValue >= 2 && rawValue <= 9
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

public typealias TF_DataType = Int32
public let TF_FLOAT: TF_DataType = 1
public let TF_DOUBLE: TF_DataType = 2
public let TF_INT32: TF_DataType = 3
public let TF_UINT8: TF_DataType = 4
public let TF_INT16: TF_DataType = 5
public let TF_INT8: TF_DataType = 6
public let TF_STRING: TF_DataType = 7
public let TF_COMPLEX64: TF_DataType = 8
public let TF_COMPLEX: TF_DataType = 8
public let TF_INT64: TF_DataType = 9
public let TF_BOOL: TF_DataType = 10
public let TF_QINT8: TF_DataType = 11
public let TF_QUINT8: TF_DataType = 12
public let TF_QINT32: TF_DataType = 13
public let TF_BFLOAT16: TF_DataType = 14
public let TF_QINT16: TF_DataType = 15
public let TF_QUINT16: TF_DataType = 16
public let TF_UINT16: TF_DataType = 17
public let TF_COMPLEX128: TF_DataType = 18
public let TF_HALF: TF_DataType = 19
public let TF_RESOURCE: TF_DataType = 20
public let TF_VARIANT: TF_DataType = 21
public let TF_UINT32: TF_DataType = 22
public let TF_UINT64: TF_DataType = 23
