//
//  DataTypes.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

import MetalExperiment1

public struct TensorDataType: Equatable {
  public var _cDataType: TF_DataType

  @usableFromInline
  internal init(_ cDataType: TF_DataType) {
    self._cDataType = cDataType
  }
}

public protocol _TensorFlowDataTypeCompatible {
  @inlinable
  static var tensorFlowDataType: TensorDataType { get }
}

public protocol TensorFlowScalar: _TensorFlowDataTypeCompatible {}

public typealias TensorFlowNumeric = TensorFlowScalar & Numeric
public typealias TensorFlowSignedNumeric = TensorFlowScalar & SignedNumeric
public typealias TensorFlowInteger = TensorFlowScalar & BinaryInteger

public protocol TensorFlowFloatingPoint:
  TensorFlowScalar & BinaryFloatingPoint
where
  Self.RawSignificand: FixedWidthInteger
{}

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension Float16: TensorFlowFloatingPoint {}
#endif
extension Float: TensorFlowFloatingPoint {}

extension Bool: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_BOOL)
  }
}

extension Int8: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_INT8)
  }
}

extension Int16: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_INT16)
  }
}

extension Int32: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_INT32)
  }
}

extension Int64: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_INT64)
  }
}

extension UInt8: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_UINT8)
  }
}

extension UInt16: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_UINT16)
  }
}

extension UInt32: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_UINT32)
  }
}

extension UInt64: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_UINT64)
  }
}

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension Float16: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_HALF)
  }
}
#endif

extension Float: TensorFlowScalar {
  @inlinable
  public static var tensorFlowDataType: TensorDataType {
    return TensorDataType(TF_FLOAT)
  }
}
