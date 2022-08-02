//
//  PluggableDevice.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

// The official header for the PluggableDevice backend API. This facilitates virtual function calls
// into PluggableDevice instances and provides API extensions deemed necessary for properly
// interacting with backends. It is designed to be API/ABI-stable, guaranteeing backward
// compatibility with any framework that adopts the interface.
//
// This is intentionally named after the TensorFlow "PluggableDevice" mechanism for supporting
// non-Nvidia GPUs. It shares no relation to the TensorFlow feature other than name, and does not
// interact with the feature internally. This allows for training models on platforms not supported
// by TensorFlow (e.g. iOS, which can only perform model inference) and for lower overhead by
// directly calling into user-defined code.
//
// Usage:
//
// Include this file in the frontend's Swift module, not the backend's module. The frontend should
// declare each backend's conformance to `PluggableDevice`.

import Atomics

// Process all backends in the form of `any PluggableDevice`, instead of resolving the concrete
// type. The frontend may violate this rule when automatically setting a default device. For
// example, it may set the default to a Metal-based backend on Apple platforms, and an OpenCL-based
// backend on non-Apple platforms.
//
// The frontend may also resolve concrete types to provide vendor-specific optimizations. If such
// optimizations create a new frontend API that can be reasonably implemented with vanilla
// PluggableDevice functionality, the frontend must support the API on all platforms.

// The protocol members below must be thread-safe and synchronized with a mutex lock.
public protocol PluggableDevice: AnyObject {
  // Release the returned tensor handle when done using it.
  func createTensor(
    _ dataType: TF_DataType,
    _ shape: UnsafeBufferPointer<Int>,
    _ contentsInitializer: (UnsafeMutableRawBufferPointer) -> Void
  ) -> OpaquePointer
  
  // `mutatingContents` determines whether to copy the data back to the accelerator. This is
  // blocking, so avoid mutating contents directly if an asynchronous operation for doing so exists.
  func readTensor(
    _ handle: OpaquePointer,
    _ mutatingContents: Bool,
    _ body: (UnsafeMutableRawBufferPointer) -> Void)
  
  // Avoid using this directly; instead use `releaseTensor` when deinitializing an object that wraps
  // a tensor handle.
  func deleteTensor(
    _ handle: OpaquePointer)
  
  // Prefer creating a `StaticString` and passing that as `name`. This provides quick access to a
  // C-style string, which bridges to `UnsafeRawBufferPointer`. Furthermore, it eliminates overhead
  // of creating and reference-counting `String` objects.
  //
  // Rules for encoding attributes:
  //
  // Atoms of data are padded to 16 bytes. For strings and arrays, encode an `UnsafeBufferPointer`
  // to their data. This rule applies recursively with arrays of strings, arrays of arrays, etc.
  // After the first level of recursion, store elements in their native layout stride.
  func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>)
}

extension PluggableDevice {
  // This is not a substitute for hardware features that transfer data between accelerators, such as
  // Infinity Fabric Link. `copyTensor` has massive overhead, so avoid using it unless absolutely
  // necessary. Release the returned tensor handle when done using it.
  public func copyTensor(
    _ handle: OpaquePointer,
    _ source: PluggableDevice
  ) -> OpaquePointer {
    // To avoid creating a synchronization deadlock with the source (which could be `self`), first
    // extract the data onto the CPU.
    let byteCount = PluggableDeviceTensorHandle(handle).byteCount
    let tensorData: UnsafeMutableRawBufferPointer = .allocate(byteCount: byteCount, alignment: 1)
    defer {
      tensorData.deallocate()
    }
    source.readTensor(handle, false) { temporaryBuffer in
      tensorData.copyMemory(from: UnsafeRawBufferPointer(temporaryBuffer))
    }
    
    // Next, create a tensor on `self` that matches the extracted data.
    let dataType = PluggableDeviceTensorHandle(handle).dataType
    let shape = PluggableDeviceTensorHandle(handle).shape
    let alias = self.createTensor(dataType, shape) { uninitializedMemory in
      uninitializedMemory.copyMemory(from: UnsafeRawBufferPointer(tensorData))
    }
    return alias
  }
  
  @inlinable @inline(__always)
  public func releaseTensor(
    _ handle: OpaquePointer
  ) {
    let referenceCount = PluggableDeviceTensorHandle(handle).referenceCount
    if referenceCount.wrappingDecrementThenLoad(ordering: .relaxed) == 0 {
      self.deleteTensor(handle)
    }
  }
  
  // A unique identifier for the backend. Use this to cache references to the backend, reducing ARC
  // overhead of passing around the PluggableDevice object. Never use this unmanaged reference to
  // reconstruct the object.
  @inlinable @inline(__always)
  public var handle: PluggableDeviceHandle {
    PluggableDeviceHandle(Unmanaged.passUnretained(self).toOpaque())
  }
}

public typealias PluggableDeviceHandle = OpaquePointer

//===------------------------------------------------------------------------------------------===//
// Tensor Handle
//===------------------------------------------------------------------------------------------===//

public struct PluggableDeviceTensorHandle {
  public var baseAddress: UnsafeMutablePointer<Int>
  
  @inlinable @inline(__always)
  public init(_ handle: OpaquePointer) {
    baseAddress = UnsafeMutablePointer(handle)
  }
  
  @inlinable @inline(__always)
  public var referenceCount: UnsafeAtomic<Int> {
    UnsafeAtomic(at: UnsafeMutablePointer(OpaquePointer(baseAddress)))
  }
  
  @inlinable @inline(__always)
  public var pluggableDeviceHandle: PluggableDeviceHandle {
    PluggableDeviceHandle(bitPattern: baseAddress[2])!
  }
  
  @inlinable @inline(__always)
  public var dataType: TF_DataType {
    // Only process the lower 4 bytes. The upper 4 bytes are undefined.
    Int32(truncatingIfNeeded: baseAddress[3])
  }
  
  @inlinable @inline(__always)
  public var byteCount: Int {
    baseAddress[4]
  }
  
  @inlinable @inline(__always)
  public var rank: Int {
    baseAddress[5]
  }
  
  @inlinable @inline(__always)
  public var shape: UnsafeBufferPointer<Int> {
    UnsafeBufferPointer(start: baseAddress + 6, count: rank)
  }
}

// The backend does not need to support every data type; unsupported ones should crash at runtime.
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
