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

import Atomics

public protocol PluggableDevice: AnyObject {
  func allocateTensor(
    _ type: Any.Type,
    _ shape: UnsafeBufferPointer<Int>
  ) -> OpaquePointer
  
  // Only call this once. Initializing a tensor multiple times results in undefined behavior,
  // possibly a runtime crash.
  func initializeTensor(
    _ handle: OpaquePointer,
    _ contentsInitializer: (UnsafeMutableRawBufferPointer) -> Void)
  
  // Allocates and initializes the tensor in a single function call, halving the overhead of calling
  // into the backend. Use this instead of calling `allocateTensor` and `initializeTensor`
  // separately whenever possible.
  func createTensor(
    _ type: Any.Type,
    _ shape: UnsafeBufferPointer<Int>,
    _ contentsInitializer: (UnsafeMutableRawBufferPointer) -> Void
  ) -> OpaquePointer
  
  func readTensor(
    _ handle: OpaquePointer,
    _ body: (UnsafeRawBufferPointer) -> Void)
  
  // Making a function like `readTensor` that modifies the underlying storage is technically
  // possible. It takes less time than a separate `readTensor` + `initializeTensor` on a discrete
  // GPU, and is relatively instantaneous on an integrated GPU. However, no frontend currently
  // requires this feature.
//  func modifyTensor(
//    _ body: (UnsafeRawBufferPointer) -> Void)
  
  // This is not a substitute for hardware features that transfer data between accelerators, such as
  // Infinity Fabric Link. Do not use this method to wrap any such features.
  func copyTensor(
    _ type: Any.Type,
    _ handle: OpaquePointer,
    _ source: any PluggableDevice
  ) -> OpaquePointer
  
  func deleteTensor(
    _ handle: OpaquePointer)
  
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

public typealias PluggableDeviceHandle = OpaquePointer

public struct PluggableDeviceTensorHandle {
  public var baseAddress: UnsafeMutablePointer<Int>
  
  @inlinable
  public init(_ handle: OpaquePointer) {
    baseAddress = UnsafeMutablePointer(handle)
  }
  
  @inlinable
  public var referenceCount: UnsafeAtomic<Int> {
    UnsafeAtomic(at: UnsafeMutablePointer(OpaquePointer(baseAddress)))
  }
  
  @inlinable
  public var pluggableDeviceHandle: PluggableDeviceHandle {
    PluggableDeviceHandle(bitPattern: baseAddress[2])!
  }
  
  @inlinable
  public var byteCount: Int {
    baseAddress[4]
  }
  
  @inlinable
  public var rank: Int {
    baseAddress[5]
  }
  
  @inlinable
  public var shape: UnsafeBufferPointer<Int> {
    UnsafeBufferPointer(start: baseAddress + 6, count: rank)
  }
}

extension PluggableDevice {
  // TODO: Remove `type` parameter and encode data types with TF_DataType (shift internal `DataType`
  // raw value to an offset of 4 bytes).
  @_disfavoredOverload
  @inlinable
  public func copyTensor(
    _ type: Any.Type,
    _ handle: OpaquePointer,
    _ source: any PluggableDevice
  ) -> OpaquePointer {
    // To avoid creating a synchronization deadlock with the source, first extract the data onto
    // the CPU.
    let byteCount = PluggableDeviceTensorHandle(handle).byteCount
    let tensorData: UnsafeMutableRawBufferPointer = .allocate(byteCount: byteCount, alignment: 1)
    defer {
      tensorData.deallocate()
    }
    source.readTensor(handle) { temporaryBuffer in
      tensorData.copyMemory(from: temporaryBuffer)
    }
    
    // Next, create a tensor on `self` that matches the extracted data.
    let shape = PluggableDeviceTensorHandle(handle).shape
    let alias = self.createTensor(type, shape) { uninitializedMemory in
      uninitializedMemory.copyMemory(from: UnsafeRawBufferPointer(tensorData))
    }
    return alias
  }
  
  @inlinable
  public func releaseTensor(
    _ handle: OpaquePointer
  ) {
    let referenceCount = PluggableDeviceTensorHandle(handle).referenceCount
    if referenceCount.wrappingDecrementThenLoad(ordering: .relaxed) == 0 {
      self.deleteTensor(handle)
    }
  }
  
  @inlinable
  public var handle: PluggableDeviceHandle {
    PluggableDeviceHandle(Unmanaged.passUnretained(self).toOpaque())
  }
}
