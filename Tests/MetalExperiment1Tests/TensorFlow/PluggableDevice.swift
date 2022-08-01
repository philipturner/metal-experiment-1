//
//  PluggableDevice.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

import Atomics

public typealias PluggableDeviceHandle = OpaquePointer

public protocol PluggableDevice: AnyObject {
  func createTensor(
    _ type: Any.Type,
    _ shape: UnsafeBufferPointer<Int>,
    _ body: (UnsafeMutableRawBufferPointer) -> Void
  ) -> OpaquePointer
  
  func readTensor(
    _ handle: OpaquePointer,
    _ body: (UnsafeRawBufferPointer) -> Void)
  
  func deleteTensor(
    _ handle: OpaquePointer)
  
  func executeOperation(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>)
}

extension PluggableDevice {
  func releaseTensor(_ handle: OpaquePointer) {
    let referenceCount = PluggableDeviceTensorHandle(handle).referenceCount
    if referenceCount.wrappingDecrementThenLoad(ordering: .relaxed) == 0 {
      deleteTensor(handle)
    }
  }
  
  var handle: PluggableDeviceHandle {
    PluggableDeviceHandle(Unmanaged.passUnretained(self).toOpaque())
  }
}

public struct PluggableDeviceTensorHandle {
  public var baseAddress: UnsafeMutablePointer<Int>
  
  public init(_ handle: OpaquePointer) {
    baseAddress = UnsafeMutablePointer(handle)
  }
  
  public var referenceCount: UnsafeAtomic<Int> {
    UnsafeAtomic(at: UnsafeMutablePointer(OpaquePointer(baseAddress)))
  }
  
  public var pluggableDeviceHandle: PluggableDeviceHandle {
    PluggableDeviceHandle(bitPattern: baseAddress[2])!
  }
  
  public var byteCount: Int {
    baseAddress[4]
  }
  
  public var rank: Int {
    baseAddress[5]
  }
  
  public var shape: UnsafeBufferPointer<Int> {
    UnsafeBufferPointer(start: baseAddress + 6, count: rank)
  }
}