//
//  File.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

import Atomics
import MetalExperiment1

public final class _ExecutionContext {
  /// Global context storing all available devices, loaded functions, etc.
  public static let global: _ExecutionContext = _ExecutionContext()
  
  /// List of devices available to this execution context.
  /// See documentation for `withDevice(_:)` to learn about devices.
  var devices: [PluggableDeviceHandle: PluggableDevice] = [:]
  
  /// The mutex for preventing potential concurrent access to `devices`.
  var devicesMutex: Mutex = Mutex()
  
  @usableFromInline let defaultDevice: PluggableDevice?
  
  @usableFromInline let defaultDeviceHandle: PluggableDeviceHandle?
  
  /// Tracks the size of every thread's device stack. When it reaches zero, do not query the
  /// thread-local device stack, which greatly increases overhead of accessing the current device.
  /// This atomic only needs to be synchronized with respect to the calling thread.
  @usableFromInline let allDeviceStacksSize: UnsafeAtomic<Int> = .create(0)
  
  /// Initializes a new execution context by initializing available devices.
  @usableFromInline
  init() {
    #if canImport(Metal)
    self.defaultDevice = MTLPluggableDevice.default
    self.defaultDeviceHandle = defaultDevice!.handle
    #elseif canImport(OpenCL)
    fatalError("OpenCL backend not implemented.")
    #else
    // For other backends, use `withDevice(PluggableDevice)` to set the device.
    #endif
  }
  
  @inlinable @inline(never)
  func execute(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeMutableBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    let targetDeviceHandle = self.currentDeviceHandle
    var misplacedTensorExists = false
    for input in inputs {
      let deviceHandle = PluggableDeviceTensorHandle(input).pluggableDeviceHandle
      if deviceHandle != targetDeviceHandle {
        misplacedTensorExists = true
      }
    }
    
    if !misplacedTensorExists {
      let targetDevice = self.getDevice(handle: targetDeviceHandle)
      targetDevice.executeOperation(name, attributes, .init(inputs), outputs)
    } else {
      // If an input is on the wrong device, swap it with a temporary tensor on the correct device.
      // This is extremely costly, but should never happen in an average use case.
      executeSlowPath(name, attributes, inputs, outputs)
    }
  }
  
  @usableFromInline @inline(never)
  func executeSlowPath(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeMutableBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    let dstDevice = self.currentDevice
    let dstDeviceHandle = dstDevice.handle
    var temporaryTensorHandles: [OpaquePointer] = []
    temporaryTensorHandles.reserveCapacity(inputs.count)
    defer {
      for newTensorHandle in temporaryTensorHandles {
        dstDevice.releaseTensor(newTensorHandle)
      }
    }
    
    for i in 0..<inputs.count {
      let oldTensorHandle = inputs[i]
      let deviceHandle = PluggableDeviceTensorHandle(oldTensorHandle).pluggableDeviceHandle
      if deviceHandle != dstDeviceHandle {
        let srcDevice = self.getDevice(handle: deviceHandle)
        let newTensorHandle = dstDevice.copyTensor(oldTensorHandle, srcDevice)
        inputs[i] = newTensorHandle
        temporaryTensorHandles.append(newTensorHandle)
      }
    }
    dstDevice.executeOperation(name, attributes, .init(inputs), outputs)
    
    fatalError("This shouldn't happen yet.")
  }
}

extension _ExecutionContext {
  @inlinable @inline(__always)
  var currentDeviceHandle: PluggableDeviceHandle {
    if allDeviceStacksSize.load(ordering: .relaxed) == 0,
       let defaultDeviceHandle = defaultDeviceHandle {
      return defaultDeviceHandle
    } else {
      return currentDeviceHandleSlowPath
    }
  }
  
  @usableFromInline @inline(never)
  var currentDeviceHandleSlowPath: PluggableDeviceHandle {
    if let device = _ThreadLocalState.local.deviceScopes._currentDevice {
      return device.handle
    } else if let defaultDeviceHandle = defaultDeviceHandle {
      return defaultDeviceHandle
    } else {
      fatalError("No default device was set.")
    }
  }
  
  @inlinable @inline(__always)
  var currentDevice: PluggableDevice {
    if allDeviceStacksSize.load(ordering: .relaxed) == 0,
       let defaultDevice = defaultDevice {
      return defaultDevice
    } else {
      return currentDeviceSlowPath
    }
  }
  
  @usableFromInline @inline(never)
  var currentDeviceSlowPath: PluggableDevice {
    if let device = _ThreadLocalState.local.deviceScopes._currentDevice {
      return device
    } else if let defaultDevice = defaultDevice {
      return defaultDevice
    } else {
      fatalError("No default device was set.")
    }
  }
  
  @inlinable @inline(__always)
  func getDevice(handle: PluggableDeviceHandle) -> PluggableDevice {
    if handle == defaultDeviceHandle {
      return defaultDevice.unsafelyUnwrapped
    } else {
      return getDeviceSlowPath(handle: handle)
    }
  }
  
  @usableFromInline @inline(never)
  func getDeviceSlowPath(handle: PluggableDeviceHandle) -> PluggableDevice {
    precondition(devicesMutex.acquire() == 0)
    let device = devices[handle]
    precondition(devicesMutex.release() == 0)
    
    guard let device = device else {
      fatalError("Device with handle \(handle) not found.")
    }
    return device
  }
}

/// A class to keep around thread local state:
///  - DeviceScopes
///  - LazyTensorContext
class _ThreadLocalState {
  var deviceScopes = DeviceScopes()

  private static let key: ThreadLocalStorage.Key =
    ThreadLocalStorage.Key {
      #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        Unmanaged<AnyObject>.fromOpaque($0).release()
      #else
        Unmanaged<AnyObject>.fromOpaque($0!).release()
      #endif
    }

  @usableFromInline
  static var local: _ThreadLocalState {
    if let state = ThreadLocalStorage.get(for: key) {
      return Unmanaged.fromOpaque(state).takeUnretainedValue()
    }

    let state = _ThreadLocalState()
    ThreadLocalStorage.set(
      value: Unmanaged.passRetained(state).toOpaque(),
      for: key)
    return state
  }
}

/// Stack of devices that models nested calls to withDevice/withDefaultDevice. Devices are
/// represented by their names in TensorFlow notation. See documentation for
/// `withDevice(named:perform:)` to learn about device names.
///
/// All TensorFlow operations will be put on the topmost device on the stack. When the stack is
/// empty or the topmost device is `nil`, that allows TensorFlow to place operations on any device
/// that it sees fit.
@usableFromInline
struct DeviceScopes {
  var deviceStack: [PluggableDevice?] = []

  var _currentDevice: PluggableDevice? {
    return deviceStack.last ?? nil
  }

  @usableFromInline
  mutating func pushDevice(_ device: PluggableDevice?) {
    if let handle = device?.handle {
      let ctx = _ExecutionContext.global
      if handle != ctx.defaultDeviceHandle {
        _ExecutionContext.global.allDeviceStacksSize.wrappingIncrement(ordering: .relaxed)
        precondition(ctx.devicesMutex.acquire() == 0)
        ctx.devices[handle] = device
        precondition(ctx.devicesMutex.release() == 0)
      }
    }
    deviceStack.append(device)
  }

  @usableFromInline
  mutating func popDevice() {
    let lastDevice = deviceStack.popLast()
    precondition(deviceStack.popLast() != nil)
    if lastDevice??.handle != _ExecutionContext.global.defaultDeviceHandle {
      _ExecutionContext.global.allDeviceStacksSize.wrappingDecrement(ordering: .relaxed)
    }
  }
}
