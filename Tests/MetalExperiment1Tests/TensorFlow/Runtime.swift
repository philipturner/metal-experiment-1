//
//  File.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

import MetalExperiment1

public final class _ExecutionContext {
  /// Global context storing all available devices, loaded functions, etc.
  public static let global: _ExecutionContext = _ExecutionContext()
  
  /// Initializes a new execution context by initializing available devices.
  @usableFromInline
  init() {
    
  }
  
  /// Returns a valid TensorFlow device name, which corresponds to the closest enclosing call to
  /// one of the overloads of withDevice. A return value of `nil` indicates the absence of a
  /// withDevice call on the call stack or the presence of an immediately enclosing
  /// `withDefaultDevice(perform)` call.
  var currentDeviceName: String? {
    return _ThreadLocalState.local.deviceScopes._currentDevice
  }
  
  @usableFromInline
  @inline(never)
  static func eagerExecute(
    _ name: UnsafeRawBufferPointer,
    _ attributes: UnsafeRawBufferPointer,
    _ inputs: UnsafeBufferPointer<OpaquePointer>,
    _ outputs: UnsafeMutableBufferPointer<OpaquePointer>
  ) {
    // If an input is on the wrong device, swap it with a temporary tensor on the correct device.
    // This is extremely costly, but should never happen in an average use case. Use `malloc`
    // instead of `withUnsafeTemporaryAllocation` to create the alternative `input`.
    
//    for input in inputs {
//      _ = AllocationHandle(input).referenceCount
//    }
//    _ = _ExecutionContext.global.something
    Context.global.executeOperation(name, attributes, inputs, outputs)
  }
  
  static let somethingKey = ThreadLocalStorage.Key(destructor: nil)
  
  @usableFromInline
  internal var something: UnsafeMutableRawPointer? {
    ThreadLocalStorage.get(for: Self.somethingKey)
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
  var deviceStack: [String?] = []

  var _currentDevice: String? {
    return deviceStack.last ?? nil
  }

  @usableFromInline
  mutating func pushDevice(_ device: String?) {
    deviceStack.append(device)
  }

  @usableFromInline
  mutating func popDevice() {
    precondition(deviceStack.popLast() != nil)
  }
}
