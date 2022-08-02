//
//  Execution.swift
//  
//
//  Created by Philip Turner on 8/2/22.
//

import MetalExperiment1

/// Executes a closure, making TensorFlow operations run on a specific device.
///
/// - Parameters:
///   - device: Device to execute operations on.
///   - body: A closure whose TensorFlow operations are to be executed on the
///     specified kind of device.
public func withDevice<R>(_ device: PluggableDevice, perform body: () throws -> R) rethrows -> R {
  return try _ExecutionContext.global.withDevice(device, perform: body)
}

/// Executes a closure, allowing TensorFlow to place TensorFlow operations on any device. This
/// should restore the default placement behavior.
///
/// - Parameters:
///   - body: A closure whose TensorFlow operations are to be executed on the specified kind of
///     device.
public func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R {
  return try _ExecutionContext.global.withDefaultDevice(perform: body)
}
