//
//  Context.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

// Renamed "FrontendContext" to avoid name collision with type "Context" that currently exists in
// the backend.

#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
  import Darwin
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

import Atomics

/// A value that indicates the phase of using a machine learning model.
public enum LearningPhase {
  case training
  case inference
}

/// A context that stores thread-local contextual information used by deep learning APIs such as
/// layers.
///
/// Use `Context.local` to retrieve the current thread-local context.
///
/// Examples:
///
/// * Set the current learning phase to training so that layers like `BatchNorm` will
///   compute mean and variance when applied to inputs.
///
///   ```swift
///   Context.local.learningPhase = .training
///   ```
/// * Set the current learning phase to inference so that layers like `Dropout` will not drop out
///   units when applied to inputs.
///
///   ```swift
///   Context.local.learningPhase = .inference
///   ```
public struct FrontendContext {
  /// The learning phase.
  public var learningPhase: LearningPhase = .inference

  internal var globalTensorCount: Int = 0
  
  static var trackTensorCount: Bool = false

  /// Creates a context with default properties.
  public init() {}

  /// The current thread-local context.
  ///
  /// - Note: Accessing this property is thread-safe.
  public static var local: FrontendContext {
    _read { yield ContextManager.local.currentContext }
    _modify { yield &ContextManager.local.currentContext }
  }
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given learning phase.
///
/// - Parameters:
///   - context: A context that will be set before the closure gets called and restored after the
///     closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withContext(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withContext<R>(_ context: FrontendContext, _ body: () throws -> R) rethrows -> R {
  ContextManager.local.push(context)
  defer { ContextManager.local.popContext() }
  return try body()
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given learning phase.
///
/// - Parameters:
///   - learningPhase: A learning phase that will be set before the closure gets called and restored
///     after the closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withLearningPhase(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withLearningPhase<R>(
  _ learningPhase: LearningPhase,
  _ body: () throws -> R
) rethrows -> R {
  var context = ContextManager.local.currentContext
  context.learningPhase = learningPhase
  return try withContext(context, body)
}

/// A manager that maintains and provides safe access to thread-local `Context` values.
private final class ContextManager {
  var contextStack: [FrontendContext] = [FrontendContext()]

  /// The data key for the singleton `Context` in the current thread.
  static let key: ThreadLocalStorage.Key =
    ThreadLocalStorage.Key {
      #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        Unmanaged<ContextManager>.fromOpaque($0).release()
      #else
        Unmanaged<ContextManager>.fromOpaque($0!).release()
      #endif
    }

  /// The thread-local singleton.
  static var local: ContextManager {
    if let address = ThreadLocalStorage.get(for: key) {
      return Unmanaged<ContextManager>.fromOpaque(address)
        .takeUnretainedValue()
    }

    let context = ContextManager()
    ThreadLocalStorage.set(
      value: Unmanaged.passRetained(context).toOpaque(),
      for: key)
    return context
  }

  /// Pushes the given context to the context stack.
  func push(_ context: FrontendContext) {
    contextStack.append(context)
  }

  /// Pops a context out of a stack.
  ///
  /// - Precondition: The context stack must contain more than `1` contexts.
  func popContext() {
    assert(
      contextStack.count > 1,
      "Internal error: Only 1 context is available. Popping is not allowed.")
    contextStack.removeLast()
  }

  /// The most recent context.
  var currentContext: FrontendContext {
    _read {
      assert(!contextStack.isEmpty, "Internal error: No contexts exist.")
      yield contextStack[contextStack.endIndex - 1]
    }
    _modify {
      assert(!contextStack.isEmpty, "Internal error: No contexts exist.")
      yield &contextStack[contextStack.endIndex - 1]
    }
  }
}
