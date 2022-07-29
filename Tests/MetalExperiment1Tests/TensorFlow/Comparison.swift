//
//  Comparison.swift
//  
//
//  Created by Philip Turner on 7/29/22.
//

import MetalExperiment1

infix operator .<: ComparisonPrecedence
infix operator .<=: ComparisonPrecedence
infix operator .>=: ComparisonPrecedence
infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

extension Tensor where Scalar: Numeric & Comparable {
  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  @inlinable
  public static func .< (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.less(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  @inlinable
  public static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.lessEqual(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  @inlinable
  public static func .> (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greater(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  @inlinable
  public static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greaterEqual(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  /// - Note: `.<` supports broadcasting.
  @inlinable
  public static func .< (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.less(Tensor(repeating: lhs, shape: []), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  /// - Note: `.<=` supports broadcasting.
  @inlinable
  public static func .<= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.lessEqual(Tensor(repeating: lhs, shape: []), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  /// - Note: `.>` supports broadcasting.
  @inlinable
  public static func .> (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greater(Tensor(repeating: lhs, shape: []), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  /// - Note: `.>=` supports broadcasting.
  @inlinable
  public static func .>= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greaterEqual(Tensor(repeating: lhs, shape: []), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  /// - Note: `.<` supports broadcasting.
  @inlinable
  public static func .< (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.less(lhs, Tensor(repeating: rhs, shape: []))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  /// - Note: `.<=` supports broadcasting.
  @inlinable
  public static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.lessEqual(lhs, Tensor(repeating: rhs, shape: []))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  /// - Note: `.>` supports broadcasting.
  @inlinable
  public static func .> (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.greater(lhs, Tensor(repeating: rhs, shape: []))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  /// - Note: `.>=` supports broadcasting.
  @inlinable
  public static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.greaterEqual(lhs, Tensor(repeating: rhs, shape: []))
  }
}

extension Tensor where Scalar: Equatable {
  /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
  /// - Note: `.==` supports broadcasting.
  @inlinable
  public static func .== (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.equal(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
  /// - Note: `.!=` supports broadcasting.
  @inlinable
  public static func .!= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.notEqual(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
  /// - Note: `.==` supports broadcasting.
  @inlinable
  public static func .== (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return Tensor(repeating: lhs, shape: []) .== rhs
  }

  /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
  /// - Note: `.!=` supports broadcasting.
  @inlinable
  public static func .!= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return Tensor(repeating: lhs, shape: []) .!= rhs
  }

  /// Returns a tensor of Boolean scalars by computing `lhs == rhs` element-wise.
  /// - Note: `.==` supports broadcasting.
  @inlinable
  public static func .== (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return lhs .== Tensor(repeating: rhs, shape: [])
  }

  /// Returns a tensor of Boolean scalars by computing `lhs != rhs` element-wise.
  /// - Note: `.!=` supports broadcasting.
  @inlinable
  public static func .!= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return lhs .!= Tensor(repeating: rhs, shape: [])
  }
}

// TODO: infix operator ≈: ComparisonPrecedence

extension Tensor where Scalar: TensorFlowFloatingPoint & Equatable {
  /// Returns a tensor of Boolean values indicating whether the elements of `self` are
  /// approximately equal to those of `other`.
  /// - Precondition: `self` and `other` must be of the same shape.
  @inlinable
  public func elementsAlmostEqual(
    _ other: Tensor,
    tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
  ) -> Tensor<Bool> {
    return _Raw.approximateEqual(self, other, tolerance: Double(tolerance))
  }
}

//extension Tensor where Scalar: TensorFlowFloatingPoint {
//  /// Returns `true` if all elements of `self` are approximately equal to those of `other`.
//  /// - Precondition: `self` and `other` must be of the same shape.
//  @inlinable
//  public func isAlmostEqual(
//    to other: Tensor,
//    tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
//  ) -> Bool {
//    elementsAlmostEqual(other, tolerance: tolerance).all()
//  }
//}
