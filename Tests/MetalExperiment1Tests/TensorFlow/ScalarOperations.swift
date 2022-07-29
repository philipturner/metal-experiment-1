//
//  ScalarOperations.swift
//  
//
//  Created by Philip Turner on 7/29/22.
//

import MetalExperiment1

extension Tensor where Scalar: Numeric {
  public static var zero: Tensor {
    var zero = Tensor(repeating: 0, shape: [])
    zero._isScalarZero = true
    return zero
  }
  
  @inlinable
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    if lhs._isScalarZero {
      return rhs
    } else if rhs._isScalarZero {
      return lhs
    }
    return _Raw.addV2(lhs, rhs)
  }
  
  @inlinable
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    if rhs._isScalarZero {
      return lhs
    }
    return _Raw.sub(lhs, rhs)
  }
}

extension Tensor where Scalar: Numeric {
  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  public static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
    return _Raw.scalarAdd(rhs, scalar: lhs)
  }

  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  public static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
    return _Raw.scalarAdd(lhs, scalar: rhs)
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference.
  @inlinable
  public static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
    return _Raw.scalarSubInverse(rhs, scalar: lhs)
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference
  @inlinable
  public static func - (lhs: Tensor, rhs: Scalar) -> Tensor {
    return _Raw.scalarSub(lhs, scalar: rhs)
  }

  /// Adds two tensors and stores the result in the left-hand-side variable.
  /// - Note: `+=` supports broadcasting.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs + rhs
  }

  /// Adds the scalar to every scalar of the tensor and stores the result in the left-hand-side
  /// variable.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs + rhs
  }

  /// Subtracts the second tensor from the first and stores the result in the left-hand-side
  /// variable.
  /// - Note: `-=` supports broadcasting.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs - rhs
  }

  /// Returns the tensor produced by multiplying the two tensors.
  /// - Note: `*` supports broadcasting.
  @inlinable
  public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.mul(lhs, rhs)
  }

  /// Returns the tensor by multiplying it with every scalar of the tensor.
  @inlinable
  public static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
    return _Raw.scalarMul(rhs, scalar: lhs)
  }

  /// Multiplies the scalar with every scalar of the tensor and produces the product.
  @inlinable
  public static func * (lhs: Tensor, rhs: Scalar) -> Tensor {
    return _Raw.scalarMul(lhs, scalar: rhs)
  }

  /// Multiplies two tensors and stores the result in the left-hand-side variable.
  /// - Note: `*=` supports broadcasting.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs * rhs
  }

  /// Multiplies the tensor with the scalar, broadcasting the scalar, and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs * rhs
  }

  /// Returns the quotient of dividing the first tensor by the second.
  /// - Note: `/` supports broadcasting.
  @inlinable
  public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.div(lhs, rhs)
  }

  /// Returns the quotient of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  public static func / (lhs: Scalar, rhs: Tensor) -> Tensor {
    return _Raw.scalarDivInverse(rhs, scalar: lhs)
  }

  /// Returns the quotient of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  public static func / (lhs: Tensor, rhs: Scalar) -> Tensor {
    return _Raw.scalarDiv(lhs, scalar: rhs)
  }

  /// Divides the first tensor by the second and stores the quotient in the left-hand-side
  /// variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs / rhs
  }

  /// Divides the tensor by the scalar, broadcasting the scalar, and stores the quotient in the
  /// left-hand-side variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs / rhs
  }

  /// Returns the remainder of dividing the first tensor by the second.
  /// - Note: `%` supports broadcasting.
  @inlinable
  public static func % (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.mod(lhs, rhs)
  }

  /// Returns the remainder of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs % Tensor(repeating: rhs, shape: [])
  }

  /// Returns the remainder of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(repeating: lhs, shape: []) % rhs
  }

  /// Divides the first tensor by the second and stores the remainder in the left-hand-side
  /// variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs % rhs
  }

  /// Divides the tensor by the scalar and stores the remainder in the left-hand-side variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs % rhs
  }
}
