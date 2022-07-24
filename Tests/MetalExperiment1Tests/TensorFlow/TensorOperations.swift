//
//  TensorOperations.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

extension Tensor where Scalar: Numeric {
  @inlinable
  public func incremented() -> Tensor {
    return _Raw.scalarAdd(self, rhs: 1)
  }
}

// Unary

@inlinable
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.abs(x)
}

@inlinable
public func acos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.acos(x)
}

@inlinable
public func acosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.acosh(x)
}

@inlinable
public func asin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.asin(x)
}

@inlinable
public func asinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.asinh(x)
}

@inlinable
public func atan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.atan(x)
}

@inlinable
public func atanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.atanh(x)
}



@inlinable
public func ceil<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.ceil(x)
}

@inlinable
public func cos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.cos(x)
}

@inlinable
public func cosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.cosh(x)
}

@inlinable
public func elu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.elu(features: x)
}

@inlinable
public func exp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.exp(x)
}

@inlinable
public func expm1<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.expm1(x)
}

@inlinable
public func floor<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.floor(x)
}



extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable public var isFinite: Tensor<Bool> { _Raw.isFinite(self) }
  
  @inlinable public var isInfinite: Tensor<Bool> { _Raw.isInf(self) }
  
  @inlinable public var isNaN: Tensor<Bool> { _Raw.isNan(self) }
}



@inlinable
public func leakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  alpha: Double = 0.2
) -> Tensor<T> {
  _Raw.leakyRelu(features: x, alpha: alpha)
}

@inlinable
public func log<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.log(x)
}

@inlinable
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.log1p(x)
}

extension Tensor where Scalar == Bool {
  @inlinable
  public func elementsLogicalNot() -> Tensor {
    return _Raw.logicalNot(self)
  }
}

extension Tensor where Scalar: SignedNumeric {
  @inlinable
  public static prefix func - (rhs: Tensor) -> Tensor {
    return _Raw.neg(rhs)
  }
}

@inlinable
public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.relu(features: x)
}

@inlinable
public func relu6<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.relu6(features: x)
}

@inlinable
public func round<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.round(x)
}



@inlinable
public func rsqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.rsqrt(x)
}

@inlinable
public func selu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.selu(features: x)
}

@inlinable
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sigmoid(x)
}

@inlinable
public func sign<T: Numeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sign(x)
}

@inlinable
public func sin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sin(x)
}

@inlinable
public func sinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sinh(x)
}

@inlinable
public func softplus<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.softplus(features: x)
}



@inlinable
public func softsign<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.softsign(features: x)
}

@inlinable
public func sqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.sqrt(x)
}

@inlinable
public func square<T: Numeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.square(x)
}

@inlinable
public func tan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.tan(x)
}

@inlinable
public func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.tanh(x)
}
