//
//  TensorOperations.swift
//  
//
//  Created by Philip Turner on 7/21/22.
//

extension Tensor where Scalar: Numeric {
  @inlinable
  public func incremented() -> Tensor {
    return _Raw.increment(self)
  }
}

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
  _Raw.elu(x)
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
