//
//  TensorShape.swift
//  
//
//  Created by Philip Turner on 7/16/22.
//

import MetalExperiment1

public struct TensorShape: ExpressibleByArrayLiteral {
  public var dimensions: [Int]

  @inlinable
  public init(_ dimensions: [Int]) {
    self.dimensions = dimensions
  }

  @inlinable
  public init<C: Collection>(_ dimensions: C) where C.Element == Int {
    self.dimensions = Array(dimensions)
  }

  @inlinable
  public init(arrayLiteral elements: Int...) {
    self.init(elements)
  }

  @inlinable
  public init(_ elements: Int...) {
    self.init(elements)
  }

  @inlinable
  public init(repeating repeatedValue: Int, count: Int) {
    self.init(Array(repeating: repeatedValue, count: count))
  }

  @inlinable
  public var rank: Int {
    return dimensions.count
  }

  @inlinable
  public var contiguousSize: Int {
    return dimensions.reduce(1, *)
  }
}

extension TensorShape: Collection, MutableCollection {
  public typealias Element = Int
  public typealias Index = Int
  public typealias Indices = Range<Int>

  @inlinable
  public var count: Int {
    return dimensions.count
  }

  @inlinable
  public var indices: Indices {
    return dimensions.indices.lowerBound..<dimensions.indices.upperBound
  }

  @inlinable
  public var startIndex: Index {
    return dimensions.startIndex
  }

  @inlinable
  public var endIndex: Index {
    return dimensions.endIndex
  }

  @inlinable
  public func index(after i: Index) -> Index {
    return dimensions.index(after: i)
  }

  @inlinable
  public subscript(position: Index) -> Element {
    _read { yield dimensions[position] }
    _modify { yield &dimensions[position] }
  }

  @inlinable
  public subscript(bounds: Range<Int>) -> TensorShape {
    get { return TensorShape(dimensions[bounds]) }
    set { dimensions[bounds] = ArraySlice(newValue.dimensions) }
  }
}

extension TensorShape: RandomAccessCollection {
  @inlinable
  public func index(_ i: Int, offsetBy distance: Int) -> Int {
    dimensions.index(i, offsetBy: distance)
  }

  @inlinable
  public func distance(from start: Int, to end: Int) -> Int {
    dimensions.distance(from: start, to: end)
  }
}

extension TensorShape: RangeReplaceableCollection {
  public typealias SubSequence = Self

  @inlinable
  public init() {
    self.init([])
  }

  @inlinable
  public mutating func append(_ newElement: Element) {
    dimensions.append(newElement)
  }

  @inlinable
  public mutating func append(contentsOf newElements: TensorShape) {
    dimensions.append(contentsOf: newElements.dimensions)
  }

  @inlinable
  public mutating func append<S: Sequence>(contentsOf newElements: S) where Element == S.Element {
    dimensions.append(contentsOf: newElements)
  }

  @inlinable
  public mutating func replaceSubrange<C>(
    _ subrange: Range<Index>, with newElements: C
  ) where C: Collection, Element == C.Element {
    dimensions.replaceSubrange(subrange, with: newElements)
  }
}

extension TensorShape: Equatable {
  @inlinable
  public static func == (lhs: TensorShape, rhs: TensorShape) -> Bool {
    return lhs.dimensions == rhs.dimensions
  }
}

extension TensorShape: Codable {
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(dimensions)
  }

  @inlinable
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let dimensions = try container.decode([Int].self)
    self.init(dimensions)
  }
}

extension TensorShape: CustomStringConvertible {
  public var description: String {
    return dimensions.description
  }
}
