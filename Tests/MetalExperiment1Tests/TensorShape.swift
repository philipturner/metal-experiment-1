//
//  TensorShape.swift
//  
//
//  Created by Philip Turner on 7/16/22.
//

import MetalExperiment1

public struct TensorShape: ExpressibleByArrayLiteral {
  @usableFromInline
  internal var storage: SIMD8<Int>
  
  // Accessing the dimensions this way brings seriously degraded performance.
  @inlinable
  public var dimensions: [Int] {
    get {
      let rank = storage[1]
      return Array(unsafeUninitializedCapacity: rank) { bufferPointer, count in
        count = rank
        let baseAddress = bufferPointer.baseAddress!
        for i in 0..<rank {
          baseAddress[i] = storage[3 + i]
        }
      }
    }
    set {
      let rank = newValue.count
      precondition(rank <= 5, "Tensor cannot have a rank greater than 5.")
      storage[1] = rank
      for i in 0..<rank {
        storage[3 + i] = newValue[i]
      }
      for i in rank + 3..<8 {
        storage[i] = 0
      }
    }
  }
  
  @inlinable @inline(__always)
  internal mutating func initializeDimensions<C: Collection>(
    _ dimensions: C
  ) where C.Element == Int, C.Index == Int {
    let rank = dimensions.count
    precondition(rank <= 5, "Tensor cannot have a rank greater than 5.")
    storage[1] = rank
    for i in 0..<rank {
      storage[3 + i] = dimensions[i]
    }
  }
  
  // Zero-cost wrapper over the internal storage of a tensor handle.
  @inlinable @inline(__always)
  internal init(tensorHandleStorage: SIMD8<Int>) {
    storage = tensorHandleStorage
  }
  
  @inlinable
  public init(_ dimensions: [Int]) {
    self.storage = .zero
    initializeDimensions(dimensions)
  }
  
  @inlinable
  public init<C: Collection>(_ dimensions: C) where C.Element == Int {
    self.storage = .zero
    let rank = dimensions.count
    precondition(rank <= 5, "Tensor cannot have a rank greater than 5.")
    storage[1] = rank
    var selfIndex = 3
    for dimIndex in dimensions.indices {
      storage[selfIndex] = dimensions[dimIndex]
      selfIndex += 1
    }
  }
  
  @inlinable
  public init(arrayLiteral elements: Int...) {
    self.storage = .zero
    initializeDimensions(elements)
  }
  
  @inlinable
  public init(_ elements: Int...) {
    self.storage = .zero
    initializeDimensions(elements)
  }
  
  @inlinable
  public init(repeating repeatedValue: Int, count: Int) {
    precondition(count <= 5, "Tensor cannot have a rank greater than 5.")
    self.storage = .zero
    storage[1] = count
    for i in 0..<count {
      storage[3 + i] = repeatedValue
    }
  }
  
  @inlinable
  public var rank: Int {
    return storage[1]
  }
  
  @inlinable
  public var contiguousSize: Int {
    var output = 1
    for i in 3..<(3 + rank) {
      output *= storage[i]
    }
    return output
  }
}

extension TensorShape: Collection, MutableCollection {
  public typealias Element = Int
  public typealias Index = Int
  public typealias Indices = Range<Int>
  
  @inlinable
  public var count: Int {
    return rank
  }
  
  @inlinable
  public var indices: Indices {
    return 0..<rank
  }
  
  @inlinable
  public var startIndex: Index {
    return indices.startIndex
  }
  
  @inlinable
  public var endIndex: Index {
    return indices.endIndex
  }
  
  @inlinable
  public func index(after i: Index) -> Index {
    return i + 1
  }
  
  @inlinable
  public subscript(position: Index) -> Element {
    _read {
      precondition(position < rank)
      yield storage[3 + position]
    }
    _modify {
      precondition(position < rank)
      yield &storage[3 + position]
    }
  }
  
  // This should be extremely slow.
  @inlinable
  public subscript(bounds: Range<Int>) -> TensorShape {
    get { return TensorShape(dimensions[bounds]) }
    set { dimensions[bounds] = ArraySlice(newValue.dimensions) }
  }
}

extension TensorShape: RandomAccessCollection {
  @inlinable
  public func index(_ i: Int, offsetBy distance: Int) -> Int {
    i + distance
  }
  
  @inlinable
  public func distance(from start: Int, to end: Int) -> Int {
    end - start
  }
}

extension TensorShape: RangeReplaceableCollection {
  public typealias SubSequence = Self

  @inlinable
  public init() {
    self.storage = .zero
  }

  // This should be extremely slow.
  @inlinable
  public mutating func append(_ newElement: Element) {
    dimensions.append(newElement)
  }

  // This should be extremely slow.
  @inlinable
  public mutating func append(contentsOf newElements: TensorShape) {
    dimensions.append(contentsOf: newElements.dimensions)
  }

  // This should be extremely slow.
  @inlinable
  public mutating func append<S: Sequence>(contentsOf newElements: S) where Element == S.Element {
    dimensions.append(contentsOf: newElements)
  }

  // This should be extremely slow.
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
    lhs.storage.highHalf == rhs.storage.highHalf && lhs.storage[3] == rhs.storage[3]
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
