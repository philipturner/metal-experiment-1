//
//  HeapAllocation.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Metal

// Implementation based on PyTorch's MPS allocator:
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/mps/MPSAllocator.h

fileprivate let megabyte = 1024 * 1024
fileprivate let kMaxSmallAlloc = megabyte
fileprivate let kMinLargeAlloc = 10 * megabyte
fileprivate let kSmallHeap = 8 * megabyte
fileprivate let kLargeHeap = 32 * megabyte
fileprivate let kRoundLarge = 2 * megabyte

protocol AllocatorBlockProtocol {
  associatedtype Wrapped: AnyObject
  var wrapped: Wrapped? { get }
  var size: Int { get }
}

struct AllocatorBlockSet<T: AllocatorBlockProtocol> {
  private(set) var blocks: [T] = []
  
  @_transparent // Debug performance
  private static func extractAddress(of block: T) -> UInt {
    if let wrapped = block.wrapped {
      let ptr = Unmanaged<T.Wrapped>.passUnretained(wrapped).toOpaque()
      return UInt(bitPattern: ptr)
    } else {
      return 0
    }
  }
  
  mutating func insert(_ block: T) {
    let inputSize = block.size
    let inputAddress = Self.extractAddress(of: block)
    
    var lowerBound = 0
    var upperBound = blocks.count - 1
    while lowerBound <= upperBound {
      let middleBound = (lowerBound + upperBound) / 2
      let element = blocks[middleBound]
      let elementSize = element.size
      
      var less = false
      if elementSize < inputSize {
        less = true
      } else if elementSize == inputSize {
        let elementAddress = Self.extractAddress(of: element)
        less = elementAddress < inputAddress
      }
      
      if less {
        lowerBound = middleBound + 1
      } else {
        upperBound = middleBound - 1
      }
    }
    blocks.insert(block, at: lowerBound)
  }
}

class BufferBlock: AllocatorBlockProtocol {
  unowned var heap: HeapBlock?
  var buffer: MTLBuffer?
  var wrapped: MTLBuffer? { buffer }
  var size: Int
  var inUse: Bool
  var bufferID: UInt32
  
  init(size: Int) {
    self.size = size
    self.inUse = false
    self.bufferID = 0
  }
}

class HeapBlock {
  unowned var pool: BufferPool?
  var heap: MTLHeap?
  
}

class BufferPool {
  
}
