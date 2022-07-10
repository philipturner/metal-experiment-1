//
//  HeapAllocator.swift
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
  var wrapped: Wrapped { get }
  var size: Int { get }
}

struct AllocatorBlockSet<T: AllocatorBlockProtocol> {
  private(set) var blocks: [T] = []
  
  @inline(__always)
  private static func extractAddress(of block: T) -> UInt {
    let unmanaged = Unmanaged<T.Wrapped>.passUnretained(block.wrapped)
    return UInt(bitPattern: unmanaged.toOpaque())
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
  unowned let heapBlock: HeapBlock
  var buffer: MTLBuffer
  var wrapped: MTLBuffer { buffer }
  
  var size: Int
  var inUse: Bool
  var bufferID: Int
  
  init(size: Int, buffer: MTLBuffer, heapBlock: HeapBlock, bufferID: Int) {
    self.heapBlock = heapBlock
    self.buffer = buffer
    self.size = size
    self.inUse = false
    self.bufferID = bufferID
  }
  
  static func alignUp(size: Int, alignment: Int) -> Int {
    precondition(alignment.nonzeroBitCount == 1)
    return (size + alignment - 1) & ~(alignment - 1)
  }
}

class HeapBlock: AllocatorBlockProtocol {
  unowned var bufferPool: BufferPool
  var heap: MTLHeap
  var wrapped: MTLHeap { heap }
  var size: Int { availableSize }
  
  var totalSize: Int
  var availableSize: Int
  var numBuffers: Int
  
  init(size: Int, heap: MTLHeap, bufferPool: BufferPool) {
    self.bufferPool = bufferPool
    self.heap = heap
    self.totalSize = size
    self.availableSize = size
    self.numBuffers = 0
  }
  
  static func makeMTLHeap(size: Int, isShared: Bool) -> MTLHeap? {
    let desc = MTLHeapDescriptor()
    if size <= kMaxSmallAlloc {
      desc.size = kSmallHeap
    } else if size < kMinLargeAlloc {
      desc.size = kLargeHeap
    } else {
      desc.size = kRoundLarge + ((size + kRoundLarge - 1) / kRoundLarge)
    }
    desc.storageMode = isShared ? .shared : .private
    desc.hazardTrackingMode = .tracked
    
    let device = Context.global.device
    // May return `nil`.
    return device.makeHeap(descriptor: desc)
  }
  
  static func availableSize(heap: MTLHeap, alignment: Int = Int(vm_page_size)) {
    heap.maxAvailableSize(alignment: alignment)
  }
//
//  func makeMTLBuffer(length: Int) throws -> MTLBuffer {
//    let buffer = heap!.makeBuffer(length: length)
//
//  }
}

class BufferPool {
  
}
