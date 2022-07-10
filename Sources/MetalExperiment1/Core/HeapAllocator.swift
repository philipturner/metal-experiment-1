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
    withUnsafeAddress(of: block.wrapped) { inputAddress in
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
          withUnsafeAddress(of: element.wrapped) { elementAddress in
            less = elementAddress < inputAddress
          }
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
  
  mutating func remove(at index: Int) {
    blocks.remove(at: index)
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
  
  static func makeHeap(size: Int, isShared: Bool) -> MTLHeap? {
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
    return device.makeHeap(descriptor: desc)
  }
  
  func makeBuffer(length: Int) -> MTLBuffer? {
    let buffer = heap.makeBuffer(length: length)
    guard let buffer = buffer else {
      return nil
    }
    availableSize = heap.maxAvailableSize(alignment: Int(vm_page_size))
    numBuffers += 1
    return buffer
  }
  
  func releaseBuffer(_ buffer: MTLBuffer) {
    availableSize = heap.maxAvailableSize(alignment: Int(vm_page_size))
    numBuffers -= 1
  }
  
  deinit {
    precondition(numBuffers == 0)
  }
}

class BufferPool {
  var isSmall: Bool
  var isShared: Bool
  // list of heaps ordered by their "available" (not total) memory size
  var heapBlocks: AllocatorBlockSet<HeapBlock> = .init()
  // list of only "available" buffers in the pool (i.e., buffers not in-use)
  var bufferBlocks: AllocatorBlockSet<HeapBlock> = .init()
  
  init(isSmall: Bool, isShared: Bool) {
    self.isSmall = isSmall
    self.isShared = isShared
  }
}

class HeapAllocator {
  static var global = HeapAllocator()
  private var allocatedBuffers: [UnsafeMutableRawPointer: BufferBlock] = [:]
  private var largePoolShared = BufferPool(isSmall: false, isShared: true)
  private var largePoolPrivate = BufferPool(isSmall: false, isShared: false)
  private var smallPoolShared = BufferPool(isSmall: true, isShared: true)
  private var smallPoolPrivate = BufferPool(isSmall: true, isShared: false)
  var totalAllocatedMemory = 0
  
  var maxBufferLength: Int {
    let device = Context.global.device
    return device.maxBufferLength
  }
  
  func pool(size: Int, usingShared: Bool) -> BufferPool {
    if size <= kMaxSmallAlloc {
      return usingShared ? smallPoolShared : smallPoolPrivate
    } else {
      return usingShared ? largePoolShared : largePoolPrivate
    }
  }
  
  func allocationSize(length: Int, usingShared: Bool) -> Int {
    let device = Context.global.device
    let options: MTLResourceOptions = usingShared ? .storageModeShared : .storageModePrivate
    let sizeAlign = device.heapBufferSizeAndAlign(length: length, options: options)
    return BufferBlock.alignUp(size: sizeAlign.size, alignment: sizeAlign.align)
  }
  
  var maxAvailableSize: Int {
    let device = Context.global.device
    return Int(device.recommendedMaxWorkingSetSize) - device.currentAllocatedSize
  }
  
  static func formatSize(_ size: Int) -> String {
    let kilobyte = 2 << 10
    let megabyte = 2 << 20
    let gigabyte = 2 << 30
    var formatString: String
    var formatArgument = Double(size)
    
    if size <= kilobyte {
      formatString = "%.2f bytes"
    } else if size <= megabyte {
      formatString = "%.2f KB"
      formatArgument /= Double(kilobyte)
    } else if size <= gigabyte {
      formatString = "%.2f MB"
      formatArgument /= Double(megabyte)
    } else {
      formatString = "%.2f GB"
      formatArgument /= Double(gigabyte)
    }
    return String(format: formatString, formatArgument)
  }
}
