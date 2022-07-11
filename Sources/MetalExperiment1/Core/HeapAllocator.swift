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

struct AllocatorBlockSet<Element: AllocatorBlockProtocol> {
  private var blocks: [Element] = []
  
  @inline(__always)
  private static func extractAddress(of element: Element) -> UInt {
    let unmanaged = Unmanaged<Element.Wrapped>.passUnretained(element.wrapped)
    return UInt(bitPattern: unmanaged.toOpaque())
  }
  
  mutating func insert(_ element: Element) {
    let inputAddress = withUnsafeAddress(of: element.wrapped) { $0 }
    if let index = firstIndex(minimumSize: element.size, minimumAddress: inputAddress) {
      blocks.insert(element, at: index)
    } else {
      blocks.append(element)
    }
  }
  
  func index(of element: Element) -> Int? {
    let inputAddress = withUnsafeAddress(of: element.wrapped) { $0 }
    guard let index = firstIndex(minimumSize: element.size, minimumAddress: inputAddress) else {
      return nil
    }
    let candidate = blocks[index]
    if candidate.size == element.size,
       candidate.wrapped === element.wrapped {
      return index
    } else {
      return nil
    }
  }
  
  func firstIndex(minimumSize: Int, minimumAddress: UnsafeMutableRawPointer? = nil) -> Int? {
    var lowerBound = 0
    var upperBound = blocks.count - 1
    while lowerBound <= upperBound {
      let middleBound = (lowerBound + upperBound) / 2
      let element = blocks[middleBound]
      
      var aimedTooLow = element.size < minimumSize
      if element.size == minimumSize {
        let elementAddress = withUnsafeAddress(of: element.wrapped) { $0 }
        if elementAddress == minimumAddress {
          return middleBound
        } else {
          aimedTooLow = elementAddress < minimumAddress.unsafelyUnwrapped
        }
      }
      if aimedTooLow {
        lowerBound = middleBound + 1
      } else {
        upperBound = middleBound - 1
      }
    }
    if lowerBound == blocks.count {
      return nil
    } else {
      return lowerBound
    }
  }
  
  mutating func remove(at index: Int) -> Element {
    return blocks.remove(at: index)
  }
}

class BufferBlock: AllocatorBlockProtocol {
  unowned let heapBlock: HeapBlock
  var buffer: MTLBuffer
  var wrapped: MTLBuffer { buffer }
  
  var size: Int
  var requestedSize: Int
  var inUse: Bool
  var bufferID: Int
  
  init(size: Int, requestedSize: Int, buffer: MTLBuffer, heapBlock: HeapBlock, bufferID: Int) {
    self.heapBlock = heapBlock
    self.buffer = buffer
    self.size = size
    self.requestedSize = requestedSize
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
  // List of heaps ordered by their "available" (not total) memory size.
  var heapBlocks: AllocatorBlockSet<HeapBlock> = .init()
  // List of only "available" buffers in the pool (i.e., buffers not in-use).
  var bufferBlocks: AllocatorBlockSet<BufferBlock> = .init()
  
  init(isSmall: Bool, isShared: Bool) {
    self.isSmall = isSmall
    self.isShared = isShared
  }
}

// MARK: - Declaration of HeapAllocator

class HeapAllocator {
  static var global = HeapAllocator()
  
  // Similar to the environment variable `PYTORCH_DEBUG_MPS_ALLOCATOR`.
  static var debugInfoEnabled = getenv("TENSORFLOW_DEBUG_PLUGGABLE_DEVICE_ALLOCATOR") != nil
  
  private var allocatedBuffers: [UnsafeMutableRawPointer: BufferBlock] = [:]
  
  // Unallocated cached buffers larger than 1 MB.
  private var largePoolShared = BufferPool(isSmall: false, isShared: true)
  private var largePoolPrivate = BufferPool(isSmall: false, isShared: false)
  
  // Unallocated cached buffers 1 MB or smaller.
  private var smallPoolShared = BufferPool(isSmall: true, isShared: true)
  private var smallPoolPrivate = BufferPool(isSmall: true, isShared: false)
  
  private var totalAllocatedMemory = 0
  
  private var maxBufferLength: Int {
    let device = Context.global.device
    return device.maxBufferLength
  }
  
  private func pool(size: Int, usingShared: Bool) -> BufferPool {
    if size <= kMaxSmallAlloc {
      return usingShared ? smallPoolShared : smallPoolPrivate
    } else {
      return usingShared ? largePoolShared : largePoolPrivate
    }
  }
  
  private func allocationSize(length: Int, usingShared: Bool) -> Int {
    let device = Context.global.device
    let options: MTLResourceOptions = usingShared ? .storageModeShared : .storageModePrivate
    let sizeAlign = device.heapBufferSizeAndAlign(length: length, options: options)
    return BufferBlock.alignUp(size: sizeAlign.size, alignment: sizeAlign.align)
  }
  
  private var maxAvailableSize: Int {
    let device = Context.global.device
    return Int(device.recommendedMaxWorkingSetSize) - device.currentAllocatedSize
  }
  
  private static func formatSize(_ size: Int) -> String {
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

internal extension HeapAllocator {
  func malloc(size: Int, usingShared: Bool) {
    precondition(size < maxBufferLength, "Invalid buffer size: \(Self.formatSize(size))")
    let allocationSize = self.allocationSize(length: size, usingShared: usingShared)
    let pool = self.pool(size: allocationSize, usingShared: usingShared)
    
    var bufferBlock = extractFreeBuffer(from: pool, size: allocationSize, requestedSize: size)
    if bufferBlock == nil {
      bufferBlock = allocateBuffer(from: pool, size: allocationSize, requestedSize: size)
    }
    if bufferBlock == nil {
      
    }
  }
}

// MARK: - Private methods of HeapAllocator

fileprivate var debugInfoHeapCounter = 0

private extension HeapAllocator {
  // Must insert the heap back into the pool afterwards.
  func extractFreeHeap(from pool: BufferPool, size: Int) -> HeapBlock? {
    if let index = pool.heapBlocks.firstIndex(minimumSize: size) {
      let heapBlock = pool.heapBlocks.remove(at: index)
      
      // Returned heap's size may not equal the requested size.
      return heapBlock
    } else {
      let heap = HeapBlock.makeHeap(size: size, isShared: pool.isShared)
      guard let heap = heap else {
        return nil
      }
      let heapSize = heap.maxAvailableSize(alignment: Int(vm_page_size))
      let heapBlock = HeapBlock(size: heapSize, heap: heap, bufferPool: pool)
      
      if HeapAllocator.debugInfoEnabled {
        debugInfoHeapCounter += 1
        print("""
          Allocated \(pool.isSmall ? "small" : "large") \(pool.isShared ? "shared" : "private") \
          heap of size \(Self.formatSize(heapSize)) (#heaps: \(debugInfoHeapCounter), free memory: \
          \(Self.formatSize(maxAvailableSize)))
          """)
      }
      return heapBlock
    }
  }
  
  func allocateBuffer(from pool: BufferPool, size: Int, requestedSize: Int) -> BufferBlock? {
    guard let heapBlock = extractFreeHeap(from: pool, size: requestedSize) else {
      return nil
    }
    let buffer = heapBlock.makeBuffer(length: size)!
    pool.heapBlocks.insert(heapBlock)
    let bufferBlock = BufferBlock(
      size: size, requestedSize: requestedSize, buffer: buffer, heapBlock: heapBlock,
      bufferID: allocatedBuffers.count + 1)
    let bufferAddress = withUnsafeAddress(of: buffer) { $0 }
    allocatedBuffers[bufferAddress] = bufferBlock
    totalAllocatedMemory += size
    
    if HeapAllocator.debugInfoEnabled {
      // Does PyTorch show just the buffer's memory address, or the entire ugly description from
      // passing it into `Swift.print`?
      print("""
        Allocated \(pool.isShared ? "shared" : "private") buffer #\(bufferBlock.bufferID) of size \
        \(Self.formatSize(size)) at \(bufferAddress) (requested size: \(requestedSize), heap size: \
        \(Self.formatSize(heapBlock.availableSize)), total allocated: \
        \(Self.formatSize(totalAllocatedMemory)))
        """)
    }
    return bufferBlock
  }
  
  // Must insert the buffer back into the pool at some point.
  func extractFreeBuffer(from pool: BufferPool, size: Int, requestedSize: Int) -> BufferBlock? {
    guard let index = pool.bufferBlocks.firstIndex(minimumSize: size) else {
      return nil
    }
    let bufferBlock = pool.bufferBlocks.remove(at: index)
    
    if HeapAllocator.debugInfoEnabled {
      // Does PyTorch show just the buffer's memory address, or the entire ugly description from
      // passing it into `Swift.print`?
      let bufferAddress = withUnsafeAddress(of: bufferBlock.buffer) { $0 }
      print("""
        Reusing \(pool.isShared ? "shared" : "private") buffer #\(bufferBlock.bufferID) of size \
        \(Self.formatSize(bufferBlock.size)) at \(bufferAddress) (requested size: \(requestedSize))
        """)
    }
    return bufferBlock
  }
  
  func releaseBufferBlock(_ bufferBlock: BufferBlock) {
    let heapBlock = bufferBlock.heapBlock
    let pool = heapBlock.bufferPool
    totalAllocatedMemory -= bufferBlock.size
    let bufferAddress = withUnsafeAddress(of: bufferBlock.buffer) { $0 }
    allocatedBuffers.removeValue(forKey: bufferAddress)
//    pool.bufferBlocks.
  }
}
