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
  private(set) var blocks: [Element] = []
  
  @inline(__always)
  private static func extractAddress(of element: Element) -> UInt {
    let unmanaged = Unmanaged<Element.Wrapped>.passUnretained(element.wrapped)
    return UInt(bitPattern: unmanaged.toOpaque())
  }
  
  mutating func insert(_ element: Element) {
    let inputAddress = withUnsafeAddress(of: element.wrapped) { $0 }
    if let index = firstIndex(minimumSize: element.size, minimumAddress: inputAddress) {
      let candidate = blocks[index]
      if candidate.size == element.size,
         candidate.wrapped === element.wrapped {
        // Catch memory management bugs.
        preconditionFailure("Cannot insert a buffer into pool twice.")
      }
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
      let middleBound = (lowerBound &+ upperBound) / 2
      let element = blocks[middleBound]
      
      var aimedTooLow = element.size < minimumSize
      if element.size == minimumSize {
        let elementAddress = withUnsafeAddress(of: element.wrapped) { $0 }
        if elementAddress == minimumAddress {
          return middleBound
        } else {
          aimedTooLow = UInt(bitPattern: elementAddress) < UInt(bitPattern: minimumAddress)
        }
      }
      if aimedTooLow {
        lowerBound = middleBound &+ 1
      } else {
        upperBound = middleBound &- 1
      }
    }
    if lowerBound == blocks.count {
      return nil
    } else {
      return lowerBound
    }
  }
  
  @discardableResult
  mutating func remove(at index: Int) -> Element {
    return blocks.remove(at: index)
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
    precondition(alignment.nonzeroBitCount == 1, "Alignment '\(alignment)' is not a power of 2.")
    return (size + alignment - 1) & ~(alignment - 1)
  }
  
  // Activate this code if you suspect there are memory leaks.
  #if true
  deinit {
    if HeapAllocator.debugInfoEnabled {
      let isShared = buffer.storageMode == .shared
      print("""
        Deinitialized \(isShared ? "shared" : "private") buffer #\(bufferID) of size \
        \(HeapAllocator.formatSize(size))
        """)
    }
  }
  #endif
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
  
  func makeBuffer(length: Int) -> MTLBuffer? {
    // The buffer's hazard tracking mode is always `.untracked`.
    let buffer = heap.makeBuffer(length: length, options: heap.resourceOptions)
    guard let buffer = buffer else {
      return nil
    }
    availableSize = heap.maxAvailableSize(alignment: Int(vm_page_size))
    numBuffers += 1
    return buffer
  }
  
  func decrementNumBuffers() {
    availableSize = heap.maxAvailableSize(alignment: Int(vm_page_size))
    numBuffers -= 1
  }
  
  deinit {
    precondition(numBuffers == 0, "Deinitialized heap block with \(numBuffers) remaining buffers.")
    
    // Activate this code if you suspect there are memory leaks.
    #if true
    if HeapAllocator.debugInfoEnabled {
      let isShared = heap.storageMode == .shared
      let mtlDevice = bufferPool.mtlDevice
      #if os(macOS)
      let maxWorkingSize = Int(mtlDevice.recommendedMaxWorkingSetSize)
      #else
      let maxWorkingSize = mtlDevice.maxBufferLength
      #endif
      let maxAvailableSize = maxWorkingSize - mtlDevice.currentAllocatedSize
      print("""
        Deinitialized \(bufferPool.isSmall ? "small" : "large") \(isShared ? "shared" : "private") \
        heap of size \(HeapAllocator.formatSize(totalSize)) (free memory: \
        \(HeapAllocator.formatSize(maxAvailableSize)))
        """)
    }
    #endif
  }
}

class BufferPool {
  var mtlDevice: MTLDevice
  var isSmall: Bool
  var isShared: Bool
  // List of heaps ordered by their "available" (not total) memory size.
  var heapBlocks: AllocatorBlockSet<HeapBlock> = .init()
  // List of only "available" buffers in the pool (i.e., buffers not in-use).
  var bufferBlocks: AllocatorBlockSet<BufferBlock> = .init()
  
  init(mtlDevice: MTLDevice, isSmall: Bool, isShared: Bool) {
    self.mtlDevice = mtlDevice
    self.isSmall = isSmall
    self.isShared = isShared
  }
  
  func makeHeap(size: Int, isShared: Bool) -> MTLHeap? {
    let desc = MTLHeapDescriptor()
    if size <= kMaxSmallAlloc {
      desc.size = kSmallHeap
    } else if size < kMinLargeAlloc {
      desc.size = kLargeHeap
    } else {
      desc.size = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge)
    }
    desc.storageMode = isShared ? .shared : .private
    
    // Don't automatically track contents. The context's encoding process manually tracks
    // dependencies between commands for better performance.
    desc.hazardTrackingMode = .untracked
    return mtlDevice.makeHeap(descriptor: desc)
  }
}

// MARK: - Declaration of HeapAllocator

class HeapAllocator {
  // Similar to the environment variable `PYTORCH_DEBUG_MPS_ALLOCATOR`.
  static var debugInfoEnabled = fetchEnvironmentBoolean("TENSORFLOW_DEBUG_HEAP_ALLOCATOR")
  
  private var mtlDevice: MTLDevice
  private var allocatedBuffers: [UnsafeMutableRawPointer: BufferBlock] = [:]
  
  // Unallocated cached buffers larger than 1 MB.
  private var largePoolShared: BufferPool
  private var largePoolPrivate: BufferPool
  
  // Unallocated cached buffers 1 MB or smaller.
  private var smallPoolShared: BufferPool
  private var smallPoolPrivate: BufferPool
  
  private(set) var totalAllocatedMemory = 0
  private var debugInfoHeapCounter = 0
  
  init(mtlDevice: MTLDevice) {
    self.mtlDevice = mtlDevice
    self.largePoolShared = BufferPool(mtlDevice: mtlDevice, isSmall: false, isShared: true)
    self.largePoolPrivate = BufferPool(mtlDevice: mtlDevice, isSmall: false, isShared: false)
    self.smallPoolShared = BufferPool(mtlDevice: mtlDevice, isSmall: true, isShared: true)
    self.smallPoolPrivate = BufferPool(mtlDevice: mtlDevice, isSmall: true, isShared: false)
  }
  
  private func pool(size: Int, usingShared: Bool) -> BufferPool {
    if size <= kMaxSmallAlloc {
      return usingShared ? smallPoolShared : smallPoolPrivate
    } else {
      return usingShared ? largePoolShared : largePoolPrivate
    }
  }
  
  func allocationSize(length: Int, usingShared: Bool) -> Int {
    let options: MTLResourceOptions = usingShared ? .storageModeShared : .storageModePrivate
    let sizeAlign = mtlDevice.heapBufferSizeAndAlign(length: length, options: options)
    precondition(sizeAlign.align >= 16, """
      Custom shaders assume heap alignment is at least 16 bytes. Instead, got \(sizeAlign.align)
      bytes.
      """)
    return BufferBlock.alignUp(size: sizeAlign.size, alignment: sizeAlign.align)
  }
  
  var maxAvailableSize: Int {
    #if os(macOS)
    let maxWorkingSize = Int(mtlDevice.recommendedMaxWorkingSetSize)
    #else
    let maxWorkingSize = mtlDevice.maxBufferLength
    #endif
    return maxWorkingSize - mtlDevice.currentAllocatedSize
  }
  
  static func formatSize(_ size: Int) -> String {
    let kilobyte = 1024
    let megabyte = 1024 * 1024
    let gigabyte = 1024 * 1024 * 1024
    var formatString: String
    var formatArgument = Double(size)
    
    if size <= kilobyte {
      // Zero decimal places.
      formatString = "%.0f bytes"
    } else if size <= megabyte {
      // Two decimal places.
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

extension HeapAllocator {
  func malloc(size: Int, usingShared: Bool) -> MTLBuffer? {
    // Important that this is `<=`, not `<` like in PyTorch.
    precondition(size <= mtlDevice.maxBufferLength, "Invalid buffer size: \(Self.formatSize(size))")
    let allocationSize = allocationSize(length: size, usingShared: usingShared)
    let pool = pool(size: allocationSize, usingShared: usingShared)
    
    var bufferBlock = removeBufferBlock(from: pool, size: allocationSize, requestedSize: size)
    if bufferBlock == nil {
      bufferBlock = makeBufferBlock(from: pool, size: allocationSize, requestedSize: size)
    }
    if bufferBlock == nil {
      releaseCachedBufferBlocks()
      bufferBlock = makeBufferBlock(from: pool, size: allocationSize, requestedSize: size)
    }
    guard let bufferBlock = bufferBlock else {
      return nil
    }
    bufferBlock.inUse = true
    return bufferBlock.buffer
  }
  
  func free(_ buffer: MTLBuffer) {
    let bufferAddress = withUnsafeAddress(of: buffer) { $0 }
    guard let bufferBlock = allocatedBuffers[bufferAddress] else {
      // Catch memory management bugs.
      preconditionFailure("Cannot free a buffer twice.")
    }
    freeBufferBlock(bufferBlock)
  }
  
  // Allow clearing of cached buffers in tests
  internal func _releaseCachedBufferBlocks() {
    self.releaseCachedBufferBlocks()
  }
}

// MARK: - Private methods of HeapAllocator

private extension HeapAllocator {
  func removeHeapBlock(from pool: BufferPool, size: Int) -> HeapBlock? {
    if let index = pool.heapBlocks.firstIndex(minimumSize: size) {
      let heapBlock = pool.heapBlocks.remove(at: index)
      
      // Returned heap's size may not equal the requested size.
      return heapBlock
    } else {
      let heap = pool.makeHeap(size: size, isShared: pool.isShared)
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
  
  func makeBufferBlock(
    from pool: BufferPool,
    size: Int,
    requestedSize: Int
  ) -> BufferBlock? {
    guard let heapBlock = removeHeapBlock(from: pool, size: size) else {
      return nil
    }
    let buffer = heapBlock.makeBuffer(length: size)!
    pool.heapBlocks.insert(heapBlock)
    let bufferBlock = BufferBlock(
      size: size, buffer: buffer, heapBlock: heapBlock, bufferID: allocatedBuffers.count + 1)
    let bufferAddress = withUnsafeAddress(of: buffer) { $0 }
    allocatedBuffers[bufferAddress] = bufferBlock
    totalAllocatedMemory += size
    
    if HeapAllocator.debugInfoEnabled {
      // Does PyTorch show just the buffer's memory address, or the entire ugly description from
      // passing it into `Swift.print`?
      print("""
        Allocated \(pool.isShared ? "shared" : "private") buffer #\(bufferBlock.bufferID) of size \
        \(Self.formatSize(size)) at \(bufferAddress) (requested size: \
        \(Self.formatSize(requestedSize)), heap size: \(Self.formatSize(heapBlock.availableSize)), \
        total allocated: \(Self.formatSize(totalAllocatedMemory)))
        """)
    }
    return bufferBlock
  }
  
  func freeBufferBlock(_ bufferBlock: BufferBlock) {
    precondition(bufferBlock.inUse)
    bufferBlock.inUse = false
    let pool = bufferBlock.heapBlock.bufferPool
    pool.bufferBlocks.insert(bufferBlock)
  }
  
  func removeBufferBlock(
    from pool: BufferPool,
    size: Int,
    requestedSize: Int
  ) -> BufferBlock? {
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
        \(Self.formatSize(bufferBlock.size)) at \(bufferAddress) (requested size: \
        \(Self.formatSize(requestedSize)))
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
    
    let bufferBlockIndex = pool.bufferBlocks.index(of: bufferBlock)!
    pool.bufferBlocks.remove(at: bufferBlockIndex)
    let heapBlockIndex = pool.heapBlocks.index(of: heapBlock)!
    pool.heapBlocks.remove(at: heapBlockIndex)
    heapBlock.decrementNumBuffers()
    
    if HeapAllocator.debugInfoEnabled {
      print("""
        Released buffer #\(bufferBlock.bufferID) of size \(Self.formatSize(bufferBlock.size)) \
        (heap size: \(Self.formatSize(heapBlock.availableSize)), total allocated: \
        \(Self.formatSize(totalAllocatedMemory)))
        """)
    }
    if heapBlock.numBuffers == 0 {
      heapBlock.availableSize = 0
      // `heapBlock` drops to a reference count of zero and deallocates.
      if HeapAllocator.debugInfoEnabled {
        print("""
          Released heap of size \(Self.formatSize(heapBlock.totalSize)) (free memory: \
          \(Self.formatSize(maxAvailableSize)))
          """)
      }
    } else {
      pool.heapBlocks.insert(heapBlock)
    }
  }
  
  func releaseBufferBlocks(from pool: BufferPool) {
    let bufferBlocks = pool.bufferBlocks
    for index in bufferBlocks.blocks.indices.reversed() {
      // Removes `bufferBlocks.blocks[index]`, which is always the last element.
      let bufferBlock = bufferBlocks.blocks[index]
      releaseBufferBlock(bufferBlock)
    }
  }
  
  func releaseCachedBufferBlocks() {
    releaseBufferBlocks(from: largePoolPrivate)
    releaseBufferBlocks(from: largePoolShared)
    releaseBufferBlocks(from: smallPoolPrivate)
    releaseBufferBlocks(from: smallPoolShared)
  }
}
