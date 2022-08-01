import XCTest
@testable import MetalExperiment1
import Metal

internal let defaultPluggableDevice = Context.global

func testHeader(_ message: String? = nil) {
  Profiler.checkpoint()
  _ = defaultPluggableDevice
  let startupTime = Profiler.checkpoint()
  if startupTime > 1000 {
    print("=== Initialize context ===")
    print("Initialization time: \(startupTime) \(Profiler.timeUnit)")
  }
  
  if let message = message {
    print()
    print("=== \(message) ===")
  }
  
  // Stop messages about references from flooding the console. You can re-activate this inside a
  // test function if you want.
  defaultPluggableDevice.sync {
    Allocation.debugInfoEnabled = false
  }
  defaultPluggableDevice.barrier()
}

fileprivate protocol DummyPluggableDevice: AnyObject {}

final class MetalExperiment1Tests: XCTestCase {
  func testSynchronizationLatency() throws {
    testHeader("Synchronization latency")
    
    for _ in 0..<2 {
      Profiler.checkpoint()
      _ = defaultPluggableDevice.sync {
        Bool.random()
      }
      Profiler.log("Synchronization latency")
    }
    
    do {
      Profiler.checkpoint()
      let iterations = 100
      for _ in 0..<iterations {
        _ = defaultPluggableDevice.sync {
          Bool.random()
        }
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("Mutex throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    do {
      let iterations = 1000
      _ = DispatchSemaphore(value: 0)
      var semaphores: [DispatchSemaphore] = []
      semaphores.reserveCapacity(iterations)
      
      Profiler.checkpoint()
      for _ in 0..<iterations {
        semaphores.append(DispatchSemaphore(value: 0))
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("Dispatch semaphore creation throughput: \(throughput) \(Profiler.timeUnit)")
    }
    
    do {
      Profiler.checkpoint()
      _ = _ExecutionContext.global.currentDeviceName
      Profiler.log("_ThreadLocalState startup latency")
    }
    
    func profileThreadLocalState(iterations: Int) {
      Profiler.checkpoint()
      for _ in 0..<iterations {
        _ = _ExecutionContext.global.currentDeviceName
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("_ThreadLocalState throughput: \(throughput) \(Profiler.timeUnit)")
    }
    profileThreadLocalState(iterations: 5)
    profileThreadLocalState(iterations: 1000)
    
    defaultPluggableDevice.barrier()
    defer {
      defaultPluggableDevice.barrier()
    }
    
    do {
      Profiler.checkpoint()
      _ = FrontendContext.local.globalTensorCount
      Profiler.log("ContextManager startup latency")
    }
    
    func profileContextManager(iterations: Int) {
      let startCount = FrontendContext.local.globalTensorCount
      
      Profiler.checkpoint()
      for _ in 0..<iterations {
        FrontendContext.local.globalTensorCount += 1
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("ContextManager throughput: \(throughput) \(Profiler.timeUnit)")
      precondition(FrontendContext.local.globalTensorCount == startCount + iterations)
      FrontendContext.local.globalTensorCount -= iterations
    }
    profileContextManager(iterations: 5)
    profileContextManager(iterations: 1000)
  }
  
  func testDynamicTyping() throws {
    testHeader("Dynamic type pointer extraction")
    
    func extractAddress(of device: any DummyPluggableDevice) -> OpaquePointer {
      func nestedVirtualFunction<T: DummyPluggableDevice>(_ type: T.Type) -> OpaquePointer {
        .init(Unmanaged.passUnretained(device as! T).toOpaque())
      }
      let metatype: DummyPluggableDevice.Type = type(of: device)
      return nestedVirtualFunction(metatype)
    }
    
    class MyDummyPluggableDevice1: DummyPluggableDevice {
      init() {}
    }
    class MyDummyPluggableDevice2: DummyPluggableDevice {
      init() {}
    }
    class MyDummyPluggableDevice3: DummyPluggableDevice {
      init() {}
    }
    class MyDummyPluggableDevice4: DummyPluggableDevice {
      init() {}
    }
    class MyDummyPluggableDevice5: DummyPluggableDevice {
      init() {}
    }
    
    let device1: any DummyPluggableDevice = MyDummyPluggableDevice1()
    let device2: any DummyPluggableDevice = MyDummyPluggableDevice2()
    XCTAssertNotIdentical(device1, device2)
    
    let address1 = extractAddress(of: device1)
    let address2 = extractAddress(of: device2)
    XCTAssertNotEqual(address1, address2)
    
    // Test speed when multiple valid types exist.
    let devices: [any DummyPluggableDevice] = [
      MyDummyPluggableDevice1(),
      MyDummyPluggableDevice2(),
      MyDummyPluggableDevice3(),
      MyDummyPluggableDevice4(),
      MyDummyPluggableDevice5(),
    ]
    var pointers: [OpaquePointer?] = Array(repeating: nil, count: 5)
    
    func profileExtract(iterations: Int) {
      Profiler.checkpoint()
      for i in 0..<iterations {
        let device = devices[i % 5]
        let address = extractAddress(of: device)
        pointers[i % 5] = address
      }
      let totalTime = Profiler.checkpoint()
      let throughput = Double(totalTime) / Double(iterations)
      print("Pointer extraction throughput: \(throughput) \(Profiler.timeUnit)")
    }
    profileExtract(iterations: 5)
    profileExtract(iterations: 1000)
  }
  
  func testStreamedBatchThroughput() throws {
    testHeader("Streamed command buffer throughput")
    
    func validate(_ tensorHandle: Tensor<Float>, value: Float) {
      XCTAssertEqual(tensorHandle.scalars[0], value)
    }
    
    func testWarmup(name: String) {
      print("--- Stream size: 1")
      Profiler.checkpoint()
      let handle1 = Tensor<Float>(repeating: 0, shape: [10])
      let creationTime = Profiler.checkpoint()
      
      let handle2 = handle1.incremented()
      let latency = Profiler.checkpoint()
      validate(handle2, value: 1.0)
      
      let totalTime = latency + Profiler.checkpoint()
      print("""
        Creation time: \(creationTime) \(Profiler.timeUnit) \
        First batch latency: \(latency) \(Profiler.timeUnit) \
        Total time: \(totalTime) \(Profiler.timeUnit)
        """)
    }
    testWarmup(name: "First")
    testWarmup(name: "Second")
    
    func profileStream(length: Int) {
      print("--- Stream size: \(length)")
      Profiler.checkpoint()
      var handle = Tensor<Float>(repeating: 0, shape: [10])
      for _ in 0..<length {
        handle = handle.incremented()
      }
      let latency = Profiler.checkpoint()
      validate(handle, value: Float(length))
      
      let totalTime = latency + Profiler.checkpoint()
      let latencyAverage = Double(latency * 100 / length) / 100
      let throughputAverage = Double(totalTime * 10 / length) / 10
      print("""
        Average CPU-side latency: \(latencyAverage) \(Profiler.timeUnit), \
        Amortized sequential throughput: \(throughputAverage) \(Profiler.timeUnit), \
        Total time: \(totalTime) \(Profiler.timeUnit)
        """)
    }
    
    let maxCommandsPerBatch = defaultPluggableDevice.sync {
      defaultPluggableDevice.maxCommandsPerBatch
    }
    for _ in 0..<5 {
      profileStream(length: maxCommandsPerBatch * 4)
    }
  }
  
  private struct ARCEvent: Equatable {
    var id: Int
    var wasAllocating: Bool
  }
  private static var history: [ARCEvent] = []
  
  // Enables automatic fusion of unary ops, with no extra work from the frontend. Deallocated
  // (invalid) tensors become placeholders for operation fusion.
  func testARCDeallocation() throws {
    testHeader()
    Self.history = []
    
    class DummyTensorHandle {
      var id: Int
      
      init(id: Int) {
        self.id = id
        MetalExperiment1Tests.history.append(ARCEvent(id: id, wasAllocating: true))
      }
      
      deinit {
        MetalExperiment1Tests.history.append(ARCEvent(id: id, wasAllocating: false))
      }
    }
    
    struct DummyTensor {
      private var handle: DummyTensorHandle
      
      init(id: Int) {
        self.handle = DummyTensorHandle(id: id)
      }
    }
    
    // Test whether inlining changes things.
    
    @inline(never)
    func f1(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 1)
    }
    
    func f2(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 2)
    }
    
    @inline(__always)
    func f3(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 3)
    }
    
    func f4(_ x: DummyTensor) -> DummyTensor {
      return DummyTensor(id: 4)
    }
    
    _ = f4(f3(f2(f1(DummyTensor(id: 0)))))
    
    let expectedHistory: [ARCEvent] = [
      // Explicitly declared input to `function1`.
      .init(id: 0, wasAllocating: true),
      
      // Call into `f1`.
      .init(id: 1, wasAllocating: true),
      .init(id: 0, wasAllocating: false),
      
      // Call into `f2`.
      .init(id: 2, wasAllocating: true),
      .init(id: 1, wasAllocating: false),
      
      // Call into `f3`.
      .init(id: 3, wasAllocating: true),
      .init(id: 2, wasAllocating: false),
      
      // Call into `f3`.
      .init(id: 4, wasAllocating: true),
      .init(id: 3, wasAllocating: false),
      
      // Discarded result of `f4`.
      .init(id: 4, wasAllocating: false)
    ]
    XCTAssertEqual(Self.history, expectedHistory)
  }
  
  func testSmallVector() throws {
    testHeader()
    
    var list1 = SmallVector<SIMD2<UInt8>>()
    list1.append(1)
    XCTAssertEqual(list1.count, 1)
    XCTAssertEqual(list1[0], 1)
    
    list1.append(2)
    list1.append(3)
    XCTAssertEqual(list1.count, 3)
    XCTAssertEqual(list1[0], 1)
    XCTAssertEqual(list1[1], 2)
    XCTAssertEqual(list1[2], 3)
    
    // Would crash with a regular Swift array, but is acceptable here.
    let list2 = SmallVector<SIMD16<UInt8>>()
    XCTAssertEqual(list2.count, 0)
    XCTAssertEqual(list2[0], 0)
    XCTAssertEqual(list2[1], 0)
  }
  
  private static var didDestroyObject = false
  
  func testUnmanagedReferences() throws {
    testHeader()
    Self.didDestroyObject = false
    
    class MyClass {
      init() {}
      
      deinit {
        MetalExperiment1Tests.didDestroyObject = true
      }
    }
    
    func getRetained() -> Unmanaged<MyClass> {
      let instance = MyClass()
      XCTAssertFalse(Self.didDestroyObject)
      let retained = Unmanaged.passRetained(instance)
      XCTAssertFalse(Self.didDestroyObject)
      return retained
    }
    let retained = getRetained()
    XCTAssertFalse(Self.didDestroyObject)
    func releaseRetained(_ input: Unmanaged<MyClass>) {
      _ = input.takeRetainedValue()
    }
    releaseRetained(retained)
    XCTAssertTrue(Self.didDestroyObject)
  }
  
  func testScalarBroadcast() throws {
    testHeader()
    
    let tensor = Tensor<Int32>(repeating: 1, shape: [1])
    XCTAssertEqual(tensor.incremented().scalars, [2])
    
    let tensor2 = Tensor<Int8>(Tensor(repeating: Float(2), shape: [1]))
    XCTAssertEqual(Tensor<Float>(square(tensor2)).scalars, [4])
    
    let tensor3 = Tensor<Int64>(Tensor(repeating: Int64(3), shape: [1]))
    XCTAssertEqual(Tensor<UInt8>(square(tensor3)).scalars, [9])
    
    let tensor4 = Tensor<Int8>(Tensor(repeating: Int8(-4), shape: [1]))
    XCTAssertEqual(Tensor<UInt64>(square(tensor4)).scalars, [16])
  }
  
  func testMovingAverage() throws {
    testHeader()
    
    var average = MovingAverage<UInt64>(repeating: 0, count: 10)
    XCTAssertEqual(average.average, 0)
    
    average.append(20)
    XCTAssertEqual(average.average, 2)
    
    average.append(9)
    XCTAssertEqual(average.average, 2)
    
    average.append(1)
    XCTAssertEqual(average.average, 3)
    
    for _ in 0..<7 {
      average.append(10)
    }
    XCTAssertEqual(average.average, (30 + 7 * 10) / 10)
    
    for _ in 0..<10 {
      average.append(5)
    }
    XCTAssertEqual(average.average, 5)
  }
  
  func testMultipleDevices() throws {
    testHeader()
  }
}
