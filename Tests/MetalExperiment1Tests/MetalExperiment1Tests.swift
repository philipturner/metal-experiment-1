import XCTest
@testable import MetalExperiment1

final class MetalExperiment1Tests: XCTestCase {
  func testContext() throws {
    _ = Context.global
  }
}
