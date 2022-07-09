import XCTest
@testable import MetalExperiment1

final class MetalExperiment1Tests: XCTestCase {
  // Force this to execute first.
  func testA() throws {
    _ = Context.global
  }
}
