// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MetalExperiment1",
    platforms: [
      .macOS(.v12),
      .iOS(.v15),
      .tvOS(.v15)
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "MetalExperiment1",
            targets: ["MetalExperiment1"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/apple/swift-atomics", branch: "main"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "MetalExperiment1",
            dependencies: [
              .product(name: "Atomics", package: "swift-atomics")
            ],
            resources: [
              .copy("Shaders")
            ]),
        .testTarget(
            name: "MetalExperiment1Tests",
            dependencies: ["MetalExperiment1"]),
    ]
)
