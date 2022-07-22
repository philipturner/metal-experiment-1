# MetalExperiment1

Create a proof-of-concept Metal backend for general-purpose eager execution, with a dominant use case of machine learning. Many of these optimizations can also be done in OpenCL.

Metal experiments:
- 1) High sequential throughput :white_check_mark:, overcoming command buffer bottleneck :white_check_mark:, virtualizing operation dispatches :white_check_mark:
- 2) Overhead of incorporating thread safety :white_check_mark:, using command buffer completion handlers to automatically flush the command stream :white_check_mark:
- 3) Fast memory allocation :white_check_mark:, flushing command stream when system runs out of memory :white_check_mark:
- 4) Use predictable ARC and pipelined execution to optimize away intermediate tensors inside long chains of operations :white_check_mark:
- 5) Custom cache and shader archive for command-line SwiftPM builds (will reuse this concept in OpenCL backend which has no system shader cache) :white_check_mark:, create elementwise operation ubershaders
- 6) Multithreaded and delayed MPSGraph creation for convolution operations
- 7) Perform "constant folding" of extremely tiny tensors on the CPU, before submitting to the GPU
- 8) Generate a graph in-flight, fuse non-adjacent unary operations, fuse binary/ternary operations into unary operation chains

> You may notice some wierd commit messages being just "z". It is easy to type "z" on a keyboard when submitting a commit message to synchronize your work with the cloud. Alternatively, I could close-source this repo to prevent the informal commit messages from being public. But the downsides of that approach are much greater.
