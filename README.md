# MetalExperiment1

Create a proof-of-concept Metal backend for general-purpose eager execution, with a dominant use case of machine learning. Many of these optimizations can also be done in OpenCL.

Metal Experiments:
- 1) High sequential throughput :white_check_mark:, overcoming command buffer bottleneck :white_check_mark:, virtualizing operation dispatches :white_check_mark:
- 2) Overhead of incorporating thread safety :white_check_mark:, using command buffer completion handlers to automatically flush the command stream :white_check_mark:
- 3) Fast memory allocation, flushing command stream when system runs out of memory
- 4) Using predictable ARC and pipelined execution to optimize away intermediate tensors inside long chains of operations
- 5) Custom cache and shader archive for command-line SwiftPM builds, will reuse this concept in OpenCL backend which has no system shader cache
- 6) Multithreaded and delayed MPSGraph creation
