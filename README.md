# MetalExperiment1

Test whether I can fuse chained unary ops in an eager execution environment, using JIT compilation and runtime-dispatched ubershaders. Profile its performance and compare to CPU, considering all factors including Metal driver overhead. There are also other optimization goals, but I can choose which ones to pursue first.

Metal Experiments:
- 1) High sequential throughput, overcoming command buffer bottleneck, virtualizing operation dispatches
- 2) Overhead of incorporating thread safety, using cmdbuf completion handlers to automatically flush the command stream
- 3) Fast memory allocation, flushing command stream when system runs out of memory
- 4) Unary op fusion
- 5) Custom cache and shader archive for command-line SwiftPM builds, will reuse this concept in OpenCL backend which has no system shader cache
- 6) Multithreaded and delayed MPSGraph creation
