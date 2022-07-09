# MetalExperiment1

Test whether I can fuse chained unary ops in an eager execution environment, using JIT compilation and runtime-dispatched ubershaders. Profile its performance and compare to CPU, considering all factors including Metal driver overhead. There are also other optimization goals, but I can choose which ones to pursue first.

Metal Experiments:
- 1) High sequential throughput, overcoming command buffer bottleneck
- 2) Fast memory allocation
- 3) Unary op fusion
- 4) Custom cache and shader archive for command-line SwiftPM builds
- 5) Multithreaded and delayed MPSGraph creation
