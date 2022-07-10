//
//  Memory.swift
//  
//
//  Created by Philip Turner on 7/10/22.
//

import Metal
import MetalPerformanceShadersGraph

struct Allocation {
  var mtlBuffer: MTLBuffer
  var mpsGraphTensorData: MPSGraphTensorData?
}
