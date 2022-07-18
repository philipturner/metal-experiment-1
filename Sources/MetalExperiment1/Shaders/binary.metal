//
//  binary.metal
//
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

kernel void binary(
  device float *input1 [[buffer(0)]],
  device float *input2 [[buffer(1)]],
  device float *output [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
