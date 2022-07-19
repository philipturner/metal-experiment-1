//
//  tertiary.metal
//
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

kernel void tertiary(
  device float *input1 [[buffer(0)]],
  device float *input2 [[buffer(1)]],
  device float *input3 [[buffer(2)]],
  device float *output [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
