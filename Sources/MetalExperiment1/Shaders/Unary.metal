//
//  Unary.metal
//  
//
//  Created by Philip Turner on 7/8/22.
//

kernel void unaryOperation(
  device float *input [[buffer(0)]],
  device float *output [[buffer(1)]],
  uint tid [[thread_position_in_grid]]
) {
  float value = input[tid];
  output[tid] = value;
}
