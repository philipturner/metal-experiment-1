//
//  Unary.metal
//  
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

// Relative execution time of bytes read/written per GPU thread:
// 1B - 1700
// 2B - 1100
// 4B - 700
// 8B - 400 (half4)
// 16B - 400 (float4)
// 32B - 400 (long4)
// 64B - ???
// 128B - starts to take longer

// Limit write alignment to 16B, taking a slight performance hit on the u32/i64/u64 ubershader. Use
// vectors of 2 scalars there. 64-bit operations are already ALU heavy, giving the performance
// characteristics of 4 32-bit types. This also lets me keep the 16B RAM alignment, which is a
// special number.

kernel void unaryOperation(
  device float *input [[buffer(0)]],
  device float *output [[buffer(1)]],
  constant float &increment [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  float value = input[tid];
  output[tid] = value + increment;
}

// u32/i64/u64 shader includes any casts that involve u32/i64/u64. The shader's start and end are
// more complex than f32-i32; it can read and write from more data types.
