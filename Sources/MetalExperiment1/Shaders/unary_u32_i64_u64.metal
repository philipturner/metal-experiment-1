//
//  unary_u32_i64_u64.metal
//
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

// u32/i64/u64 shader includes any casts that involve u32/i64/u64. The shader's start and end are
// more complex than f32/i32; it can read and write from more data types.

kernel void unary_u32_i64_u64(
  device float *input [[buffer(0)]],
  device float *output [[buffer(1)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
