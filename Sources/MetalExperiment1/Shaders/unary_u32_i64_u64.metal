//
//  unary_u32_i64_u64.metal
//
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

// The u32/i64/u64 ubsershader includes any casts that involve u32/i64/u64. Its start and end are
// more complex than f32/i32; it can read and write from more data types.

enum UnaryOperationType2: ushort {
  abs_i64 = 0,
  neg_i64 = 1,
  sign_i64 = 2,
  sign_u64 = 3,
  square_i64 = 4,
  square_u64 = 5,
  
  cast_f32_to_i64 = 10,
  cast_i64_to_bool = 11,
  cast_i64_to_f16 = 12,
  cast_i64_to_f32 = 13,
  cast_i64_to_u8 = 14,
  cast_i64_to_u16 = 15,
  cast_i64_to_u32 = 16,
  
  case_f32_to_u32 = 20,
  cast_f32_to_u64 = 21,
  cast_u64_to_bool = 22,
  cast_u64_to_f16 = 23,
  cast_u64_to_f32 = 24,
  cast_u64_to_u8 = 25,
  cast_u64_to_u16 = 26,
  cast_u64_to_u32 = 27,
  
  increment_i64 = 30, // for testing purposes only
};

kernel void unary_u32_i64_u64(
  device float *input [[buffer(0)]],
  device float *output [[buffer(1)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
