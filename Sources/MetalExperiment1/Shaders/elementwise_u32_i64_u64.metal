//
//  elementwise_u32_i64_u64.metal
//
//
//  Created by Philip Turner on 7/8/22.
//

#include <metal_stdlib>
using namespace metal;

// The u32/i64/u64 ubsershader includes any casts that involve u32/i64/u64. Its start and end are
// more complex than f32/i32; it can read and write from more data types.

enum MemoryCast2: ushort {
  
};

enum ElementwiseOperationType2: ushort {
  abs_i64 = 0,
  neg_i64 = 1,
  sign_i64 = 2,
  sign_u64 = 3,
  square_i64 = 4,
  square_u64 = 5,
  
  cast_f32_to_u32 = 10,
  cast_f32_to_i64 = 11,
  cast_i64_to_f16 = 12,
  cast_i64_to_f32 = 13,
  cast_i64_to_bool = 14,
  // cast_i64_to_i32 = cast_u64_to_i32,
  // cast_i64_to_u32 = cast_u64_to_u32
  
  cast_f32_to_u64 = 20,
  cast_u64_to_f16 = 21,
  cast_u64_to_f32 = 22,
  // cast_u64_to_bool = cast_i64_to_bool
  cast_u64_to_i32 = 23, // requires metadata
  cast_u64_to_u32 = 24, // requires metadata
  
  scalar_add_i64 = 30, // requires metadata
  scalar_mul_i64 = 31, // requires metadata
  scalar_mul_u64 = 32, // requires metadata
};

kernel void elementwise_u32_i64_u64(
//  constant DispatchParams &params [[buffer(0)]],
  constant ElementwiseOperationType2 *operations [[buffer(1)]],
  constant void *metadata [[buffer(2)]],
  device void *input1 [[buffer(3)]],
  device void *input2 [[buffer(4)]],
  device void *input3 [[buffer(5)]],
  device void *output [[buffer(6)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
