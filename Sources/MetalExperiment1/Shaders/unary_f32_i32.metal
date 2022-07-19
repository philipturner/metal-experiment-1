//
//  unary_f32_i32.metal
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

kernel void unary_f32_i32(
  device float *input [[buffer(0)]],
  device float *output [[buffer(1)]],
  constant float &increment [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  float value = input[tid];
  output[tid] = value + increment;
}

// MARK: - New Code

struct DispatchParams {
  bool read_scalar_broadcast;
  ushort read_size;
  ushort write_size;
};

enum MemoryCast: ushort {
  f32_i32_native,
  f16_as_f32,
  i8_as_i32,
  i16_as_i32,
  u8_as_i32,
  u16_as_i32,
  // `bool` can be masked as either `i8` or `u8`.
};

enum UnaryOperationType: ushort {
  cast_f32_i32,
  cast_i32_f32,
};

class Storage {
  uint4 data;
  
public:
  Storage() {}
  
  void set_f32_i32(uint4 mem_slice) {
    
  }
  
  void set_f16(ushort4 mem_slice) {
    
  }
  
  void set_i8(uchar4 mem_slice) {
    
  }
  
  void set_i16(ushort4 mem_slice) {
    
  }
  
  void set_u8(uchar4 mem_slice) {
    
  }
  
  void set_u16(ushort4 mem_slice) {
    
  }
  
};


kernel void unary_f32_i32_new(
  device void *input [[buffer(0)]],
  device void *output [[buffer(1)]],
  constant DispatchParams &params [[buffer(2)]],
  constant MemoryCast &mem_cast [[buffer(3)]],
  constant UnaryOperationType &op_type [[buffer(4)]],
  uint tid [[thread_position_in_grid]]
) {
  if (params.read_scalar_broadcast) {
    
  }
}




