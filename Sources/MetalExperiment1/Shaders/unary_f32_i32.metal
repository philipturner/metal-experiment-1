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

// MARK: - Enumerations

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

// Cast operation
// - X = must modify the bits
// - . = no-op
//
// Horizontal (top) axis is input, with integers masked as i32.
// Vertical (left) axis is output, with integers masked as i32.
//
//     | i8  | i16 | i32 | u8  | u16 | f16 | f32 |
// ----|-----|-----|-----|-----|-----|-----|-----|
// i8  |  .  |  X  |  X  |  .  |  X  |  X  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// i16 |  .  |  .  |  X  |  .  |  .  |  X  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// i32 |  .  |  .  |  .  |  .  |  .  |  X  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// u8  |  .  |  X  |  X  |  .  |  X  |  X  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// u16 |  .  |  .  |  X  |  .  |  .  |  X  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// f16 |  X  |  X  |  X  |  X  |  X  |  .  |  X  |
// ----|-----|-----|-----|-----|-----|-----|-----|
// f32 |  X  |  X  |  X  |  X  |  X  |  .  |  .  |
// ----|-----|-----|-----|-----|-----|-----|-----|
//
// Unique operations:
// - (i8/i16/i32/u8/u16) -> f16 = (i32) -> f16
// - (i8/i16/i32/u8/u16) -> f32 = (i32) -> f32
// - (f32) -> (f16)
// - (f16/f32) -> i8 = (f32) -> i8
// - (f16/f32) -> i16 = (f32) -> i16
// - (f16/f32) -> i32 = (f32) -> i32
// - (f16/f32) -> u8 = (f32) -> u8
// - (f16/f32) -> u16 = (f32) -> u16
// - (i16/i32/u16) -> (i8/u8) = (i32) -> u8
// - (i32) -> (i16/u16) = (i32) -> u16

enum UnaryOperationType: ushort {
  // Casts may occur in multiple instructions because of a large number of permutations.
  cast_i32_to_f16,
  cast_i32_to_f32,
  
  
//  expand_i8_to_i32, // no-op
//  expand_i16_to_i32, // no-op
//  expand_i32_to_i32,
//
//  _f32_to_f16,
//  cast_f32_to_i32,
//  cast_i32_f32,
};

// MARK: - Classes

class CompressedStorage {
  uint4 data;
  
public:
  // Scalar setters
  
  void set_scalar_u8(uchar mem_slice) {
    set_vector_u8(uchar4(mem_slice));
  }
  
  void set_scalar_u16(ushort mem_slice) {
    set_vector_u16(ushort4(mem_slice));
  }
  
  void set_scalar_u32(uint mem_slice) {
    set_vector_u32(uint4(mem_slice));
  }
  
  // Vector setters
  
  void set_vector_u8(uchar4 mem_slice) {
    data[0] = as_type<uint>(mem_slice);
  }
  
  void set_vector_u16(ushort4 mem_slice) {
    data[0] = as_type<uint2>(mem_slice)[0];
    data[1] = as_type<uint2>(mem_slice)[1];
  }
  
  void set_vector_u32(uint4 mem_slice) {
    data = mem_slice;
  }
  
  // Vector getters
  
  uchar4 get_vector_u8() {
    return as_type<uchar4>(data[0]);
  }
  
  ushort4 get_vector_u16() {
    uint2 out(data[0], data[1]);
    return as_type<ushort4>(out);
  }
  
  uint4 get_vector_u32() {
    return data;
  }
};

class Storage {
  uint4 data;
  
public:
  
  
  void set_f32_i32(CompressedStorage storage) {
    data = storage.get_vector_u32();
  }
  
  void set_f16(CompressedStorage storage) {
    half4 in = as_type<half4>(storage.get_vector_u16());
    float4 casted = float4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_i8(CompressedStorage storage) {
    
  }
  
  void set_i16(CompressedStorage storage) {
    
  }
  
  void set_u8(CompressedStorage storage) {
    
  }
  
  void set_u16(CompressedStorage storage) {
    
  }
  
};

// MARK: - Shader Function

kernel void unary_f32_i32_new(
  device void *input [[buffer(0)]],
  device void *output [[buffer(1)]],
  constant DispatchParams &params [[buffer(2)]],
  constant MemoryCast &mem_cast [[buffer(3)]],
  constant UnaryOperationType &op_type [[buffer(4)]],
  uint tid [[thread_position_in_grid]]
) {
  Storage storage;
  if (params.read_scalar_broadcast) {
    switch (params.read_size) {
    case 1:
      break;
    case 2:
      break;
    case 4:
      break;
    }
  } else {
    
  }
}
