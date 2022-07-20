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

enum MemoryCast: ushort {
  f32_i32_native,
  f16_as_f32,
  i8_as_i32,
  i16_as_i32,
  u8_as_i32,
  u16_as_i32,
  // `bool` can be masked as either `i8` or `u8`.
};

struct DispatchParams {
  bool read_scalar_broadcast;
  ushort read_size;
  MemoryCast memory_cast;
  ushort num_ops;
  ushort write_size;
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
  abs_f32, // 0
  abs_i32, // 1 - integer operator
  acos_f32, // 2
  acosh_f32, // 3
  asin_f32, // 4
  asinh_f32, // 5
  atan_f32, // 6
  atanh_f32, // 7
  
  cast_i32_to_f16, // 10
  cast_i32_to_f32, // 11
  cast_f32_to_f16, // 12
  cast_f32_to_i8, // 13
  cast_f32_to_i16, // 14
  cast_f32_to_i32, // 15
  cast_f32_to_u8, // 16
  cast_f32_to_u16, // 17
  cast_i32_to_u8, // 18
  cast_i32_to_u16, // 19
  
  // Skipping 20-29 to give `cast` room to expand.
  
  ceil_f32, // 30
  cos_f32, // 31
  cosh_f32, // 32
  elu_f32, // 33
  exp_f32, // 34
  expm1_f32, // 35
  floor_f32, // 36
  
  is_finite_f32, // 40 - returns bool/u8
  is_inf_f32, // 41 - returns bool/u8
  is_nan_f32, // 42 - returns bool/u8
  
  leaky_relu_f32, // 50
  log_f32, // 51
  log1p_f32, // 52
  neg_f32, // 53
  neg_i32, // 54 - integer operator
  relu_f32, // 55
  relu6_f32, // 56
  round_f32, // 57
  
  rsqrt_f32, // 60
  selu_f32, // 61
  sigmoid_f32, // 62
  sign_f32, // 63
  sign_i32, // 64 - integer operator
  sin_f32, // 65
  sinh_f32, // 66
  softplus_f32, // 67
  
  softsign_f32, // 70
  sqrt_f32, // 71
  square_f32, // 72
  square_i32, // 73 - integer operator
  tan_f32, // 74
  tanh_f32, // 75
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
  
  uchar4 get_vector_u8() const {
    return as_type<uchar4>(data[0]);
  }
  
  ushort4 get_vector_u16() const {
    uint2 out(data[0], data[1]);
    return as_type<ushort4>(out);
  }
  
  uint4 get_vector_u32() const {
    return data;
  }
};

class Storage {
  uint4 data;
  
public:
  // Memory cast setters
  
  void set_f32_i32(CompressedStorage storage) {
    data = storage.get_vector_u32();
  }
  
  void set_f16(CompressedStorage storage) {
    half4 in = as_type<half4>(storage.get_vector_u16());
    float4 casted = float4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_i8(CompressedStorage storage) {
    char4 in = as_type<char4>(storage.get_vector_u8());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_i16(CompressedStorage storage) {
    short4 in = as_type<short4>(storage.get_vector_u16());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_u8(CompressedStorage storage) {
    uchar4 in = as_type<uchar4>(storage.get_vector_u8());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_u16(CompressedStorage storage) {
    ushort4 in = as_type<ushort4>(storage.get_vector_u16());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  // Instruction execution utilities
  
  void set_f32(float4 input) {
    data = as_type<uint4>(input);
  }
  
  void set_i32(int4 input) {
    data = as_type<uint4>(input);
  }
  
  float4 get_f32() const {
    return as_type<float4>(data);
  }
  
  int4 get_i32() const {
    return as_type<int4>(data);
  }
  
  // Memory writing utilities
  
  uint4 get_vector_u32() const {
    return data;
  }
};

// MARK: - Shader Function

kernel void unary_f32_i32_new(
  device void *input [[buffer(0)]],
  device void *output [[buffer(1)]],
  constant DispatchParams &params [[buffer(2)]],
  constant UnaryOperationType *op_types [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  uint read_pos = params.read_scalar_broadcast ? 0 : tid;
  CompressedStorage compressed_storage;
  if (params.read_scalar_broadcast) {
    uint mem_slice_u32 = ((device uint*)input)[0];
    switch (params.read_size) {
      case 1: {
        uchar mem_slice = uchar(mem_slice_u32);
        compressed_storage.set_scalar_u8(mem_slice);
        break;
      }
      case 2: {
        ushort mem_slice = ushort(mem_slice_u32);
        compressed_storage.set_scalar_u16(mem_slice);
        break;
      }
      case 4: {
        uint mem_slice = uint(mem_slice_u32);
        compressed_storage.set_scalar_u32(mem_slice);
        break;
      }
    }
  } else {
    switch (params.read_size) {
      case 1: {
        uchar4 mem_slice = ((device uchar4*)input)[read_pos];
        compressed_storage.set_vector_u8(mem_slice);
        break;
      }
      case 2: {
        ushort4 mem_slice = ((device ushort4*)input)[read_pos];
        compressed_storage.set_vector_u16(mem_slice);
        break;
      }
      case 4: {
        uint4 mem_slice = ((device uint4*)input)[read_pos];
        compressed_storage.set_vector_u32(mem_slice);
        break;
      }
    }
  }
  
  Storage storage;
  switch (params.memory_cast) {
    case f32_i32_native: {
      storage.set_f32_i32(compressed_storage);
      break;
    }
    case f16_as_f32: {
      storage.set_f16(compressed_storage);
      break;
    }
    case i8_as_i32: {
      storage.set_i8(compressed_storage);
      break;
    }
    case i16_as_i32: {
      storage.set_i16(compressed_storage);
      break;
    }
    case u8_as_i32: {
      storage.set_u8(compressed_storage);
      break;
    }
    case u16_as_i32: {
      storage.set_u16(compressed_storage);
      break;
    }
  }
}
