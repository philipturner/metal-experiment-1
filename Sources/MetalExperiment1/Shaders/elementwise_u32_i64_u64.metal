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

enum MemoryCast: ushort {
  i64_u64_native = 0,
  i32_as_i64 = 1,
  i16_as_i64 = 2,
  i8_as_i64 = 3,
  
  u32_as_i64 = 4,
  u16_as_i64 = 5,
  u8_as_i64 = 6,
  f32_padded = 7,
  f16_as_f32_padded = 8,
};

struct ReadParams {
  // (1 << 7) bit marks whether it's scalar broadcasting. Lowest bits mark # bytes per element.
  ushort layout;
  MemoryCast memory_cast;
};

struct DispatchParams {
  ReadParams read_params[3];
  ushort num_inputs;
  ushort num_operations;
  MemoryCast write_memory_cast;
};

// TODO: Investigate whether to include native u32 instructions.
enum ElementwiseOperationType: ushort {
  // Unary (0 - 999)
  
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
  cast_i64_u64_to_bool = 14,
  
  cast_f32_to_u64 = 20,
  cast_u64_to_f16 = 21,
  cast_u64_to_f32 = 22,
  cast_i64_u64_to_i32 = 23, // requires metadata
  cast_i64_u64_to_u32 = 24, // requires metadata
  
  scalar_add_i64 = 30, // requires metadata
  scalar_mul_i64 = 31, // requires metadata
  scalar_mul_u64 = 32, // requires metadata
  
  // Binary (1000 - 1999)
  
  add_i64 = 1000,
  add_u64 = 1001,
  
  // Ternary (2000 - 2999)
  
  clip_by_value_i64 = 2000,
  clip_by_value_u64 = 2001,
};

// MARK: - Virtual Assembly Registers

class CompressedRegister {
  ulong2 data;
  
public:
  // Scalar setters
  
  void set_scalar_u8(uchar mem_slice) {
    set_vector_u8(uchar2(mem_slice));
  }
  
  void set_scalar_u16(ushort mem_slice) {
    set_vector_u16(ushort2(mem_slice));
  }
  
  void set_scalar_u32(uint mem_slice) {
    set_vector_u32(uint2(mem_slice));
  }
  
  void set_scalar_u64(ulong mem_slice) {
    set_vector_u64(ulong2(mem_slice));
  }
  
  // Vector setters
  
  void set_vector_u8(uchar2 mem_slice) {
    data[0] = as_type<ushort>(mem_slice);
  }
  
  void set_vector_u16(ushort2 mem_slice) {
    data[0] = as_type<uint>(mem_slice);
  }
  
  void set_vector_u32(uint2 mem_slice) {
    data[0] = as_type<ulong>(mem_slice);
  }
  
  void set_vector_u64(ulong2 mem_slice) {
    data = mem_slice;
  }
  
  // Vector getters
  
  uchar2 get_vector_u8() const {
    ushort mask = as_type<ushort4>(data[0])[0];
    return as_type<uchar2>(mask);
  }
  
  ushort2 get_vector_u16() const {
    uint mask = as_type<uint2>(data[0])[0];
    return as_type<ushort2>(mask);
  }
  
  uint2 get_vector_u32() const {
    return as_type<uint2>(data[0]);
  }
  
  ulong2 get_vector_u64() const {
    return data;
  }
};

class Register {
  ulong2 data;
  
public:
  // Memory cast setters
  
  void set_vector_i64_u64(CompressedRegister compressed) {
    data = compressed.get_vector_u64();
  }
  
  void set_vector_i32(CompressedRegister compressed) {
    int2 in = as_type<int2>(compressed.get_vector_u32());
    long2 casted = long2(in);
    data = as_type<ulong2>(casted);
  }
  
  void set_vector_i16(CompressedRegister compressed) {
    short2 in = as_type<short2>(compressed.get_vector_u16());
    long2 casted = long2(in);
    data = as_type<ulong2>(casted);
  }
  
  void set_vector_i8(CompressedRegister compressed) {
    char2 in = as_type<char2>(compressed.get_vector_u8());
    long2 casted = long2(in);
    data = as_type<ulong2>(casted);
  }
  
  void set_vector_u32(CompressedRegister compressed) {
    data = ulong2(compressed.get_vector_u32());
  }
  
  void set_vector_u16(CompressedRegister compressed) {
    data = ulong2(compressed.get_vector_u16());
  }
  
  void set_vector_u8(CompressedRegister compressed) {
    data = ulong2(compressed.get_vector_u8());
  }
  
  void set_vector_f32(CompressedRegister compressed) {
    float2 in = as_type<float2>(compressed.get_vector_u32());
    data[0] = as_type<ulong>(in);
  }
  
  void set_vector_f16(CompressedRegister compressed) {
    half2 in = as_type<half2>(compressed.get_vector_u16());
    float2 casted = float2(in);
    data[0] = as_type<ulong>(casted);
  }
  
  // Memory cast getters
  
  ulong2 get_vector_i64_u64() const {
    return data;
  }
  
  uint2 get_vector_i32_u32() const {
    return uint2(data);
  }
  
  ushort2 get_vector_i16_u16() const {
    return ushort2(data);
  }
  
  uchar2 get_vector_i8_u8() const {
    return uchar2(data);
  }
  
  uint2 get_vector_f32() const {
    return as_type<uint2>(data[0]);
  }
  
  ushort2 get_vector_f16() const {
    float2 out = as_type<float2>(data[0]);
    half2 casted = half2(out);
    return as_type<ushort2>(casted);
  }
  
  // Instruction execution utilities
  
  void set_i64(long2 input) {
    data = as_type<ulong2>(input);
  }
  
  void set_u64(ulong2 input) {
    data = input;
  }
  
  void set_f32(float2 input) {
    data[0] = as_type<ulong>(input);
  }
  
  long2 get_i64() const {
    return as_type<long2>(data);
  }
  
  ulong2 get_u64() const {
    return data;
  }
  
  float2 get_f32() const {
    return as_type<float2>(data[0]);
  }
};

// MARK: - Shader Function Utilities

#define SET_I64(expr)    \
register1.set_i64(expr); \
break;                   \

#define SET_U64(expr)    \
register1.set_u64(expr); \
break;                   \

#define SET_F32(expr)    \
register1.set_f32(expr); \
break;                   \

// Bytes of metadata allowed per operation.
constant ushort METADATA_BYTES = 8;

// Warning: `index` is a captured mutable reference.
inline constant void* get_metadata(constant void *metadata, thread ushort &index) {
  ushort byte_offset = index * METADATA_BYTES;
  index += 1;
  return (constant uchar*)metadata + byte_offset;
}

kernel void elementwise_u32_i64_u64(
  constant DispatchParams &params [[buffer(0)]],
  constant ElementwiseOperationType *operations [[buffer(1)]],
  constant void *metadata [[buffer(2)]],
  device void *input1 [[buffer(3)]],
  device void *input2 [[buffer(4)]],
  device void *input3 [[buffer(5)]],
  device void *output [[buffer(6)]],
  uint tid [[thread_position_in_grid]]
) {
  
}
