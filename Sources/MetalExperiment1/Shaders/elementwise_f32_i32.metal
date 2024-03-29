//
//  elementwise_f32_i32.metal
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

// Limit write alignment to 16B, taking a slight performance hit on the u32/i64/u64 ubershader: it
// processes two elements at once, while the f32/i32 shader processes four. 16B is a special number:
// the alignment of `malloc` pointers and memory alignment of all Swift SIMD types.

enum MemoryCast: ushort {
  f32_i32_native = 0,
  f16_as_f32 = 1,
  i8_as_i32 = 2,
  i16_as_i32 = 3,
  u8_as_i32 = 4,
  u16_as_i32 = 5,
  // `bool` can be masked as either `i8` or `u8`.
};

struct ReadParams {
  // (1 << 7) bit marks whether it's scalar broadcasting. Lowest bits mark # bytes per element.
  ushort layout;
  MemoryCast memory_cast;
};

struct DispatchParams {
  ReadParams read_params[4];
  ushort num_inputs;
  ushort num_operations;
  MemoryCast write_memory_cast;
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
// - (i16/i32/u16) -> (i8) = (i32) -> i8
// - (i16/i32/u16) -> (u8) = (i32) -> u8
// - (i32) -> (i16) = (i32) -> i16
// - (i32) -> (u16) = (i32) -> u16
//
// Additional casts not listed in the table:
// - (f16/f32) -> bool = (f32) -> bool
// - (i8/i16/i32/u8/u16) -> bool = (i32) -> bool
//
// The (f16/f32) -> (i8/i16/i32/u8/u16) casts can be represented by clamping. An operation can come
// with some metadata, so use the metadata to provide min/max bounds. This eliminates the need to
// have multiple instructions for very similar casts. `cast_f32_to_i32` encapsulates all of these
// casts.
//
// Casting integer types to other integer types falls in the same boat. It can all be emulated with
// some masking. Casting to `bool` behaves differently than casting to integers. It returns 1 if the
// if the input is nonzero, regardless of the bits.

enum ElementwiseOperationType: ushort {
  // Unary (0 - 999)
  
  abs_f32 = 0,
  abs_i32 = 1, // integer operation
  acos_f32 = 2,
  acosh_f32 = 3,
  asin_f32 = 4,
  asinh_f32 = 5,
  atan_f32 = 6,
  atanh_f32 = 7,
  
  cast_f32_to_f16 = 10,
  cast_f32_to_bool = 11,
  cast_f32_to_i32 = 12, // requires metadata
  cast_i32_to_f16 = 13,
  cast_i32_to_f32 = 14,
  cast_i32_to_bool = 15,
  cast_i32_to_i32 = 16, // requires metadata
  
  ceil_f32 = 20,
  cos_f32 = 21,
  cosh_f32 = 22,
  elu_f32 = 23,
  exp_f32 = 24,
  expm1_f32 = 25,
  floor_f32 = 26,
  
  is_finite_f32 = 30, // returns bool
  is_inf_f32 = 31, // returns bool
  is_nan_f32 = 32, // returns bool
  
  // Extra conditional branch here to incrementally improve GPU-side performance. This only has a
  // measure impact (3.9 µs -> 3.2 µs) when fusing dozens of unary operations. It comes just before
  // `neg_f32` (a commonly fused operation), probably improving real-world performance more than if
  // it came afterward.
  
  leaky_relu_f32 = 40, // requires metadata
  log_f32 = 41,
  log1p_f32 = 42,
  logical_not_bool = 43, // boolean operation
  neg_f32 = 44,
  neg_i32 = 45, // integer operation
  relu_f32 = 46,
  relu6_f32 = 47,
  round_f32 = 48, // rounds to nearest even
  
  rsqrt_f32 = 50,
  selu_f32 = 51,
  sigmoid_f32 = 52,
  sign_f32 = 53,
  sign_i32 = 54, // integer operation
  sin_f32 = 55,
  sinh_f32 = 56,
  softplus_f32 = 57,
  
  softsign_f32 = 60,
  sqrt_f32 = 61,
  square_f32 = 62,
  square_i32 = 63, // integer operation
  tan_f32 = 64,
  tanh_f32 = 65,
  
  scalar_add_f32 = 70, // requires metadata
  scalar_sub_f32 = 71, // requires metadata
  scalar_sub_inverse_f32 = 72, // requires metadata
  scalar_mul_f32 = 73, // requires metadata
  scalar_div_f32 = 74, // requires metadata
  scalar_div_inverse_f32 = 75, // requires metadata
  
  scalar_add_i32 = 80, // requires metadata
  scalar_sub_i32 = 81, // requires metadata
  scalar_sub_inverse_i32 = 82, // requires metadata
  scalar_mul_i32 = 83, // requires metadata
  scalar_div_i32 = 84, // requires metadata
  scalar_div_inverse_i32 = 85, // requires metadata
  
  // Binary (1000 - 1999)
  
  add_f32 = 1000,
  add_i32 = 1001,
  approximate_equal_f32 = 1002, // requires metadata
  comparison_f32 = 1003, // requires metadata
  comparison_i32 = 1004, // requires metadata
  
  div_f32 = 1010,
  div_i32 = 1011,
  elu_grad_f32 = 1012,
  leaky_relu_grad_f32 = 1013, // requires metadata
  logical_and_bool = 1014,
  logical_or_bool = 1015,
  
  maximum_f32 = 1020,
  maximum_i32 = 1021,
  minimum_f32 = 1022,
  minimum_i32 = 1023,
  mod_f32 = 1024,
  mod_i32 = 1025,
  
  mul_f32 = 1030,
  mul_i32 = 1031,
  pow_f32 = 1032,
  relu6_grad_f32 = 1033,
  relu_grad_f32 = 1034,
  
  rsqrt_grad_f32 = 1040,
  selu_grad_f32 = 1041,
  sigmoid_grad_f32 = 1042,
  softplus_grad_f32 = 1043,
  softsign_grad_f32 = 1044,
  
  squared_difference_f32 = 1050,
  squared_difference_i32 = 1051,
  sub_f32 = 1052,
  sub_i32 = 1053,
  xdivy_f32 = 1054,
  
  // Ternary (2000 - 2999)
  
  clip_by_value_f32 = 2000,
  clip_by_value_i32 = 2001,
  select_f32_i32 = 2002,
  
  // Other (3000+)
  
  swap_registers_1_2 = 3000,
  swap_registers_1_3 = 3001,
  swap_registers_1_4 = 3002,
  
  swap_registers_2_3 = 3010,
  swap_registers_2_4 = 3011,
  swap_registers_3_4 = 3012,
};

// MARK: - Virtual Assembly Registers

class CompressedRegister {
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

class Register {
  uint4 data;
  
public:
  // Memory cast setters
  
  void set_vector_f32_i32(CompressedRegister compressed) {
    data = compressed.get_vector_u32();
  }
  
  void set_vector_f16(CompressedRegister compressed) {
    half4 in = as_type<half4>(compressed.get_vector_u16());
    float4 casted = float4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_i8(CompressedRegister compressed) {
    char4 in = as_type<char4>(compressed.get_vector_u8());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_i16(CompressedRegister compressed) {
    short4 in = as_type<short4>(compressed.get_vector_u16());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_u8(CompressedRegister compressed) {
    data = uint4(compressed.get_vector_u8());
  }
  
  void set_vector_u16(CompressedRegister compressed) {
    data = uint4(compressed.get_vector_u16());
  }
  
  // Memory cast getters
  
  uint4 get_vector_f32_i32() const {
    return data;
  }
  
  ushort4 get_vector_f16() const {
    float4 out = as_type<float4>(data);
    half4 casted = half4(out);
    return as_type<ushort4>(casted);
  }
  
  uchar4 get_vector_i8_u8() const {
    return uchar4(data);
  }
  
  ushort4 get_vector_i16_u16() const {
    return ushort4(data);
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
};

// MARK: - Shader Function Utilities

#define SET_F32(expr)    \
register1.set_f32(expr); \
break;                   \

#define SET_I32(expr)    \
register1.set_i32(expr); \
break;                   \

// NOTE: For gradient operations, gradient always goes in register 1. This reduces the amount of
// register swapping when fusing two adjacent gradient ops. Carefully inspect the `_Raw` namespace
// to ensure correct order of arguments.
//
// There was once a helper macro for setting gradients, but I removed it. This forces me to declare
// which register the gradient comes from.
//
//#define SET_BINARY_GRADIENT_F32(expr) \
//SET_F32(expr * register1.get_f32())   \

#define GET_SET_UNARY_F32(expr)    \
SET_F32(expr(register1.get_f32())) \

#define GET_SET_UNARY_I32(expr)    \
SET_I32(expr(register1.get_i32())) \

#define GET_SET_BINARY_F32(expr)                        \
SET_F32(expr(register1.get_f32(), register2.get_f32())) \

#define GET_SET_BINARY_I32(expr)                        \
SET_I32(expr(register1.get_i32(), register2.get_i32())) \

#define GET_SET_BINARY_INFIX_F32(infix_expr)                \
SET_F32(register1.get_f32() infix_expr register2.get_f32()) \

#define GET_SET_BINARY_INFIX_I32(infix_expr)                \
SET_I32(register1.get_i32() infix_expr register2.get_i32()) \

// Bytes of metadata allowed per operation.
constant ushort METADATA_BYTES = 8;

// Warning: `index` is a captured mutable reference.
inline constant void* get_metadata(constant void *metadata, thread ushort &index) {
  ushort byte_offset = index * METADATA_BYTES;
  index += 1;
  return (constant uchar*)metadata + byte_offset;
}

namespace metal {
  namespace precise {
    // `inline` prevents symbol duplication errors in Xcode.
    inline float4 expm1(float4 x) {
      return precise::exp(x) - 1;
    }
  }
}

// MARK: - Shader Function

// Performance of existing bucketed loop:
// Iteration and instruction fetching: >=2 clock cycles
// First switch statement: 1 - 7 comparisons (8 execution paths)
// Second switch statement: 1 - 8 comparisons (2 - 9 execution paths)
// Overhead range: 4 - 17 clock cycles
// Overhead per element: 1 - 4.25 clock cycles
// Average overhead: (2 + (1+2+...7+7)/8 + (1+2+...5+5)/6 + 0.25)/4 = 2.49 clock cycles
// - The '6' in the second series comes from ((#operations=50)/(1st switch=8)) = 6.25 ≈ 6. The
//   '0.25' makes up for the rounding-down during addition.
//
// Overhead for `increment`: 8/4 = 2 clock cycles
// Amortized sequential throughput for `increment_f32`: 2.6 µs
//
// Performance of perfect radix-4 dispatch:
// Iteration and instruction fetching: >=2 clock cycles
// <=64 unique operations, so 3 levels of recursion
// Overhead per level of recursion: 1 - 3 comparisons (4 execution paths)
// Overhead range: 5 - 11 clock cycles
// Overhead per element: 1.25 - 2.75 clock cycles
// Average overhead: (2 + 3 * (1+2+3+3)/4)/4 = 2.19 clock cycles
//
// Overhead for `increment`: (#49 -> 3 + 1 + 2 = 6)/4 = 1.5 clock cycles
// Amortized sequential throughput for `increment_f32`: 2.4 µs
// - For 1.4 µs, before making ubershader, execution time was (0 overhead) + (2 ALU) = 2
// - For 2.6 µs, after making ubershader, execution time was (2 overhead) + (2 ALU) = 4
// - With perfect radix-4 dispatch, execution time should be (1.5 overhead) + (2 ALU) = 3.5
// - 3.5/4 = 87.5%, rounded down to 87%
// - Using a linear interpolation of (1.5=overhead), (13% x 1.4) + (87% x 2.6) = 2.44 µs
//
//===------------------------------------------------------------------------------------------===//
//
// Perfect radix-4 overhead takes an extrapolated 93.8% of the execution time for `increment`. In a
// quick calculation that optimized the pre-ubershader increment (changing 1.4 -> 0.7 µs), the
// percentage was 90.5%. Keeping pre-ubershader = 1.4 µs, results with different combos:
//
// - Horizontal axis = overhead reduction
//   - A = Δ(upper bounds) = -1.5 cycles  = 4.25 -> 2.75
//   - B = Δ(average)      = -0.3 cycles  = 2.49 -> 2.19
//   - C = Δ(lower bounds) = +0.25 cycles = 1.00 -> 1.25
// - Vertical axis = clocks/operation
//   - D = abs_f32 = 1
//   - E = increment_f32 = 2
//   - F = estimated trig = 10
//
// Absolute clock cycle ratio (increment = 0.87)
//    |  A  |  B  |  C  |
// ---|-----|-----|------|
//  D | .71 | .91 | 1.13 |
// ---|-----|-----|------|
//  E | .76 | .93 | 1.08 |
// ---|-----|-----|------|
//  F | .89 | .98 | 1.02 |
// ---|-----|-----|------|
//
// The results are very disappointing. Operations taking >10 cycles/element have worse sequential
// throughput than cheap ones, making the linear regression for sequential throughput useless. If I
// ran the calculation, results might be 10% speedup at best. This marginal benefit is outweighed by
// the costs of making the ubershader unmaintable. The doubling of amortized execution time from
// 1.4 µs to 2.6 µs seems alarming, but I can't do much about it.
//
// Furthermore, these numbers assume around 100 operations have been fused. In real-world code,
// fusion is not that common. If only 10 operations were fused, the benefits of switching to perfect
// radix-4 could reduce by a factor of 10. If 2 operations were fused (the most common case), the
// benefits reduce by a factor of 50.
//
// Update: The `increment` operation was changed to a `scalar_add` operation that fetches its right-
// hand side from metadata. This will be used in real-world code, speeding up some addition
// operations. The sequential throughput benchmark now reads 3.2 µs instead of 2.6 µs.

kernel void elementwise_f32_i32(
  constant DispatchParams &params [[buffer(0)]],
  constant ElementwiseOperationType *operations [[buffer(1)]],
  constant void *metadata [[buffer(2)]],
  device void *input1 [[buffer(3)]],
  device void *input2 [[buffer(4)]],
  device void *input3 [[buffer(5)]],
  device void *input4 [[buffer(6)]],
  device void *output [[buffer(7)]],
  uint tid [[thread_position_in_grid]]
) {
  Register register1;
  Register register2;
  Register register3;
  Register register4;
  for (int i = 0; i < params.num_inputs; ++i) {
    ReadParams read_params = params.read_params[i];
    CompressedRegister compressed;
    
    device void *input;
    switch (i) {
      case 0: {
        input = input1;
        break;
      }
      case 1: {
        input = input2;
        break;
      }
      case 2: {
        input = input3;
        break;
      }
      default: /*3*/ {
        input = input4;
        break;
      }
    }
    if (read_params.layout & 128) {
      uint mem_slice_u32 = ((device uint*)input)[0];
      switch (read_params.layout) {
        case 128 + 1: {
          uchar mem_slice = uchar(mem_slice_u32);
          compressed.set_scalar_u8(mem_slice);
          break;
        }
        case 128 + 2: {
          ushort mem_slice = ushort(mem_slice_u32);
          compressed.set_scalar_u16(mem_slice);
          break;
        }
        default: /*128 + 4*/ {
          uint mem_slice = uint(mem_slice_u32);
          compressed.set_scalar_u32(mem_slice);
          break;
        }
      }
    } else {
      switch (read_params.layout) {
        case 1: {
          uchar4 mem_slice = ((device uchar4*)input)[tid];
          compressed.set_vector_u8(mem_slice);
          break;
        }
        case 2: {
          ushort4 mem_slice = ((device ushort4*)input)[tid];
          compressed.set_vector_u16(mem_slice);
          break;
        }
        default: /*4*/ {
          uint4 mem_slice = ((device uint4*)input)[tid];
          compressed.set_vector_u32(mem_slice);
          break;
        }
      }
    }
    
    Register expanded;
    switch (read_params.memory_cast) {
      case f32_i32_native: {
        expanded.set_vector_f32_i32(compressed);
        break;
      }
      case f16_as_f32: {
        expanded.set_vector_f16(compressed);
        break;
      }
      case i8_as_i32: {
        expanded.set_vector_i8(compressed);
        break;
      }
      case i16_as_i32: {
        expanded.set_vector_i16(compressed);
        break;
      }
      case u8_as_i32: {
        expanded.set_vector_u8(compressed);
        break;
      }
      default: /*u16_as_i32*/ {
        expanded.set_vector_u16(compressed);
        break;
      }
    }
    switch (i) {
      case 0: {
        register1 = expanded;
        break;
      }
      case 1: {
        register2 = expanded;
        break;
      }
      case 2: {
        register3 = expanded;
      }
      default: /*3*/ {
        register4 = expanded;
        break;
      }
    }
  }
  
  ushort metadata_index = 0;
  
  // pc = program counter
  for (ushort pc = 0; pc < params.num_operations; ++pc) {
    ElementwiseOperationType operation = operations[pc];
    if (operation < 1000) {
      // MARK: - Unary
      if (operation < 40) {
      if (operation <= atanh_f32) {
        switch (operation) {
          case abs_f32: {
            GET_SET_UNARY_F32(precise::abs)
          }
          case abs_i32: {
            GET_SET_UNARY_I32(abs)
          }
          case acos_f32: {
            GET_SET_UNARY_F32(precise::acos)
          }
          case acosh_f32: {
            GET_SET_UNARY_F32(precise::acosh)
          }
          case asin_f32: {
            GET_SET_UNARY_F32(precise::asin)
          }
          case asinh_f32: {
            GET_SET_UNARY_F32(precise::asinh)
          }
          case atan_f32: {
            GET_SET_UNARY_F32(precise::atan)
          }
          default: /*atanh_f32*/ {
            GET_SET_UNARY_F32(precise::atanh)
          }
        }
      } else if (operation <= cast_i32_to_i32) {
        switch (operation) {
          case cast_f32_to_f16: {
            auto x = register1.get_f32();
            auto casted = float4(half4(x));
            SET_F32(casted)
          }
          case cast_f32_to_bool: {
            auto x = register1.get_f32();
            auto casted = int4(bool4(x));
            SET_I32(casted)
          }
          case cast_f32_to_i32: {
            auto x = register1.get_f32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            int2 bounds = ((constant int2*)operation_metadata)[0];
            
            auto casted = int4(x);
            casted = clamp(casted, bounds[0], bounds[1]);
            SET_I32(casted)
          }
          case cast_i32_to_f16: {
            auto x = register1.get_i32();
            auto casted = float4(half4(x));
            SET_F32(casted)
          }
          case cast_i32_to_f32: {
            auto x = register1.get_i32();
            auto casted = float4(x);
            SET_F32(casted)
          }
          case cast_i32_to_bool: {
            auto x = register1.get_i32();
            auto casted = int4(bool4(x));
            SET_I32(casted)
          }
          default: /*cast_i32_to_i32*/ {
            auto x = register1.get_i32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            int2 masks = ((constant int2*)operation_metadata)[0];
            x &= int4(masks[0]); // truncate
            
            if (masks[1] != 0) { // sign extend
              int inverted_mask = ~masks[0];
              for (int i = 0; i < 4; ++i) {
                // Sign mask has one bit activated.
                if (x[i] & masks[1]) {
                  x[i] |= inverted_mask;
                }
              }
            }
            SET_I32(x)
          }
        }
      } else if (operation <= floor_f32) {
        switch (operation) {
          case ceil_f32: {
            GET_SET_UNARY_F32(precise::ceil)
          }
          case cos_f32: {
            GET_SET_UNARY_F32(precise::cos)
          }
          case cosh_f32: {
            GET_SET_UNARY_F32(precise::cosh)
          }
          case elu_f32: {
            auto x = register1.get_f32();
            x = select(x, precise::expm1(x), x < 0);
            SET_F32(x)
          }
          case exp_f32: {
            GET_SET_UNARY_F32(precise::exp)
          }
          case expm1_f32: {
            GET_SET_UNARY_F32(precise::expm1)
          }
          default: /*floor_f32*/ {
            GET_SET_UNARY_F32(precise::floor)
          }
        }
      } else /*(operation <= is_nan_f32)*/ {
        switch (operation) {
          case is_finite_f32: {
            auto x = register1.get_f32();
            auto mask = int4(isfinite(x));
            SET_I32(mask)
          }
          case is_inf_f32: {
            auto x = register1.get_f32();
            auto mask = int4(isinf(x));
            SET_I32(mask)
          }
          default: /*is_nan_f32*/ {
            auto x = register1.get_f32();
            auto mask = int4(isnan(x));
            SET_I32(mask)
          }
        }
      } // if (operation < 40)
      } else /*(operation >= 40)*/ {
      // MARK: - Extra Conditional Branch
      if (operation <= round_f32) {
        switch (operation) {
          case leaky_relu_f32: {
            auto x = register1.get_f32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            float alpha = ((constant float*)operation_metadata)[0];
            x = precise::max(x, x * alpha);
            SET_F32(x)
          }
          case log_f32: {
            GET_SET_UNARY_F32(precise::log)
          }
          case log1p_f32: {
            auto x = register1.get_f32();
            SET_F32(precise::log(1 + x))
          }
          case logical_not_bool: {
            auto x = register1.get_i32();
            auto casted = bool4(x);
            auto mask = int4(!casted);
            SET_I32(mask)
          }
          case neg_f32: {
            GET_SET_UNARY_F32(-)
          }
          case neg_i32: {
            GET_SET_UNARY_I32(-)
          }
          case relu_f32: {
            auto x = register1.get_f32();
            SET_F32(precise::max(0, x))
          }
          case relu6_f32: {
            auto x = register1.get_f32();
            SET_F32(precise::clamp(x, 0, 6))
          }
          default: /*round_f32*/ {
            GET_SET_UNARY_F32(precise::rint)
          }
        }
      } else if (operation <= softplus_f32) {
        switch (operation) {
          case rsqrt_f32: {
            GET_SET_UNARY_F32(precise::rsqrt)
          }
          case selu_f32: {
            auto x = register1.get_f32();
            constexpr float ALPHA = 1.6732632423543772848170429916717;
            constexpr float SCALE = 1.0507009873554804934193349852946;
            x = select(SCALE * x, (SCALE * ALPHA) * precise::expm1(x), x < 0);
            SET_F32(x)
          }
          case sigmoid_f32: {
            auto x = register1.get_f32();
            x = 1 + precise::exp(-x);
            x = precise::divide(1, x);
            SET_F32(x)
          }
          case sign_f32: {
            GET_SET_UNARY_F32(sign)
          }
          case sign_i32: {
            auto x = register1.get_i32();
            auto mask = select(int4(1), int4(-1), x < 0);
            mask = select(mask, int4(0), x == 0);
            SET_I32(mask)
          }
          case sin_f32: {
            GET_SET_UNARY_F32(precise::sin)
          }
          case sinh_f32: {
            GET_SET_UNARY_F32(precise::sinh)
          }
          default: /*softplus_f32*/ {
            auto x = register1.get_f32();
            x = precise::exp(x) + 1;
            x = precise::log(x);
            SET_F32(x)
          }
        }
      } else if (operation <= tanh_f32) {
        switch (operation) {
          case softsign_f32: {
            auto x = register1.get_f32();
            auto denominator = precise::abs(x) + 1;
            x = precise::divide(x, denominator);
            SET_F32(x)
          }
          case sqrt_f32: {
            GET_SET_UNARY_F32(precise::sqrt)
          }
          case square_f32: {
            auto x = register1.get_f32();
            SET_F32(x * x)
          }
          case square_i32: {
            auto x = register1.get_i32();
            SET_I32(x * x)
          }
          case tan_f32: {
            GET_SET_UNARY_F32(precise::tan)
          }
          default: /*tanh_f32*/ {
            GET_SET_UNARY_F32(precise::tanh)
          }
        }
      } else if (operation <= scalar_div_inverse_f32) {
        float4 x = register1.get_f32();
        auto operation_metadata = get_metadata(metadata, metadata_index);
        float scalar = ((constant float*)operation_metadata)[0];
        switch (operation) {
          case scalar_add_f32: {
            x += scalar;
            break;
          }
          case scalar_sub_f32: {
            x -= scalar;
            break;
          }
          case scalar_sub_inverse_f32: {
            x = scalar - x;
            break;
          }
          case scalar_mul_f32: {
            x *= scalar;
            break;
          }
          case scalar_div_f32: {
            x = precise::divide(x, scalar);
            break;
          }
          default: /*scalar_div_inverse_f32*/ {
            x = precise::divide(scalar, x);
            break;
          }
        }
        register1.set_f32(x);
      } else /*(operation <= scalar_div_inverse_i32)*/ {
        auto x = register1.get_i32();
        auto operation_metadata = get_metadata(metadata, metadata_index);
        int scalar = ((constant int*)operation_metadata)[0];
        switch (operation) {
          case scalar_add_i32: {
            x += scalar;
            break;
          }
          case scalar_sub_i32: {
            x -= scalar;
            break;
          }
          case scalar_sub_inverse_i32: {
            x = scalar - x;
            break;
          }
          case scalar_mul_i32: {
            x *= scalar;
            break;
          }
          case scalar_div_i32: {
            x /= scalar;
            break;
          }
          default: /*scalar_div_inverse_i32*/ {
            x = scalar / x;
            break;
          }
        }
        register1.set_i32(x);
      }
      } // else /*(operation >= 40)*/
    } else if (operation < 2000) {
      // MARK: - Binary
      if (operation <= comparison_i32) {
        switch (operation) {
          case add_f32: {
            GET_SET_BINARY_INFIX_F32(+)
          }
          case add_i32: {
            GET_SET_BINARY_INFIX_I32(+)
          }
          case approximate_equal_f32: {
            auto x = register1.get_f32();
            auto y = register2.get_f32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            float tolerance = ((constant float*)operation_metadata)[0];
            float4 difference = abs(x - y);
            bool4 mask = bool4(difference < tolerance);
            SET_I32(int4(mask))
          }
          case comparison_f32: {
            auto x = register1.get_f32();
            auto y = register2.get_f32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            ushort2 codes = ((constant ushort2*)operation_metadata)[0];
            
            bool4 out;
            switch (codes[0]) {
              case 0:
                out = x == y;
                break;
              case 1:
                out = x < y;
                break;
              default: /*2*/ {
                out = x > y;
                break;
              }
            }
            if (codes[1] == 1) {
              out = !out;
            }
            SET_I32(int4(out))
          }
          default: /*comparison_i32*/ {
            auto x = register1.get_i32();
            auto y = register2.get_i32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            ushort2 codes = ((constant ushort2*)operation_metadata)[0];
            
            bool4 out;
            switch (codes[0]) {
              case 0:
                out = x == y;
                break;
              case 1:
                out = x < y;
                break;
              default: /*2*/ {
                out = x > y;
                break;
              }
            }
            if (codes[1] == 1) {
              out = !out;
            }
            SET_I32(int4(out))
          }
        }
      } else if (operation <= logical_or_bool) {
        switch (operation) {
          case div_f32: {
            GET_SET_BINARY_F32(precise::divide)
          }
          case div_i32: {
            GET_SET_BINARY_INFIX_I32(/)
          }
          case elu_grad_f32: {
            auto dy = register1.get_f32();
            auto y = register2.get_f32();
            auto out = select(1, y + 1, y < 0);
            SET_F32(dy * out)
          }
          case leaky_relu_grad_f32: {
            auto dy = register1.get_f32();
            auto x = register2.get_f32();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            float alpha = ((constant float*)operation_metadata)[0];
            auto out = select(1, float4(alpha), x <= 0);
            SET_F32(dy * out)
          }
          case logical_and_bool: {
            auto x = register1.get_vector_i16_u16();
            auto y = register1.get_vector_i16_u16();
            auto out = bool4(x) && bool4(y);
            SET_I32(int4(out));
          }
          default: /*logical_or_bool*/ {
            auto x = register1.get_vector_i16_u16();
            auto y = register1.get_vector_i16_u16();
            auto out = bool4(x) || bool4(y);
            SET_I32(int4(out));
          }
        }
      } else if (operation <= mod_i32) {
        switch (operation) {
          case maximum_f32: {
            GET_SET_BINARY_F32(precise::max)
          }
          case maximum_i32: {
            GET_SET_BINARY_I32(max)
          }
          case minimum_f32: {
            GET_SET_BINARY_F32(precise::min)
          }
          case minimum_i32: {
            GET_SET_BINARY_I32(min)
          }
          case mod_f32: {
            GET_SET_BINARY_F32(precise::fmod)
          }
          default: /*mod_i32*/ {
            GET_SET_BINARY_INFIX_I32(%)
          }
        }
      } else if (operation <= relu_grad_f32) {
        switch (operation) {
          case mul_f32: {
            GET_SET_BINARY_INFIX_F32(*);
          }
          case mul_i32: {
            GET_SET_BINARY_INFIX_I32(*);
          }
          case pow_f32: {
            GET_SET_BINARY_F32(precise::pow)
          }
          case relu6_grad_f32: {
            auto dy = register1.get_f32();
            auto x = register2.get_f32();
            auto out = precise::clamp(x, 0, 6);
            out = select(0, dy, x == out);
            SET_F32(out)
          }
          default: /*relu_grad_f32:*/ {
            auto dy = register1.get_f32();
            auto x = register2.get_f32();
            auto out = select(0, dy, x > 0);
            SET_F32(out)
          }
        }
      } else if (operation <= softsign_grad_f32) {
        switch (operation) {
          case rsqrt_grad_f32: {
            auto dy = register1.get_f32();
            auto y = register2.get_f32();
            auto out = -0.5 * (y * y * y);
            SET_F32(dy * out)
          }
          case selu_grad_f32: {
            auto dy = register1.get_f32();
            auto y = register2.get_f32();
            constexpr float ALPHA = 1.6732632423543772848170429916717;
            constexpr float SCALE = 1.0507009873554804934193349852946;
            auto out = select(SCALE, y + (SCALE * ALPHA), y < 0);
            SET_F32(dy * out)
          }
          case sigmoid_grad_f32: {
            auto dy = register1.get_f32();
            auto y = register2.get_f32();
            auto out = y * (1 - y);
            SET_F32(dy * out)
          }
          case softplus_grad_f32: {
            auto dy = register1.get_f32();
            auto x = register2.get_f32();
            auto out = 1 + precise::exp(-x);
            out = precise::divide(dy, out);
            SET_F32(out)
          }
          default: /*softsign_grad_f32*/ {
            auto dy = register1.get_f32();
            auto x = register2.get_f32();
            auto out = 1 + precise::abs(x);
            out = precise::divide(dy, out * out);
            SET_F32(out)
          }
        }
      } else /*(operation <= xdivy_f32)*/ {
        switch (operation) {
          case squared_difference_f32: {
            auto x = register1.get_f32();
            auto y = register2.get_f32();
            auto out = x - y;
            out = out * out;
            SET_F32(out)
          }
          case squared_difference_i32: {
            auto x = register1.get_i32();
            auto y = register2.get_i32();
            auto out = int4(absdiff(x, y));
            out = out * out;
            SET_I32(out)
          }
          case sub_f32: {
            GET_SET_BINARY_INFIX_F32(-);
          }
          case sub_i32: {
            GET_SET_BINARY_INFIX_I32(-);
          }
          default: /*xdivy_f32*/ {
            auto x = register1.get_f32();
            auto y = register2.get_f32();
            auto out = precise::divide(x, y);
            out = select(out, 0, x == 0);
            SET_F32(out);
          }
        }
      }
    } else if (operation < 3000) {
      // MARK: - Ternary
      switch (operation) {
        case clip_by_value_f32: {
          auto x = register1.get_f32();
          auto y = register2.get_f32();
          auto z = register3.get_f32();
          auto out = precise::clamp(x, y, z);
          SET_F32(out);
        }
        case clip_by_value_i32: {
          auto x = register1.get_i32();
          auto y = register2.get_i32();
          auto z = register3.get_i32();
          auto out = clamp(x, y, z);
          SET_I32(out);
        }
        default: /*select_f32_i32*/ {
          auto condition = register1.get_vector_i16_u16();
          auto true_value = register2.get_i32();
          auto else_value = register3.get_i32();
          auto out = select(else_value, true_value, bool4(condition));
          SET_I32(out);
        }
      }
    } else /*(operation >= 3000)*/ {
      // MARK: - Other
      if (operation <= swap_registers_1_4) {
        auto lhs = register1.get_i32();
        int4 rhs;
        switch (operation) {
          case swap_registers_1_2: {
            rhs = register2.get_i32();
            register2.set_i32(lhs);
            break;
          }
          case swap_registers_1_3: {
            rhs = register3.get_i32();
            register3.set_i32(lhs);
            break;
          }
          default: /*swap_registers_1_4*/ {
            rhs = register4.get_i32();
            register4.set_i32(lhs);
            break;
          }
        }
        register1.set_i32(rhs);
      } else {
        switch (operation) {
          case swap_registers_2_3: {
            auto lhs = register2.get_i32();
            auto rhs = register3.get_i32();
            register3.set_i32(lhs);
            register2.set_i32(rhs);
            break;
          }
          case swap_registers_2_4: {
            auto lhs = register2.get_i32();
            auto rhs = register4.get_i32();
            register4.set_i32(lhs);
            register2.set_i32(rhs);
            break;
          }
          default: /*swap_registers_3_4*/ {
            auto lhs = register3.get_i32();
            auto rhs = register4.get_i32();
            register4.set_i32(lhs);
            register3.set_i32(rhs);
            break;
          }
        }
      }
    }
  }
  
  switch (params.write_memory_cast) {
    case f32_i32_native: {
      uint4 mem_slice = register1.get_vector_f32_i32();
      ((device uint4*)output)[tid] = mem_slice;
      break;
    }
    case f16_as_f32: {
      ushort4 mem_slice = register1.get_vector_f16();
      ((device ushort4*)output)[tid] = mem_slice;
      break;
    }
    case i8_as_i32:
    case u8_as_i32: {
      uchar4 mem_slice = register1.get_vector_i8_u8();
      ((device uchar4*)output)[tid] = mem_slice;
      break;
    }
    default: /*i16_as_i32
               u16_as_i32*/ {
      ushort4 mem_slice = register1.get_vector_i16_u16();
      ((device ushort4*)output)[tid] = mem_slice;
      break;
    }
  }
}
