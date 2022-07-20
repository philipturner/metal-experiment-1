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
  f32_i32_native = 0,
  f16_as_f32 = 1,
  i8_as_i32 = 2,
  i16_as_i32 = 3,
  u8_as_i32 = 4,
  u16_as_i32 = 5,
  // `bool` can be masked as either `i8` or `u8`.
};

struct DispatchParams {
  // Reading
  bool read_scalar_broadcast;
  ushort read_size;
  MemoryCast read_memory_cast;
  
  // Execution
  ushort num_ops;
  
  // Writing
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
// - (i16/i32/u16) -> (i8/u8) = (i32) -> u8
// - (i32) -> (i16/u16) = (i32) -> u16
//
// The (f16/f32) -> (i8/i16/i32/u8/u16) casts can be represented by clamping. An operation can come
// with some metadata, so use the metadata to provide min/max bounds. This eliminates the need to
// have multiple instructions for very similar casts. `cast_f32_to_i32` includes all of these casts.

enum UnaryOperationType: ushort {
  abs_f32, // 0
  abs_i32, // 1 - integer operation
  acos_f32, // 2
  acosh_f32, // 3
  asin_f32, // 4
  asinh_f32, // 5
  atan_f32, // 6
  atanh_f32, // 7
  
  cast_f32_to_f16, // 10
  cast_f32_to_i32, // 11
  cast_i32_to_f16, // 12
  cast_i32_to_f32, // 13
  cast_i32_to_u8, // 14
  cast_i32_to_u16, // 15
  
  ceil_f32, // 20
  cos_f32, // 21
  cosh_f32, // 22
  elu_f32, // 23
  exp_f32, // 24
  expm1_f32, // 25
  floor_f32, // 26
  
  is_finite_f32, // 30 - returns bool/u8
  is_inf_f32, // 31 - returns bool/u8
  is_nan_f32, // 32 - returns bool/u8
  
  leaky_relu_f32, // 40
  log_f32, // 41
  log1p_f32, // 42
  neg_f32, // 43
  neg_i32, // 44 - integer operation
  relu_f32, // 45
  relu6_f32, // 46
  round_f32, // 47 - rounds to nearest even
  
  rsqrt_f32, // 50
  selu_f32, // 51
  sigmoid_f32, // 52
  sign_f32, // 53
  sign_i32, // 54 - integer operation
  sin_f32, // 55
  sinh_f32, // 56
  softplus_f32, // 57
  
  softsign_f32, // 60
  sqrt_f32, // 61
  square_f32, // 62
  square_i32, // 63 - integer operation
  tan_f32, // 64
  tanh_f32, // 65
  
  increment_f32, // 70 - for testing purposes only
  increment_i32, // 71 - for testing purposes only
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
  
  void set_vector_f32_i32(CompressedStorage storage) {
    data = storage.get_vector_u32();
  }
  
  void set_vector_f16(CompressedStorage storage) {
    half4 in = as_type<half4>(storage.get_vector_u16());
    float4 casted = float4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_i8(CompressedStorage storage) {
    char4 in = as_type<char4>(storage.get_vector_u8());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_i16(CompressedStorage storage) {
    short4 in = as_type<short4>(storage.get_vector_u16());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_u8(CompressedStorage storage) {
    uchar4 in = as_type<uchar4>(storage.get_vector_u8());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
  }
  
  void set_vector_u16(CompressedStorage storage) {
    ushort4 in = as_type<ushort4>(storage.get_vector_u16());
    int4 casted = int4(in);
    data = as_type<uint4>(casted);
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
    int4 out = as_type<int4>(data);
    return uchar4(out);
  }
  
  ushort4 get_vector_i16_u16() const {
    int4 out = as_type<int4>(data);
    return ushort4(out);
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

#define SET_F32(expr)  \
storage.set_f32(expr); \
break;                 \

#define SET_I32(expr)  \
storage.set_i32(expr); \
break;                 \

#define GET_SET_F32(expr)        \
SET_F32(expr(storage.get_f32())) \

#define GET_SET_I32(expr)        \
SET_I32(expr(storage.get_i32())) \

// Bytes of metadata allowed per operation.
constant ushort METADATA_BYTES = 8;

constant void* get_metadata(constant void *metadata, ushort pc) {
  ushort byte_offset = pc * METADATA_BYTES;
  return (constant uchar*)metadata + byte_offset;
}

namespace metal {
  namespace precise {
    inline float4 expm1(float4 x) {
      return precise::exp(x) - 1;
    }
  }
}

// MARK: - Shader Function

kernel void unary_f32_i32_new(
  device void *input [[buffer(0)]],
  device void *output [[buffer(1)]],
  constant DispatchParams &params [[buffer(2)]],
  constant UnaryOperationType *operations [[buffer(3)]],
  constant void *metadata [[buffer(4)]],
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
  switch (params.read_memory_cast) {
    case f32_i32_native: {
      storage.set_vector_f32_i32(compressed_storage);
      break;
    }
    case f16_as_f32: {
      storage.set_vector_f16(compressed_storage);
      break;
    }
    case i8_as_i32: {
      storage.set_vector_i8(compressed_storage);
      break;
    }
    case i16_as_i32: {
      storage.set_vector_i16(compressed_storage);
      break;
    }
    case u8_as_i32: {
      storage.set_vector_u8(compressed_storage);
      break;
    }
    case u16_as_i32: {
      storage.set_vector_u16(compressed_storage);
      break;
    }
  }
  
  // pc = program counter
  for (ushort pc = 0; pc < params.num_ops; ++pc) {
    UnaryOperationType operation = operations[pc];
    if (operation <= atan_f32) {
      switch (operation) {
        case abs_f32: {
          GET_SET_F32(precise::abs)
        }
        case abs_i32: {
          GET_SET_I32(abs)
        }
        case acos_f32: {
          GET_SET_F32(precise::acos)
        }
        case acosh_f32: {
          GET_SET_F32(precise::acosh)
        }
        case asin_f32: {
          GET_SET_F32(precise::asin)
        }
        case asinh_f32: {
          GET_SET_F32(precise::asinh)
        }
        case atan_f32: {
          GET_SET_F32(precise::atan)
        }
        case atanh_f32: {
          GET_SET_F32(precise::atanh)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= cast_i32_to_u16) {
      switch (operation) {
        case cast_f32_to_f16: {
          auto x = storage.get_f32();
          auto casted = half4(x);
          SET_F32(float4(casted))
        }
        case cast_f32_to_i32: {
          auto x = storage.get_f32();
          auto operation_metadata = get_metadata(metadata, pc);
          int2 bounds = ((constant int2*)operation_metadata)[0];
          auto casted = int4(x);
          casted = clamp(casted, bounds[0], bounds[1]);
          SET_I32(casted)
        }
        case cast_i32_to_f16: {
          auto x = storage.get_i32();
          auto casted = half4(x);
          SET_F32(float4(casted))
        }
        case cast_i32_to_f32: {
          auto x = storage.get_i32();
          SET_F32(float4(x))
        }
        case cast_i32_to_u8: {
          auto x = storage.get_i32();
          auto casted = uchar4(x);
          SET_I32(int4(casted))
        }
        case cast_i32_to_u16: {
          auto x = storage.get_i32();
          auto casted = ushort4(x);
          SET_I32(int4(casted))
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= floor_f32) {
      switch (operation) {
        case ceil_f32: {
          GET_SET_F32(precise::ceil)
        }
        case cos_f32: {
          GET_SET_F32(precise::cos)
        }
        case cosh_f32: {
          GET_SET_F32(precise::cos)
        }
        case elu_f32: {
          auto x = storage.get_f32();
          x = select(x, precise::expm1(x), x < 0);
          SET_F32(x)
        }
        case exp_f32: {
          GET_SET_F32(precise::exp)
        }
        case expm1_f32: {
          GET_SET_F32(precise::expm1)
        }
        case floor_f32: {
          GET_SET_F32(precise::floor)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= is_nan_f32) {
      switch (operation) {
        case is_finite_f32: {
          auto x = storage.get_f32();
          auto mask = int4(isfinite(x));
          SET_I32(mask)
        }
        case is_inf_f32: {
          auto x = storage.get_f32();
          auto mask = int4(isinf(x));
          SET_I32(mask)
        }
        case is_nan_f32: {
          auto x = storage.get_f32();
          auto mask = int4(isnan(x));
          SET_I32(mask)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= round_f32) {
      switch (operation) {
        case leaky_relu_f32: {
          auto x = storage.get_f32();
          auto operation_metadata = get_metadata(metadata, pc);
          float alpha = ((constant float*)operation_metadata)[0];
          x = precise::max(x, x * alpha);
          SET_F32(x);
        }
        case log_f32: {
          GET_SET_F32(precise::log);
        }
        case log1p_f32: {
          auto x = storage.get_f32();
          SET_F32(precise::log(1 + x));
        }
        case neg_f32: {
          GET_SET_F32(-)
        }
        case neg_i32: {
          GET_SET_I32(-)
        }
        case relu_f32: {
          auto x = storage.get_f32();
          SET_F32(precise::max(0, x))
        }
        case relu6_f32: {
          auto x = storage.get_f32();
          SET_F32(precise::clamp(x, 0, 6))
        }
        case round_f32: {
          GET_SET_F32(precise::rint)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= softplus_f32) {
      switch (operation) {
        case rsqrt_f32: {
          GET_SET_F32(precise::rsqrt)
        }
        case selu_f32: {
          auto x = storage.get_f32();
          constexpr float ALPHA = 1.6732632423543772848170429916717;
          constexpr float SCALE = 1.0507009873554804934193349852946;
          x = select(x, ALPHA * precise::expm1(x), x < 0);
          x = SCALE * x;
          SET_F32(x);
        }
        case sigmoid_f32: {
          auto x = storage.get_f32();
          x = 1 + precise::exp(-x);
          x = precise::divide(1, x);
          SET_F32(x);
        }
        case sign_f32: {
          GET_SET_F32(sign)
        }
        case sign_i32: {
          auto x = storage.get_i32();
          x = select(int4(1), int4(-1), x < 0);
          SET_I32(x)
        }
        case sin_f32: {
          GET_SET_F32(precise::sin)
        }
        case sinh_f32: {
          GET_SET_F32(precise::sinh)
        }
        case softplus_f32: {
          auto x = storage.get_f32();
          x = precise::exp(x) + 1;
          x = precise::log(x);
          SET_F32(x)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= tanh_f32) {
      switch (operation) {
        case softsign_f32: {
          auto x = storage.get_f32();
          auto denominator = precise::abs(x) + 1;
          x = precise::divide(x, denominator);
          SET_F32(x)
        }
        case sqrt_f32: {
          GET_SET_F32(precise::sqrt)
        }
        case square_f32: {
          auto x = storage.get_f32();
          SET_F32(x * x)
        }
        case square_i32: {
          auto x = storage.get_i32();
          SET_I32(x * x)
        }
        case tan_f32: {
          GET_SET_F32(precise::tan)
        }
        case tanh_f32: {
          GET_SET_F32(precise::tanh)
        }
        default:
          return; // This should never happen.
      }
    } else if (operation <= increment_i32) {
      switch (operation) {
        case increment_f32: {
          auto x = storage.get_f32();
          SET_F32(x + 1);
        }
        case increment_i32: {
          auto x = storage.get_i32();
          SET_I32(x + 1);
        }
        default:
          return; // This should never happen.
      }
    } else {
      return; // This should never happen.
    }
  }
  
  switch (params.write_memory_cast) {
    case f32_i32_native: {
      uint4 mem_slice = storage.get_vector_f32_i32();
      ((device uint4*)output)[tid] = mem_slice;
      break;
    }
    case f16_as_f32: {
      ushort4 mem_slice = storage.get_vector_f16();
      ((device ushort4*)output)[tid] = mem_slice;
    }
    case i8_as_i32:
    case u8_as_i32: {
      uchar4 mem_slice = storage.get_vector_i8_u8();
      ((device uchar4*)output)[tid] = mem_slice;
    }
    case i16_as_i32:
    case u16_as_i32: {
      ushort4 mem_slice = storage.get_vector_i16_u16();
      ((device ushort4*)output)[tid] = mem_slice;
    }
  }
}
