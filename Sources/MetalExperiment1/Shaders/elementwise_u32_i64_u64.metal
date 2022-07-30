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

// Prioritizes performance of large integers over small integers. The `f32_i32` shader does the
// opposite, bringing float-point point numbers and small integers to the top of switch statements.
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
  
  scalar_add_i64_u64 = 30, // requires metadata
  scalar_sub_i64_u64 = 31, // requires metadata
  scalar_sub_inverse_i64_u64 = 32, // requires metadata
  scalar_mul_i64 = 33, // requires metadata
  scalar_div_i64 = 34, // requires metadata
  scalar_div_inverse_i64 = 35, // requires metadata
  
  scalar_mul_u64 = 40, // requires metadata
  scalar_div_u64 = 41, // requires metadata
  scalar_div_inverse_u64 = 42, // requires metadata
  
  // Binary (1000 - 1999)
  
  add_i64_u64 = 1000,
  comparison_i64 = 1001, // requires metadata
  comparison_u64 = 1002, // requires metadata
  
  div_i64 = 1010,
  div_u64 = 1011,
  maximum_i64 = 1012,
  maximum_u64 = 1013,
  
  minimum_i64 = 1020,
  minimum_u64 = 1021,
  mod_i64 = 1022,
  mod_u64 = 1023,
  
  mul_i64 = 1030,
  mul_u64 = 1031,
  squared_difference_i64 = 1032,
  squared_difference_u64 = 1033,
  
  sub_i64_u64 = 1040,
  
  // Ternary (2000 - 2999)
  
  clip_by_value_i64 = 2000,
  clip_by_value_u64 = 2001,
  select_i64_u64 = 2002,
  
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

#define GET_SET_I64(expr)          \
SET_I64(expr(register1.get_i64())) \

#define GET_SET_U64(expr)          \
SET_U64(expr(register1.get_u64())) \

#define GET_SET_BINARY_I64(expr)                        \
SET_I64(expr(register1.get_i64(), register2.get_i64())) \

#define GET_SET_BINARY_U64(expr)                        \
SET_U64(expr(register1.get_u64(), register2.get_u64())) \

#define GET_SET_BINARY_INFIX_I64(infix_expr)                \
SET_I64(register1.get_i64() infix_expr register2.get_i64()) \

#define GET_SET_BINARY_INFIX_U64(infix_expr)                \
SET_U64(register1.get_u64() infix_expr register2.get_u64()) \

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
        // TODO: Change to `input4`.
        input = input3;
        break;
      }
    }
    if (read_params.layout & 128) {
      ulong mem_slice_u64 = ((device ulong*)input)[0];
      switch (read_params.layout) {
        case 128 + 8: {
          ulong mem_slice = ulong(mem_slice_u64);
          compressed.set_scalar_u64(mem_slice);
          break;
        }
        case 128 + 4: {
          uint mem_slice = uint(mem_slice_u64);
          compressed.set_scalar_u32(mem_slice);
          break;
        }
        case 128 + 2: {
          ushort mem_slice = ushort(mem_slice_u64);
          compressed.set_scalar_u16(mem_slice);
          break;
        }
        default: /*128 + 1*/ {
          uchar mem_slice = uchar(mem_slice_u64);
          compressed.set_scalar_u8(mem_slice);
          break;
        }
      }
    } else {
      switch (read_params.layout) {
        case 8: {
          ulong2 mem_slice = ((device ulong2*)input)[tid];
          compressed.set_vector_u64(mem_slice);
          break;
        }
        case 4: {
          uint2 mem_slice = ((device uint2*)input)[tid];
          compressed.set_vector_u32(mem_slice);
          break;
        }
        case 2: {
          ushort2 mem_slice = ((device ushort2*)input)[tid];
          compressed.set_vector_u16(mem_slice);
          break;
        }
        default: /*1*/ {
          uchar2 mem_slice = ((device uchar2*)input)[tid];
          compressed.set_vector_u8(mem_slice);
          break;
        }
      }
    }
    
    Register expanded;
    if (read_params.memory_cast <= i8_as_i64) {
      switch (read_params.memory_cast) {
        case i64_u64_native: {
          expanded.set_vector_i64_u64(compressed);
          break;
        }
        case i32_as_i64: {
          expanded.set_vector_i32(compressed);
          break;
        }
        case i16_as_i64: {
          expanded.set_vector_i16(compressed);
          break;
        }
        default: /*i8_as_i64*/ {
          expanded.set_vector_i8(compressed);
          break;
        }
      }
    } else {
      switch (read_params.memory_cast) {
        case u32_as_i64: {
          expanded.set_vector_u32(compressed);
          break;
        }
        case u16_as_i64: {
          expanded.set_vector_u16(compressed);
          break;
        }
        case u8_as_i64: {
          expanded.set_vector_u8(compressed);
          break;
        }
        case f32_padded: {
          expanded.set_vector_f32(compressed);
          break;
        }
        default: /*f16_as_f32_padded*/ {
          expanded.set_vector_f16(compressed);
          break;
        }
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
      if (operation <= square_u64) {
        switch (operation) {
          case abs_i64: {
            GET_SET_I64(abs)
          }
          case neg_i64: {
            GET_SET_I64(-)
          }
          case sign_i64: {
            auto x = register1.get_i64();
            auto mask = select(long2(1), long2(-1), x < 0);
            mask = select(mask, long2(0), x == 0);
            SET_I64(mask)
          }
          case sign_u64: {
            auto x = register1.get_u64();
            x = ulong2(bool2(x));
            SET_U64(x)
          }
          case square_i64: {
            auto x = register1.get_i64();
            SET_I64(x * x)
          }
          default: /*square_u64*/ {
            auto x = register1.get_u64();
            SET_U64(x * x)
          }
        }
      } else if (operation <= cast_i64_u64_to_bool) {
        switch (operation) {
          case cast_f32_to_u32: {
            auto x = register1.get_f32();
            auto casted = long2(uint2(x));
            SET_I64(casted)
          }
          case cast_f32_to_i64: {
            auto x = register1.get_f32();
            auto casted = long2(x);
            SET_I64(casted)
          }
          case cast_i64_to_f16: {
            auto x = register1.get_i64();
            auto casted = float2(half2(x));
            SET_F32(casted)
          }
          case cast_i64_to_f32: {
            auto x = register1.get_i64();
            auto casted = float2(x);
            SET_F32(casted)
          }
          default: /*cast_i64_u64_to_bool*/ {
            auto x = register1.get_i64();
            auto casted = long2(bool2(x));
            SET_I64(casted)
          }
        }
      } else if (operation <= cast_i64_u64_to_u32) {
        switch (operation) {
          case cast_f32_to_u64: {
            auto x = register1.get_f32();
            auto casted = ulong2(x);
            SET_U64(casted)
          }
          case cast_u64_to_f16: {
            auto x = register1.get_u64();
            auto casted = float2(half2(x));
            SET_F32(casted)
          }
          case cast_u64_to_f32: {
            auto x = register1.get_u64();
            auto casted = float2(x);
            SET_F32(casted)
          }
          default: /*cast_i64_u64_to_i32
                     cast_i64_u64_to_u32*/ {
            auto x = register1.get_u64();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            ulong mask = ((constant ulong*)operation_metadata)[0];
            x &= ulong2(mask); // truncate
            
            if (operation == cast_i64_u64_to_i32) { // sign extend
              ulong sign_mask = mask ^ (mask >> 1);
              ulong inverted_mask = ~mask;
              for (int i = 0; i < 2; ++i) {
                // Sign mask has one bit activated.
                if (x[i] & sign_mask) {
                  x[i] |= inverted_mask;
                }
              }
            }
            SET_U64(x)
          }
        }
      } else if (operation <= scalar_div_inverse_i64) {
        auto x = register1.get_i64();
        auto operation_metadata = get_metadata(metadata, metadata_index);
        long scalar = ((constant long*)operation_metadata)[0];
        switch (operation) {
          case scalar_add_i64_u64: {
            x += scalar;
            break;
          }
          case scalar_sub_i64_u64: {
            x -= scalar;
            break;
          }
          case scalar_sub_inverse_i64_u64: {
            x = scalar - x;
            break;
          }
          case scalar_mul_i64: {
            x *= scalar;
            break;
          }
          case scalar_div_i64: {
            x /= scalar;
            break;
          }
          default: /*scalar_div_inverse_i64*/ {
            x = scalar / x;
            break;
          }
        }
        register1.set_i64(x);
      } else /*(operation <= scalar_div_inverse_u64)*/ {
        auto x = register1.get_u64();
        auto operation_metadata = get_metadata(metadata, metadata_index);
        ulong scalar = ((constant long*)operation_metadata)[0];
        switch (operation) {
          case scalar_mul_u64: {
            x *= scalar;
            break;
          }
          case scalar_div_u64: {
            x /= scalar;
            break;
          }
          default: /*scalar_div_inverse_u64*/ {
            x = scalar / x;
            break;
          }
        }
        register1.set_u64(x);
      }
    } else if (operation < 2000) {
      // MARK: - Binary
      if (operation <= comparison_u64) {
        switch (operation) {
          case add_i64_u64: {
            GET_SET_BINARY_INFIX_I64(+)
          }
          case comparison_i64: {
            auto x = register1.get_i64();
            auto y = register2.get_i64();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            ushort2 codes = ((constant ushort2*)operation_metadata)[0];
            
            bool2 out;
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
            SET_U64(ulong2(out))
          }
          default: /*comparison_u64*/ {
            auto x = register1.get_u64();
            auto y = register2.get_u64();
            auto operation_metadata = get_metadata(metadata, metadata_index);
            ushort2 codes = ((constant ushort2*)operation_metadata)[0];
            
            bool2 out;
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
            SET_U64(ulong2(out))
          }
        }
      } else if (operation <= maximum_u64) {
        switch (operation) {
          case div_i64: {
            GET_SET_BINARY_INFIX_I64(/)
          }
          case div_u64: {
            GET_SET_BINARY_INFIX_U64(/)
          }
          case maximum_i64: {
            GET_SET_BINARY_I64(max)
          }
          default: /*maximum_u64*/ {
            GET_SET_BINARY_U64(max)
          }
        }
      } else if (operation <= mod_u64) {
        switch (operation) {
          case minimum_i64: {
            GET_SET_BINARY_I64(min)
          }
          case minimum_u64: {
            GET_SET_BINARY_U64(min)
          }
          case mod_i64: {
            GET_SET_BINARY_INFIX_I64(%)
          }
          default: /*mod_u64*/ {
            GET_SET_BINARY_INFIX_U64(%)
          }
        }
      } else if (operation <= squared_difference_u64) {
        switch (operation) {
          case mul_i64: {
            GET_SET_BINARY_INFIX_I64(*);
          }
          case mul_u64: {
            GET_SET_BINARY_INFIX_U64(*);
          }
          case squared_difference_i64: {
            auto x = register1.get_i64();
            auto y = register2.get_i64();
            auto out = long2(absdiff(x, y));
            out = out * out;
            SET_I64(out)
          }
          default: /*squared_difference_u64*/ {
            auto x = register1.get_u64();
            auto y = register2.get_u64();
            auto out = ulong2(absdiff(x, y));
            out = out * out;
            SET_U64(out)
          }
        }
      } else /*(operation <= sub_i64_u64)*/ {
        switch (operation) {
          default: /*sub_i64_u64*/ {
            GET_SET_BINARY_INFIX_I64(-)
          }
        }
      }
    } else if (operation < 3000) {
      // MARK: - Ternary
      switch (operation) {
        case clip_by_value_i64: {
          auto x = register1.get_i64();
          auto y = register2.get_i64();
          auto z = register3.get_i64();
          auto out = clamp(x, y, z);
          SET_I64(out);
        }
        case clip_by_value_u64: {
          auto x = register1.get_u64();
          auto y = register2.get_u64();
          auto z = register3.get_u64();
          auto out = clamp(x, y, z);
          SET_U64(out);
        }
        default: /*select_i64_u64*/ {
          auto x = register1.get_u64();
          auto y = register2.get_u64();
          auto z = register3.get_vector_i16_u16();
          auto out = select(x, y, bool2(z));
          SET_U64(out);
        }
      }
    } else /*(operation >= 3000)*/ {
      // MARK: - Other
      if (operation <= swap_registers_1_4) {
        auto lhs = register1.get_u64();
        ulong2 rhs;
        switch (operation) {
          case swap_registers_1_2: {
            rhs = register2.get_u64();
            register2.set_u64(lhs);
            break;
          }
          case swap_registers_1_3: {
            rhs = register3.get_u64();
            register3.set_u64(lhs);
            break;
          }
          default: /*swap_registers_1_4*/ {
            rhs = register4.get_u64();
            register4.set_u64(lhs);
            break;
          }
        }
        register1.set_u64(rhs);
      } else {
        switch (operation) {
          case swap_registers_2_3: {
            auto lhs = register2.get_u64();
            auto rhs = register3.get_u64();
            register3.set_u64(lhs);
            register2.set_u64(rhs);
            break;
          }
          case swap_registers_2_4: {
            auto lhs = register2.get_u64();
            auto rhs = register4.get_u64();
            register4.set_u64(lhs);
            register2.set_u64(rhs);
            break;
          }
          default: /*swap_registers_3_4*/ {
            auto lhs = register3.get_u64();
            auto rhs = register4.get_u64();
            register4.set_u64(lhs);
            register3.set_u64(rhs);
            break;
          }
        }
      }
    }
  }
  
  switch (params.write_memory_cast) {
    case i64_u64_native: {
      ulong2 mem_slice = register1.get_vector_i64_u64();
      ((device ulong2*)output)[tid] = mem_slice;
      break;
    }
    case i32_as_i64:
    case u32_as_i64: {
      uint2 mem_slice = register1.get_vector_i32_u32();
      ((device uint2*)output)[tid] = mem_slice;
      break;
    }
    case i16_as_i64:
    case u16_as_i64: {
      ushort2 mem_slice = register1.get_vector_i16_u16();
      ((device ushort2*)output)[tid] = mem_slice;
      break;
    }
    case i8_as_i64:
    case u8_as_i64: {
      uchar2 mem_slice = register1.get_vector_i8_u8();
      ((device uchar2*)output)[tid] = mem_slice;
      break;
    }
    case f32_padded: {
      uint2 mem_slice = register1.get_vector_f32();
      ((device uint2*)output)[tid] = mem_slice;
      break;
    }
    default: /*f16_as_f32_padded*/ {
      ushort2 mem_slice = register1.get_vector_f16();
      ((device ushort2*)output)[tid] = mem_slice;
      break;
    }
  }
}
