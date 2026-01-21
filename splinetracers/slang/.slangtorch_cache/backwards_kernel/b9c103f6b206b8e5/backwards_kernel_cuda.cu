#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

// Boolean vector types should follow CUDA's builtin vector alignment rules
// Align boolX the same as charX according to CUDA spec:
// char1/uchar1: 1-byte aligned, char2/uchar2: 2-byte aligned
// char3/uchar3: 1-byte aligned, char4/uchar4: 4-byte aligned
struct __align__(1) bool1
{
    bool x;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(2) bool2
{
    bool x, y;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(1) bool3
{
    bool x, y, z;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(4) bool4
{
    bool x, y, z, w;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool __ldg(const bool* ptr)
{
    return (bool)(__ldg((const char*)ptr));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 __ldg(const bool2* ptr)
{
    auto val = __ldg((const char2*)ptr);
    return {val.x != 0, val.y != 0};
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 __ldg(const bool4* ptr)
{
    auto val = __ldg((const char4*)ptr);
    return {val.x != 0, val.y != 0, val.z != 0, val.w != 0};
}

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef ptrdiff_t intptr_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef size_t uintptr_t;

typedef long long longlong;
typedef unsigned long long ulonglong;

#else

// When not using NVRTC, match the platform's int64_t definition for signed type
// On Linux: int64_t is 'long', on Windows: int64_t is 'long long'
typedef int64_t longlong;
// ulonglong must remain 'unsigned long long' to match CUDA's atomic operations
typedef unsigned long long ulonglong;

#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

#if SLANG_CUDA_ENABLE_HALF
typedef __half half;
#endif

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(bool)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##1 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##2 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##3 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##4 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(bool)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other)            \
    {                                                                                           \
        bool##n result;                                                                         \
        for (int i = 0; i < n; i++)                                                             \
            *_slang_vector_get_element_ptr(&result, i) =                                        \
                (_slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i)); \
        return result;                                                                          \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(bool)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
#if defined(SLANG_CUDA_ENABLE_HALF) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, __half, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, __half, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(bool)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }

    SLANG_FORCE_INLINE SLANG_CUDA_CALL const Vector<T, COLS>& operator[](size_t index) const
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(bool)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust


// surf1Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF1DWRITE_CONVERT_IMPL(T, c)                                                     \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T>(                              \
        T v,                                                                                     \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        asm volatile(                                                                            \
            "sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};" ::"l"(surfObj),        \
            "r"(x),                                                                              \
            c(v));                                                                               \
    }                                                                                            \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T##2>(                           \
        T##2 v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const T vx = v.x, vy = v.y;                                                              \
        asm volatile(                                                                            \
            "sust.p.1d.v2.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2, %3};" ::"l"(surfObj), \
            "r"(x),                                                                              \
            c(vx),                                                                               \
            c(vy));                                                                              \
    }                                                                                            \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T##4>(                           \
        T##4 v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                                          \
        asm volatile(                                                                            \
            "sust.p.1d.v4.b32." SLANG_PTX_BOUNDARY_MODE                                          \
            " [%0, {%1}], {%2, %3, %4, %5};" ::"l"(surfObj),                                     \
            "r"(x),                                                                              \
            c(vx),                                                                               \
            c(vy),                                                                               \
            c(vz),                                                                               \
            c(vw));                                                                              \
    }

SLANG_SURF1DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF1DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF1DWRITE_CONVERT_IMPL(int, "r")

// surf1DLayeredwrite_convert (not supported)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1DLayeredwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int layer,
    cudaSurfaceBoundaryMode boundaryMode)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA doesn't support formatted surface writes on 1D array surfaces");
}

// surf2Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF2DWRITE_CONVERT_IMPL(T, c)                                                  \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T>(                           \
        T v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        asm volatile(                                                                         \
            "sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1, %2}], {%3};" ::"l"(surfObj), \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(v));                                                                            \
    }                                                                                         \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T##2>(                        \
        T##2 v,                                                                               \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        const T vx = v.x, vy = v.y;                                                           \
        asm volatile(                                                                         \
            "sust.p.2d.v2.b32." SLANG_PTX_BOUNDARY_MODE                                       \
            " [%0, {%1, %2}], {%3, %4};" ::"l"(surfObj),                                      \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(vx),                                                                            \
            c(vy));                                                                           \
    }                                                                                         \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T##4>(                        \
        T##4 v,                                                                               \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                                       \
        asm volatile(                                                                         \
            "sust.p.2d.v4.b32." SLANG_PTX_BOUNDARY_MODE                                       \
            " [%0, {%1, %2}], {%3, %4, %5, %6};" ::"l"(surfObj),                              \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(vx),                                                                            \
            c(vy),                                                                            \
            c(vz),                                                                            \
            c(vw));                                                                           \
    }

SLANG_SURF2DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF2DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF2DWRITE_CONVERT_IMPL(int, "r")

// surf2DLayeredwrite_convert (not supported)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2DLayeredwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int layer,
    cudaSurfaceBoundaryMode boundaryMode)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA doesn't support formatted surface writes on 2D array surfaces");
}

// surf3Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF3DWRITE_CONVERT_IMPL(T, c)                             \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T>(      \
        T v,                                                             \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        asm volatile(                                                    \
            "sust.p.3d.b32." SLANG_PTX_BOUNDARY_MODE                     \
            " [%0, {%1, %2, %3, %4}], {%5};" ::"l"(surfObj),             \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(v));                                                       \
    }                                                                    \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T##2>(   \
        T##2 v,                                                          \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        const T vx = v.x, vy = v.y;                                      \
        asm volatile(                                                    \
            "sust.p.3d.v2.b32." SLANG_PTX_BOUNDARY_MODE                  \
            " [%0, {%1, %2, %3, %4}], {%5, %6};" ::"l"(surfObj),         \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(vx),                                                       \
            c(vy));                                                      \
    }                                                                    \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T##4>(   \
        T##4 v,                                                          \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                  \
        asm volatile(                                                    \
            "sust.p.3d.v4.b32." SLANG_PTX_BOUNDARY_MODE                  \
            " [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};" ::"l"(surfObj), \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(vx),                                                       \
            c(vy),                                                       \
            c(vz),                                                       \
            c(vw));                                                      \
    }

SLANG_SURF3DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF3DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF3DWRITE_CONVERT_IMPL(int, "r")

// ----------------------------- F16 -----------------------------------------
#if SLANG_CUDA_ENABLE_HALF
// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_ceil(__half f)
{
    return ::hceil(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_floor(__half f)
{
    return ::hfloor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_round(__half f)
{
    return ::hrint(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sin(__half f)
{
    return ::hsin(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_cos(__half f)
{
    return ::hcos(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F16_sincos(__half f, __half* s, __half* c)
{
    *s = ::hsin(f);
    *c = ::hcos(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_tan(__half f)
{
    return __float2half(::tanf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_asin(__half f)
{
    return __float2half(::asinf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_acos(__half f)
{
    return __float2half(::acosf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atan(__half f)
{
    return __float2half(::atanf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sinh(__half f)
{
    return __float2half(::sinhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_cosh(__half f)
{
    return __float2half(::coshf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_tanh(__half f)
{
    return __float2half(::tanhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_asinh(__half f)
{
    return __float2half(::asinhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_acosh(__half f)
{
    return __float2half(::acoshf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atanh(__half f)
{
    return __float2half(::atanhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log2(__half f)
{
    return ::hlog2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log(__half f)
{
    return ::hlog(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log10(__half f)
{
    return ::hlog10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_exp2(__half f)
{
    return ::hexp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_exp(__half f)
{
    return ::hexp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_abs(__half f)
{
    return __habs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_trunc(__half f)
{
    return ::htrunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sqrt(__half f)
{
    return ::hsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_rsqrt(__half f)
{
    return ::hrsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F16_sign(__half f)
{
    return (f == __half(0.0f)) ? 0 : ((f < __half(0.0f)) ? -1 : 1);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_frac(__half f)
{
    return f - F16_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isnan(__half f)
{
    return __hisnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isfinite(__half f)
{
    return !__hisinf(f) && !__hisnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isinf(__half f)
{
    return __hisinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_min(__half a, __half b)
{
    return __hmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_max(__half a, __half b)
{
    return __hmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_pow(__half a, __half b)
{
    return __float2half(::powf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_fmod(__half a, __half b)
{
    return __float2half(::fmodf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_remainder(__half a, __half b)
{
    return __float2half(::remainderf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atan2(__half a, __half b)
{
    return __float2half(::atan2(__half2float(a), __half2float(b)));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_frexp(__half x, int* e)
{
    return __float2half(frexpf(__half2float(x), e));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_modf(__half x, __half* ip)
{
    float ipf;
    float res = ::modff(__half2float(x), &ipf);
    *ip = __float2half(ipf);
    return __float2half(res);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint16_t F16_asuint(__half h)
{
    return __half_as_ushort(h);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int16_t F16_asint(__half h)
{
    return __half_as_short(h);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_fma(__half a, __half b, __half c)
{
    return __hfma(a, b, c);
}

#endif

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asinh(float f)
{
    return ::asinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acosh(float f)
{
    return ::acoshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atanh(float f)
{
    return ::atanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F32_sign(float f)
{
    return (f == 0.0f) ? 0 : ((f < 0.0f) ? -1 : 1);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F64_sign(double f)
{
    return (f == 0.0) ? 0 : ((f < 0.0) ? -1 : 1);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- U8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I8_countbits(int8_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U16_countbits(uint16_t v)
{
    // No native 16bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    return __popc(v);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_firstbitlow(uint32_t v)
{
    // __ffs returns 1-based bit position or 0 if no bits set
    // firstbitlow should return 0-based bit position or ~0u if no bits set
    return v == 0 ? ~0u : (__ffs(v) - 1);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_firstbithigh(uint32_t v)
{
    // maps to hlsl firstbithigh
    if ((int32_t)v < 0)
        v = ~v;
    if (v == 0)
        return ~0u;
    return 31 - __clz(v);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_firstbitlow(int32_t v)
{
    return U32_firstbitlow(uint32_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_firstbithigh(int32_t v)
{
    return U32_firstbithigh(uint32_t(v));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    return __popcll(v);
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride) const
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Atomic operations support

// Signed 64-bit atomic wrappers
// CUDA only supports unsigned long long atomics, so we cast signed to unsigned
// Use longlong type with explicit unsigned long long casts for platform portability
__device__ __forceinline__ longlong atomicExch(longlong* address, longlong val)
{
    return (longlong)atomicExch((unsigned long long*)address, (unsigned long long)val);
}

__device__ __forceinline__ longlong atomicCAS(longlong* address, longlong compare, longlong val)
{
    return (longlong)atomicCAS(
        (unsigned long long*)address,
        (unsigned long long)compare,
        (unsigned long long)val);
}

__device__ __forceinline__ longlong atomicAdd(longlong* address, longlong val)
{
    return (longlong)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// Float bitwise atomic compare-and-swap
// Uses integer atomics to preserve exact float bit patterns
__device__ __forceinline__ float atomicCAS(float* address, float compare, float val)
{
    int* addr_as_int = (int*)address;
    int old = atomicCAS(addr_as_int, __float_as_int(compare), __float_as_int(val));
    return __int_as_float(old);
}

// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a, bool exclusive = false);
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a, bool exclusive = false);
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

// Compact specializations using macro for getInitial
#define SLANG_WAVE_MIN_SPEC(T, EXCL_VAL)                                  \
    template<>                                                            \
    __inline__ __device__ T WaveOpMin<T>::getInitial(T a, bool exclusive) \
    {                                                                     \
        return exclusive ? (EXCL_VAL) : a;                                \
    }

#define SLANG_WAVE_MAX_SPEC(T, EXCL_VAL)                                  \
    template<>                                                            \
    __inline__ __device__ T WaveOpMax<T>::getInitial(T a, bool exclusive) \
    {                                                                     \
        return exclusive ? (EXCL_VAL) : a;                                \
    }

// Min specializations (exclusive identity = max value)
SLANG_WAVE_MIN_SPEC(float, SLANG_INFINITY)
SLANG_WAVE_MIN_SPEC(double, SLANG_INFINITY)
SLANG_WAVE_MIN_SPEC(int, 0x7FFFFFFF)
SLANG_WAVE_MIN_SPEC(uint, 0xFFFFFFFF)
SLANG_WAVE_MIN_SPEC(char, (char)0x7F)
SLANG_WAVE_MIN_SPEC(int8_t, (int8_t)0x7F)
SLANG_WAVE_MIN_SPEC(uint8_t, (uint8_t)0xFF)
SLANG_WAVE_MIN_SPEC(int16_t, (int16_t)0x7FFF)
SLANG_WAVE_MIN_SPEC(uint16_t, (uint16_t)0xFFFF)
SLANG_WAVE_MIN_SPEC(int64_t, 0x7FFFFFFFFFFFFFFFLL)
SLANG_WAVE_MIN_SPEC(uint64_t, 0xFFFFFFFFFFFFFFFFULL)
#if SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_MIN_SPEC(__half, __ushort_as_half(0x7BFF))
#endif

// Max specializations (exclusive identity = min value)
SLANG_WAVE_MAX_SPEC(float, -SLANG_INFINITY)
SLANG_WAVE_MAX_SPEC(double, -SLANG_INFINITY)
SLANG_WAVE_MAX_SPEC(int, (int)0x80000000)
SLANG_WAVE_MAX_SPEC(uint, 0)
SLANG_WAVE_MAX_SPEC(char, (char)0x80)
SLANG_WAVE_MAX_SPEC(int8_t, (int8_t)0x80)
SLANG_WAVE_MAX_SPEC(uint8_t, 0)
SLANG_WAVE_MAX_SPEC(int16_t, (int16_t)0x8000)
SLANG_WAVE_MAX_SPEC(uint16_t, 0)
SLANG_WAVE_MAX_SPEC(int64_t, (int64_t)0x8000000000000000LL)
SLANG_WAVE_MAX_SPEC(uint64_t, 0)
#if SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_MAX_SPEC(__half, __ushort_as_half(0xFBFF))
#endif

#undef SLANG_WAVE_MIN_SPEC
#undef SLANG_WAVE_MAX_SPEC

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<char>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<uchar>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<short>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<ushort>
{
    typedef ushort Type;
};
#if SLANG_CUDA_ENABLE_HALF
template<>
struct ElementTypeTrait<__half>
{
    typedef __half Type;
};
#endif

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Additional vector types
template<>
struct ElementTypeTrait<char2>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<char3>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<char4>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<uchar2>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<uchar3>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<uchar4>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<short2>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<short3>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<short4>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<ushort2>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<ushort3>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<ushort4>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<longlong2>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<longlong3>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<longlong4>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong2>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong3>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong4>
{
    typedef uint64_t Type;
};
#if SLANG_CUDA_ENABLE_HALF
template<>
struct ElementTypeTrait<__half2>
{
    typedef __half Type;
};
template<>
struct ElementTypeTrait<__half3>
{
    typedef __half Type;
};
template<>
struct ElementTypeTrait<__half4>
{
    typedef __half Type;
};
#endif

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixMin(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixMax(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMax<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpMin<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpMax<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

// Wrapper structures for exclusive operations that use the overloaded getInitial method
template<typename T>
struct WaveOpExclusiveMin
{
    __inline__ __device__ static T getInitial(T a) { return WaveOpMin<T>::getInitial(a, true); }
    __inline__ __device__ static T doOp(T a, T b) { return WaveOpMin<T>::doOp(a, b); }
};

template<typename T>
struct WaveOpExclusiveMax
{
    __inline__ __device__ static T getInitial(T a) { return WaveOpMax<T>::getInitial(a, true); }
    __inline__ __device__ static T doOp(T a, T b) { return WaveOpMax<T>::doOp(a, b); }
};

// Inclusive prefix min/max functions (for WaveMultiPrefixInclusive*)
template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMin(WarpMask mask, T val)
{
    return _wavePrefixMin(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMax(WarpMask mask, T val)
{
    return _wavePrefixMax(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMinMultiple(WarpMask mask, T val)
{
    return _wavePrefixMinMultiple(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMaxMultiple(WarpMask mask, T val)
{
    return _wavePrefixMaxMultiple(mask, val);
}

// Explicit exclusive prefix min/max functions (for WaveMultiPrefixExclusive*)
template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMin(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpExclusiveMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMax(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpExclusiveMax<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpExclusiveMin<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpExclusiveMax<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX

struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* optixTrace(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ float4 optixGetSpherePositionAndRadius()
{
    float4 data[1];
    optixGetSphereData(data);
    return data[0];
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ float4
optixHitObjectGetSpherePositionAndRadius(OptixTraversableHandle* Obj)
{
    float4 data[1];
    optixHitObjectGetSphereData(data);
    return data[0];
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ Matrix<float, 2, 4> optixGetLssPositionsAndRadii()
{
    float4 data[2];
    optixGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ Matrix<float, 2, 4> optixHitObjectGetLssPositionsAndRadii(
    OptixTraversableHandle* Obj)
{
    float4 data[2];
    optixHitObjectGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixIsSphereHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_SPHERE;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixHitObjectIsSphereHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_SPHERE;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixIsLSSHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixHitObjectIsLSSHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}
#endif

template<typename T>
__forceinline__ __device__ void* optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTraverse(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

template<typename T>
__forceinline__ __device__ void* optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    float RayTime,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTraverse(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        RayTime,
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsHit(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsHit();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsMiss(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsMiss();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsNop(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsNop();
}
#endif

#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ uint
slangOptixHitObjectGetClusterId(OptixTraversableHandle* hitObj)
{
    return optixHitObjectGetClusterId();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    OptixTraversableHandle* missObj)
{
    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f /* rayTime */
#if (OPTIX_VERSION >= 90000)
        ,
        OPTIX_RAY_FLAG_NONE /* rayFlags*/
#endif
    );
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    float CurrentTime,
    OptixTraversableHandle* missObj)
{
    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        CurrentTime
#if (OPTIX_VERSION >= 90000)
        ,
        OPTIX_RAY_FLAG_NONE /* rayFlags*/
#endif
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call native optixMakeHitObject directly
    optixMakeHitObject(
        AccelerationStructure,                            // handle
        Ray.Origin,                                       // rayOrigin
        Ray.Direction,                                    // rayDirection
        Ray.TMin,                                         // tmin
        Ray.TMax,                                         // tmax
        0.f,                                              // rayTime
        RayContributionToHitGroupIndex,                   // sbtOffset
        MultiplierForGeometryContributionToHitGroupIndex, // sbtStride
        InstanceIndex,                                    // instIdx
        nullptr,                                          // transforms
        0,                                                // numTransforms
        GeometryIndex,                                    // sbtGASIdx
        PrimitiveIndex,                                   // primIdx
        HitKind                                           // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call optixMakeHitObjectWithRecord directly
    optixMakeHitObjectWithRecord(
        AccelerationStructure, // handle
        Ray.Origin,            // rayOrigin
        Ray.Direction,         // rayDirection
        Ray.TMin,              // tmin
        Ray.TMax,              // tmax
        0.f,                   // rayTime
        HitGroupRecordIndex,   // sbtRecordIndex
        InstanceIndex,         // instIdx
        nullptr,               // transforms
        0,                     // numTransforms
        GeometryIndex,         // sbtGASIdx
        PrimitiveIndex,        // primIdx
        HitKind                // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call native optixMakeHitObject directly
    optixMakeHitObject(
        AccelerationStructure,                            // handle
        Ray.Origin,                                       // rayOrigin
        Ray.Direction,                                    // rayDirection
        Ray.TMin,                                         // tmin
        Ray.TMax,                                         // tmax
        CurrentTime,                                      // rayTime
        RayContributionToHitGroupIndex,                   // sbtOffset
        MultiplierForGeometryContributionToHitGroupIndex, // sbtStride
        InstanceIndex,                                    // instIdx
        nullptr,                                          // transforms
        0,                                                // numTransforms
        GeometryIndex,                                    // sbtGASIdx
        PrimitiveIndex,                                   // primIdx
        HitKind                                           // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call optixMakeHitObjectWithRecord directly
    optixMakeHitObjectWithRecord(
        AccelerationStructure, // handle
        Ray.Origin,            // rayOrigin
        Ray.Direction,         // rayDirection
        Ray.TMin,              // tmin
        Ray.TMax,              // tmax
        CurrentTime,           // rayTime
        HitGroupRecordIndex,   // sbtRecordIndex
        InstanceIndex,         // instIdx
        nullptr,               // transforms
        0,                     // numTransforms
        GeometryIndex,         // sbtGASIdx
        PrimitiveIndex,        // primIdx
        HitKind                // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void slangOptixMakeNopHitObject(OptixTraversableHandle* Obj)
{
    optixMakeNopHitObject();
}
#endif

#if (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixInvoke(
    OptixTraversableHandle AccelerationStructure,
    OptixTraversableHandle* HitOrMiss,
    T Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixInvoke(r0, r1);
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ RayDesc optixHitObjectGetRayDesc(OptixTraversableHandle* obj)
{
    RayDesc ray = {
        optixHitObjectGetWorldRayOrigin(),
        optixHitObjectGetRayTmin(),
        optixHitObjectGetWorldRayDirection(),
        optixHitObjectGetRayTmax()};
    return ray;
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetInstanceIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint slangOptixHitObjectGetInstanceId(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceId();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetSbtGASIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtGASIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetPrimitiveIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetPrimitiveIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ T optixHitObjectGetAttribute(OptixTraversableHandle* Obj)
{
    constexpr size_t numInts = (sizeof(T) + sizeof(uint32_t) - 1) /
                               sizeof(uint32_t); // Number of 32-bit values, rounded up
    static_assert(numInts <= 8, "Attribute type is too large");

    // Create an array to hold the attribute values
    uint32_t values[numInts == 0 ? 1 : numInts] = {0}; // Ensure we have at least one element

    // Read the appropriate number of attribute registers
    if constexpr (numInts > 0)
        values[0] = optixHitObjectGetAttribute_0();
    if constexpr (numInts > 1)
        values[1] = optixHitObjectGetAttribute_1();
    if constexpr (numInts > 2)
        values[2] = optixHitObjectGetAttribute_2();
    if constexpr (numInts > 3)
        values[3] = optixHitObjectGetAttribute_3();
    if constexpr (numInts > 4)
        values[4] = optixHitObjectGetAttribute_4();
    if constexpr (numInts > 5)
        values[5] = optixHitObjectGetAttribute_5();
    if constexpr (numInts > 6)
        values[6] = optixHitObjectGetAttribute_6();
    if constexpr (numInts > 7)
        values[7] = optixHitObjectGetAttribute_7();

    // Reinterpret the array as the desired type
    T result;
    memcpy(&result, values, sizeof(T));
    return result;
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetSbtRecordIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtRecordIndex();
}
#endif

#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ uint
slangOptixHitObjectSetSbtRecordIndex(OptixTraversableHandle* Obj, uint sbtRecordIndex)
{
    optixHitObjectSetSbtRecordIndex(sbtRecordIndex); // returns void
    return sbtRecordIndex;
}
#endif

// OptiX multi-level traversal wrappers
// These wrappers convert OptiX's float[12] matrix pointer returns to Slang's Matrix type
__device__ __forceinline__ Matrix<float, 3, 4> _slang_optixGetInstanceTransformFromHandle(
    ulonglong handle)
{
    const float4* m = optixGetInstanceTransformFromHandle(handle);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(m[0], m[1], m[2]);
}

__device__ __forceinline__ Matrix<float, 3, 4> _slang_optixGetInstanceInverseTransformFromHandle(
    ulonglong handle)
{
    const float4* m = optixGetInstanceInverseTransformFromHandle(handle);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(m[0], m[1], m[2]);
}

// OptiX transformation matrix wrappers
// These wrappers convert OptiX's float[12] matrix format to Slang's Matrix type
__device__ __forceinline__ Matrix<float, 3, 4> slangOptixGetObjectToWorldTransformMatrix()
{
    float m[12];
    optixGetObjectToWorldTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(
        make_float4(m[0], m[1], m[2], m[3]),
        make_float4(m[4], m[5], m[6], m[7]),
        make_float4(m[8], m[9], m[10], m[11]));
}

__device__ __forceinline__ Matrix<float, 3, 4> slangOptixGetWorldToObjectTransformMatrix()
{
    float m[12];
    optixGetWorldToObjectTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(
        make_float4(m[0], m[1], m[2], m[3]),
        make_float4(m[4], m[5], m[6], m[7]),
        make_float4(m[8], m[9], m[10], m[11]));
}

__device__ __forceinline__ Matrix<float, 4, 3> slangOptixGetObjectToWorldTransformMatrix4x3()
{
    float m[12];
    optixGetObjectToWorldTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}

__device__ __forceinline__ Matrix<float, 4, 3> slangOptixGetWorldToObjectTransformMatrix4x3()
{
    float m[12];
    optixGetWorldToObjectTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}

#else
// Define OptixTraversableHandle even if OptiX is not enabled.
// This allows RaytracingAccelerationStructure to be properly reflected in non-OptiX code.
typedef unsigned long long OptixTraversableHandle;
#endif
static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};

// Implementations for texture fetch/load functions using tex PTX intrinsics
// These are used for read-only texture access with integer coordinates.

// 1D is not supported via PTX. Keeping the implementation below in case it ever gets supported.
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1Dfetch_int(CUtexObject texObj, int x, int mip)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA does not support fetching from 1D textures");
}

#if 0
#define SLANG_TEX1DFETCH_INT_IMPL(T, dtype, c)                                                 \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1Dfetch_int(CUtexObject texObj, int x, int mip)    \
    {                                                                                          \
        T result;                                                                              \
        T stub;                                                                                \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result), c(stub), c(stub), c(stub)                                             \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return result;                                                                         \
    }                                                                                          \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 tex1Dfetch_int(CUtexObject texObj, int x, int mip) \
    {                                                                                          \
        T result_x, result_y;                                                                  \
        T stub;                                                                                \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result_x), c(result_y), c(stub), c(stub)                                       \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return make_##T##2(result_x, result_y);                                                \
    }                                                                                          \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 tex1Dfetch_int(CUtexObject texObj, int x, int mip) \
    {                                                                                          \
        T result_x, result_y, result_z, result_w;                                              \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                               \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return make_##T##4(result_x, result_y, result_z, result_w);                            \
    }

SLANG_TEX1DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX1DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX1DFETCH_INT_IMPL(int, "s32", "=r")
#endif

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip);

#define SLANG_TEX2DFETCH_INT_IMPL(T, dtype, c)                                                     \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip) \
    {                                                                                              \
        T result;                                                                                  \
        T stub;                                                                                    \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result), c(stub), c(stub), c(stub)                                                 \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return result;                                                                             \
    }                                                                                              \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                             \
        T##2 tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip)                             \
    {                                                                                              \
        T result_x, result_y;                                                                      \
        T stub;                                                                                    \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result_x), c(result_y), c(stub), c(stub)                                           \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return make_##T##2(result_x, result_y);                                                    \
    }                                                                                              \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                             \
        T##4 tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip)                             \
    {                                                                                              \
        T result_x, result_y, result_z, result_w;                                                  \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                                   \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return make_##T##4(result_x, result_y, result_z, result_w);                                \
    }

SLANG_TEX2DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX2DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX2DFETCH_INT_IMPL(int, "s32", "=r")


template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip);

#define SLANG_TEX3DFETCH_INT_IMPL(T, dtype, c)                                            \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                                  \
    tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)                      \
    {                                                                                     \
        T result;                                                                         \
        T stub;                                                                           \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result), c(stub), c(stub), c(stub)                                        \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return result;                                                                    \
    }                                                                                     \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                    \
        T##2 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)             \
    {                                                                                     \
        T result_x, result_y;                                                             \
        T stub;                                                                           \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result_x), c(result_y), c(stub), c(stub)                                  \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return make_##T##2(result_x, result_y);                                           \
    }                                                                                     \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                    \
        T##4 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)             \
    {                                                                                     \
        T result_x, result_y, result_z, result_w;                                         \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result_x), c(result_y), c(result_z), c(result_w)                          \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return make_##T##4(result_x, result_y, result_z, result_w);                       \
    }

SLANG_TEX3DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX3DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX3DFETCH_INT_IMPL(int, "s32", "=r")

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip);

#define SLANG_TEX1DARRAYFETCH_INT_IMPL(T, dtype, c)                                \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                           \
    tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)             \
    {                                                                              \
        T result;                                                                  \
        T stub;                                                                    \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result), c(stub), c(stub), c(stub)                                 \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return result;                                                             \
    }                                                                              \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                             \
        T##2 tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)    \
    {                                                                              \
        T result_x, result_y;                                                      \
        T stub;                                                                    \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result_x), c(result_y), c(stub), c(stub)                           \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return make_##T##2(result_x, result_y);                                    \
    }                                                                              \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                             \
        T##4 tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)    \
    {                                                                              \
        T result_x, result_y, result_z, result_w;                                  \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result_x), c(result_y), c(result_z), c(result_w)                   \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return make_##T##4(result_x, result_y, result_z, result_w);                \
    }

SLANG_TEX1DARRAYFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX1DARRAYFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX1DARRAYFETCH_INT_IMPL(int, "s32", "=r")

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip);

#define SLANG_TEX2DARRAYFETCH_INT_IMPL(T, dtype, c)                                         \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                                    \
    tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)               \
    {                                                                                       \
        T result;                                                                           \
        T stub;                                                                             \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result), c(stub), c(stub), c(stub)                                          \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return result;                                                                      \
    }                                                                                       \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                      \
        T##2 tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)      \
    {                                                                                       \
        T result_x, result_y;                                                               \
        T stub;                                                                             \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result_x), c(result_y), c(stub), c(stub)                                    \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return make_##T##2(result_x, result_y);                                             \
    }                                                                                       \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                      \
        T##4 tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)      \
    {                                                                                       \
        T result_x, result_y, result_z, result_w;                                           \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                            \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return make_##T##4(result_x, result_y, result_z, result_w);                         \
    }

SLANG_TEX2DARRAYFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX2DARRAYFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX2DARRAYFETCH_INT_IMPL(int, "s32", "=r")

// Wave rotate helper functions - templated approach
#define SLANG_WARP_FULL_MASK 0xFFFFFFFF

// Macro-based wave rotate implementation following codebase patterns
#define SLANG_WAVE_ROTATE_IMPL(T)                                                     \
    __device__ __forceinline__ T##2 _slang_waveRotate(T##2 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##2(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }                                                                                 \
    __device__ __forceinline__ T##3 _slang_waveRotate(T##3 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##3(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.z,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }                                                                                 \
    __device__ __forceinline__ T##4 _slang_waveRotate(T##4 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##4(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.z,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.w,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }

// Generate wave rotate functions for all standard vector types
SLANG_WAVE_ROTATE_IMPL(uint)
SLANG_WAVE_ROTATE_IMPL(int)
SLANG_WAVE_ROTATE_IMPL(float)
SLANG_WAVE_ROTATE_IMPL(short)
SLANG_WAVE_ROTATE_IMPL(ushort)
SLANG_WAVE_ROTATE_IMPL(char)
SLANG_WAVE_ROTATE_IMPL(uchar)
SLANG_WAVE_ROTATE_IMPL(longlong)
SLANG_WAVE_ROTATE_IMPL(ulonglong)

#ifdef SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_ROTATE_IMPL(__half)
#endif

// Special handling for boolean vectors (requires int conversion)
__device__ __forceinline__ bool2 _slang_waveRotate(bool2 value, unsigned int delta)
{
    int2 intValue = make_int2((int)value.x, (int)value.y);
    int2 result = _slang_waveRotate(intValue, delta);
    return make_bool2((bool)result.x, (bool)result.y);
}

__device__ __forceinline__ bool3 _slang_waveRotate(bool3 value, unsigned int delta)
{
    int3 intValue = make_int3((int)value.x, (int)value.y, (int)value.z);
    int3 result = _slang_waveRotate(intValue, delta);
    return make_bool3((bool)result.x, (bool)result.y, (bool)result.z);
}

__device__ __forceinline__ bool4 _slang_waveRotate(bool4 value, unsigned int delta)
{
    int4 intValue = make_int4((int)value.x, (int)value.y, (int)value.z, (int)value.w);
    int4 result = _slang_waveRotate(intValue, delta);
    return make_bool4((bool)result.x, (bool)result.y, (bool)result.z, (bool)result.w);
}

#undef SLANG_WAVE_ROTATE_IMPL

// Quad control operations for CUDA
__device__ __forceinline__ bool _slang_quadAny(bool expr)
{
    // Get values from all 4 lanes in the quad
    bool v0 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 0);
    bool v1 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 1);
    bool v2 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 2);
    bool v3 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 3);
    return v0 || v1 || v2 || v3;
}

__device__ __forceinline__ bool _slang_quadAll(bool expr)
{
    // Get values from all 4 lanes in the quad
    bool v0 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 0);
    bool v1 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 1);
    bool v2 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 2);
    bool v3 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 3);
    return v0 && v1 && v2 && v3;
}

// Clustered wave rotate operations for CUDA
// Clustered rotate rotates values within clusters of specified size
#define SLANG_WAVE_CLUSTERED_ROTATE_IMPL(T)                                                       \
    __device__ __forceinline__ T                                                                  \
    _slang_waveClusteredRotate(T value, unsigned int delta, unsigned int clusterSize)             \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return __shfl_sync(SLANG_WARP_FULL_MASK, value, targetLane);                              \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##2 _slang_waveClusteredRotate(T##2 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##2(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane));                           \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##3 _slang_waveClusteredRotate(T##3 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##3(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.z, targetLane));                           \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##4 _slang_waveClusteredRotate(T##4 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##4(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.z, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.w, targetLane));                           \
    }

// Generate clustered wave rotate functions for all standard types
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(uint)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(int)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(float)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(short)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(ushort)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(char)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(uchar)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(longlong)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(ulonglong)

#ifdef SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(__half)
#endif

// Special handling for boolean clustered rotate
__device__ __forceinline__ bool _slang_waveClusteredRotate(
    bool value,
    unsigned int delta,
    unsigned int clusterSize)
{
    int intValue = (int)value;
    int result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return (bool)result;
}

__device__ __forceinline__ bool2
_slang_waveClusteredRotate(bool2 value, unsigned int delta, unsigned int clusterSize)
{
    int2 intValue = make_int2((int)value.x, (int)value.y);
    int2 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool2((bool)result.x, (bool)result.y);
}

__device__ __forceinline__ bool3
_slang_waveClusteredRotate(bool3 value, unsigned int delta, unsigned int clusterSize)
{
    int3 intValue = make_int3((int)value.x, (int)value.y, (int)value.z);
    int3 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool3((bool)result.x, (bool)result.y, (bool)result.z);
}

__device__ __forceinline__ bool4
_slang_waveClusteredRotate(bool4 value, unsigned int delta, unsigned int clusterSize)
{
    int4 intValue = make_int4((int)value.x, (int)value.y, (int)value.z, (int)value.w);
    int4 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool4((bool)result.x, (bool)result.y, (bool)result.z, (bool)result.w);
}

#undef SLANG_WAVE_CLUSTERED_ROTATE_IMPL

// ---------------------- OptiX Cooperative Vector Wrappers --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX

#if (OPTIX_VERSION >= 90000)

// Constexpr function to map Slang component type enum to OptiX cooperative vector element type
__host__ __device__ constexpr OptixCoopVecElemType slangToOptixComponentType(unsigned slangEnum)
{
    switch (slangEnum)
    {
    case 0:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3; // FloatE4M3
    case 1:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2; // FloatE5M2
    case 2:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16; // Float16
    case 3:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; // Float32
    case 5:
        return OPTIX_COOP_VEC_ELEM_TYPE_INT8; // SignedInt8
    case 7:
        return OPTIX_COOP_VEC_ELEM_TYPE_INT32; // SignedInt32
    case 10:
        return OPTIX_COOP_VEC_ELEM_TYPE_UINT8; // UnsignedInt8
    case 12:
        return OPTIX_COOP_VEC_ELEM_TYPE_UINT32; // UnsignedInt32
    default:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; // Default
    }
}

// Constexpr function to map Slang matrix layout enum to OptiX cooperative vector matrix layout
__host__ __device__ constexpr OptixCoopVecMatrixLayout slangToOptixMatrixLayout(unsigned slangEnum)
{
    switch (slangEnum)
    {
    case 0:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR; // RowMajor
    case 1:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR; // ColumnMajor
    case 2:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL; // InferencingOptimal
    case 3:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL; // TrainingOptimal
    default:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR; // Default
    }
}

// Wrapper structs to maintain compatibility with existing template-based interface
template<unsigned SlangEnum>
struct SlangToOptixComponentType
{
    static constexpr OptixCoopVecElemType value = slangToOptixComponentType(SlangEnum);
};

template<unsigned SlangEnum>
struct SlangToOptixMatrixLayout
{
    static constexpr OptixCoopVecMatrixLayout value = slangToOptixMatrixLayout(SlangEnum);
};

// Template trait to extract vector size from OptixCoopVec<T, N>
// Conditional compilation for NVRTC compatibility
template<typename T>
struct OptixCoopVecTraits;

// Template specialization for OptiX's OptixCoopVec - only enabled when cooperative vectors are
// available NVRTC explicitly disables cooperative vectors by setting
// OPTIX_INCLUDE_COOPERATIVE_VECTOR to 0
#if defined(OPTIX_VERSION) && OPTIX_VERSION > 90000
template<typename T, unsigned int N>
struct OptixCoopVecTraits<OptixCoopVec<T, N>>
{
    static constexpr unsigned int size = N;
};
#endif

template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    bool transpose,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        matrixStride);
}

// OptiX cooperative vector matrix multiplication wrapper (WITH bias - 6 runtime params)
template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout,
    unsigned biasInterpretation>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    CUdeviceptr bias,
    unsigned biasOffset,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    // Call OptiX SDK with bias (6 runtime parameters)
    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value,
        SlangToOptixComponentType<biasInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        bias,
        biasOffset,
        matrixStride);
}

// OptiX cooperative vector matrix multiplication wrapper (WITHOUT bias, 4 runtime params -
// StructuredBuffer variant)
template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    // Call OptiX SDK without bias and without transpose (4 runtime parameters)
    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        matrixStride);
}

#endif // (OPTIX_VERSION >= 90000)

#endif // SLANG_CUDA_ENABLE_OPTIX


// This implementation can only be enabled on CUDA Toolkit 12.5+
#if (((__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 5)) || (CUDA_VERSION >= 12050))
// The reason we have to implement our own wmma operation on CUDA is the interface
// design of cooperative_matrix on Vulkan is quite different from CUDA WMMA API, where
// SPIRV spec doesn't require the matrix layout during declaration of the cooperative_matrix,
// instead it is only required during load/store operations. However, in CUDA WMMA API, the layout
// has to be specified during the declaration of the fragment itself. Slang's interface desgin
// is more similar to SPIRV's cooperative_matrix. So to bridge this gap, we have to implement our
// wmma operation by using PTX wmma instructions directly, because PTX wmma instructions is quite
// similar to SPIRV's cooperative_matrix spec.
namespace Slang_CUDA_WMMA
{

// Enums for template specialization
enum MatrixUse : int
{
    MatrixA = 0,
    MatrixB = 1,
    MatrixC = 2,
    MatrixD = 3,
};

enum Layout : int
{
    RowMajor = 0,
    ColMajor = 1
};

enum ShapeCombination : int
{
    m16n16k16 = 0,
    m8n32k16 = 1,
    m32n8k16 = 2
};

// ====================================================================================
// PTX Name Helpers
// ====================================================================================

// Shape names
template<int M, int N, int K>
struct PtxShapeName;
template<>
struct PtxShapeName<16, 16, 16>
{
    static constexpr const char name[] = "m16n16k16";
};
template<>
struct PtxShapeName<8, 32, 16>
{
    static constexpr const char name[] = "m8n32k16";
};
template<>
struct PtxShapeName<32, 8, 16>
{
    static constexpr const char name[] = "m32n8k16";
};

// Matrix role names
template<MatrixUse use>
struct PtxMatrixRoleName;
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixA>
{
    static constexpr const char name[] = "a";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixB>
{
    static constexpr const char name[] = "b";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixC>
{
    static constexpr const char name[] = "c";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixD>
{
    static constexpr const char name[] = "d";
};

// Layout names
template<Layout layout>
struct PtxLayoutName;
template<>
struct PtxLayoutName<Layout::RowMajor>
{
    static constexpr const char name[] = "row";
};
template<>
struct PtxLayoutName<Layout::ColMajor>
{
    static constexpr const char name[] = "col";
};

// Type names
template<typename T>
struct PtxTypeName;

#if SLANG_CUDA_ENABLE_HALF
template<>
struct PtxTypeName<half>
{
    static constexpr const char name[] = "f16";
};
#endif // #if SLANG_CUDA_ENABLE_HALF

template<>
struct PtxTypeName<float>
{
    static constexpr const char name[] = "f32";
};
template<>
struct PtxTypeName<char>
{
    static constexpr const char name[] = "s8";
};
template<>
struct PtxTypeName<unsigned char>
{
    static constexpr const char name[] = "u8";
};
template<>
struct PtxTypeName<int32_t>
{
    static constexpr const char name[] = "s32";
};

// ====================================================================================
// Register Counts for different matrices
// ====================================================================================
template<typename ElemT, int M, int N, int K, MatrixUse use>
struct RegisterCount;

#if SLANG_CUDA_ENABLE_HALF
// Half (f16) - 8 regs for A/B, 4 regs for C/D
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixA>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixB>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 4;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 4;
};
#endif // #if SLANG_CUDA_ENABLE_HALF

// Float (f32) - 8 regs for C/D only
template<int M, int N, int K>
struct RegisterCount<float, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<float, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 8;
};

// Int32 (s32) - 8 regs for C/D (accumulator for int8 operations)
template<int M, int N, int K>
struct RegisterCount<int32_t, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<int32_t, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 8;
};

// Uint8 (u8) - varies by shape
template<>
struct RegisterCount<unsigned char, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<unsigned char, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<unsigned char, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<unsigned char, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<unsigned char, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<unsigned char, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};

// Int8 (s8) - same as u8
template<>
struct RegisterCount<char, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<char, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<char, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<char, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<char, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<char, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};


// ====================================================================================
// Saturation at the output for integer MMA
// ====================================================================================
template<bool saturatingAccumulation>
struct IsSaturated;

template<>
struct IsSaturated<true>
{
    static constexpr const char name[] = ".satfinite";
};

template<>
struct IsSaturated<false>
{
    static constexpr const char name[] = "";
};

// ====================================================================================
// WMMA Load - Inline PTX
// ====================================================================================

template<typename ElemT, int M, int N, int K, MatrixUse use, Layout layout>
__device__ inline void wmmaLoad(uint32_t* regs, const ElemT* ptr, int stride)
{
    constexpr int nregs = RegisterCount<ElemT, M, N, K, use>::value;

    switch (nregs)
    {
    case 1:
        asm volatile("wmma.load.%1.sync.aligned.%2.%3.%4 {%0}, [%5], %6;\n"
                     : "=r"(regs[0])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 2:
        asm volatile("wmma.load.%2.sync.aligned.%3.%4.%5 {%0, %1}, [%6], %7;\n"
                     : "=r"(regs[0]), "=r"(regs[1])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 4:
        asm volatile("wmma.load.%4.sync.aligned.%5.%6.%7 {%0, %1, %2, %3}, [%8], %9;\n"
                     : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 8:
        asm volatile("wmma.load.%8.sync.aligned.%9.%10.%11 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%12], %13;\n"
                     : "=r"(regs[0]),
                       "=r"(regs[1]),
                       "=r"(regs[2]),
                       "=r"(regs[3]),
                       "=r"(regs[4]),
                       "=r"(regs[5]),
                       "=r"(regs[6]),
                       "=r"(regs[7])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;
    }
}

// ====================================================================================
// WMMA Store - Inline PTX
// ====================================================================================

template<typename ElemT, int M, int N, int K, Layout layout>
__device__ inline void wmmaStore(ElemT* ptr, const uint32_t* regs, int stride)
{
    constexpr int nregs = RegisterCount<ElemT, M, N, K, MatrixUse::MatrixD>::value;

    switch (nregs)
    {
    case 4:
        asm volatile("wmma.store.d.sync.aligned.%0.%1.%2 [%3], {%4, %5, %6, %7}, %8;\n"
                     :
                     : "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(regs[0]),
                       "r"(regs[1]),
                       "r"(regs[2]),
                       "r"(regs[3]),
                       "r"(stride));
        break;

    case 8:
        asm volatile("wmma.store.d.sync.aligned.%0.%1.%2 "
                     "[%3], {%4, %5, %6, %7, %8, %9, %10, %11}, %12;\n"
                     :
                     : "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(regs[0]),
                       "r"(regs[1]),
                       "r"(regs[2]),
                       "r"(regs[3]),
                       "r"(regs[4]),
                       "r"(regs[5]),
                       "r"(regs[6]),
                       "r"(regs[7]),
                       "r"(stride));
        break;
    }
}

// Helper to get M, N, K from ShapeCombination
template<ShapeCombination shape>
struct ShapeToMNK;
template<>
struct ShapeToMNK<ShapeCombination::m16n16k16>
{
    static constexpr int M = 16, N = 16, K = 16;
};
template<>
struct ShapeToMNK<ShapeCombination::m8n32k16>
{
    static constexpr int M = 8, N = 32, K = 16;
};
template<>
struct ShapeToMNK<ShapeCombination::m32n8k16>
{
    static constexpr int M = 32, N = 8, K = 16;
};

template<typename T>
inline unsigned __device__ Pack32Helper(T value);

#if SLANG_CUDA_ENABLE_HALF
template<>
inline unsigned __device__ Pack32Helper<half>(half value)
{
    return __half_as_ushort(value) | (__half_as_ushort(value) << 16);
};
#endif

template<>
inline unsigned __device__ Pack32Helper<float>(float value)
{
    return __float_as_uint(value);
};

template<>
inline unsigned __device__ Pack32Helper<int>(int value)
{
    return (unsigned)value;
};
template<>
inline unsigned __device__ Pack32Helper<char>(char value)
{
    return value << 24 | value << 16 | value << 8 | value;
};
template<>
inline unsigned __device__ Pack32Helper<unsigned char>(unsigned char value)
{
    return value << 24 | value << 16 | value << 8 | value;
};


// The dimensions of the fragment are specified by M, N, K which are totally determined during
// compile time, so slang already did the pre-filter on the shape & type combination.
template<typename T, int M, int N, int K, MatrixUse R, Layout MatrixLayout = RowMajor>
struct WmmaFragment
{
    typedef WmmaFragment<T, M, N, K, R> This;
    template<Layout layout>
    void __device__ Store(RWStructuredBuffer<T> buffer, uint element, uint stride)
    {
        Store<layout>(buffer.data, element, stride);
    }

    template<Layout layout>
    static This __device__ Load(StructuredBuffer<T> buffer, uint element, uint stride)
    {
        return Load<layout>(buffer.data, element, stride);
    }

    // There is no fill intrinsic in PTX wmma, so it's just 'move' value
    // to the fragment registers.
    void __device__ fill(T value)
    {
        unsigned packed = Pack32Helper(value);

        // Manually assign to prevent register coalescing
        regs[0] = packed;
        regs[1] = packed;
        regs[2] = packed;
        regs[3] = packed;
        regs[4] = packed;
        regs[5] = packed;
        regs[6] = packed;
        regs[7] = packed;
    }

    __device__ This operator*(T b)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        // This loop will be unrolled by the compiler becuase nregs is constexpr
        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) * b);
        }
        return result;
    }

    __device__ This operator*(const This& b)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        // This loop will be unrolled by the compiler becuase nregs is constexpr
        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) * b.get(i));
        }
        return result;
    }

    __device__ This operator/(const This& other)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) / other.get(i));
        }
        return result;
    }

    __device__ This operator-(const This& other)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) - other.get(i));
        }
        return result;
    }

    __device__ This operator-()
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        for (int i = 0; i < nregs; i++)
        {
            result.set(i, -get(i));
        }
        return result;
    }

    __device__ This operator+(const This& other)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) + other.get(i));
        }
        return result;
    }

    __device__ This operator%(const This& other)
    {
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        This result;

        for (int i = 0; i < nregs; i++)
        {
            result.set(i, get(i) % other.get(i));
        }
        return result;
    }

    template<typename U>
    __device__ void copyFrom(const WmmaFragment<U, M, N, K, R>& other)
    {
        // If the data type is different, we need to copy element by element.
        // Since the shape of two matrices are the same, they have the same
        // number of elements.
        for (int i = 0; i < elements_per_thread; i++)
        {
            set(i, static_cast<T>(other.get(i)));
        }
    }

    // Get element by index (handles bit-level access for packed types)
    // For example: u8/s8 matrices have 4 elements per register (32-bit)
    //   - index 0: bits [0:7]   of regs[0]
    //   - index 1: bits [8:15]  of regs[0]
    //   - index 2: bits [16:23] of regs[0]
    //   - index 3: bits [24:31] of regs[0]
    //   - index 4: bits [0:7]   of regs[1], etc.
    __device__ T get(int index) const
    {
        if constexpr (sizeof(T) == 4)
        {
            // T is 32-bit (float or int32): 1 element per register
            return *reinterpret_cast<const T*>(&regs[index]);
        }
        else if constexpr (sizeof(T) == 2)
        {
            // T is 16-bit (half): 2 elements per register
            // Elements per register: [0:15] and [16:31]
            int regIndex = index / 2;
            int elementOffset = index % 2;
            int bitOffset = elementOffset * 16;
            uint32_t extracted = (regs[regIndex] >> bitOffset) & 0xFFFF;
            uint16_t value16 = static_cast<uint16_t>(extracted);
            return *reinterpret_cast<const T*>(&value16);
        }
        else if constexpr (sizeof(T) == 1)
        {
            // T is 8-bit (int8_t, uint8_t): 4 elements per register
            // Elements per register: [0:7], [8:15], [16:23], [24:31]
            int regIndex = index / 4;
            int elementOffset = index % 4;
            int bitOffset = elementOffset * 8;
            uint32_t extracted = (regs[regIndex] >> bitOffset) & 0xFF;
            uint8_t value8 = static_cast<uint8_t>(extracted);
            return *reinterpret_cast<const T*>(&value8);
        }
    }

    // Set element by index (handles bit-level access for packed types)
    __device__ void set(int index, T value)
    {
        if constexpr (sizeof(T) == 4)
        {
            // T is 32-bit (float or int32): 1 element per register
            regs[index] = *reinterpret_cast<const uint32_t*>(&value);
        }
        else if constexpr (sizeof(T) == 2)
        {
            // T is 16-bit (half): 2 elements per register
            int regIndex = index / 2;
            int elementOffset = index % 2;
            int bitOffset = elementOffset * 16;
            uint32_t mask = 0xFFFF;
            uint16_t value16 = *reinterpret_cast<const uint16_t*>(&value);

            // Clear the bits at the target position
            regs[regIndex] &= ~(mask << bitOffset);

            // Set the new value
            regs[regIndex] |= (static_cast<uint32_t>(value16) << bitOffset);
        }
        else if constexpr (sizeof(T) == 1)
        {
            // T is 8-bit (int8_t, uint8_t): 4 elements per register
            int regIndex = index / 4;
            int elementOffset = index % 4;
            int bitOffset = elementOffset * 8;
            uint32_t mask = 0xFF;
            uint8_t value8 = *reinterpret_cast<const uint8_t*>(&value);

            // Clear the bits at the target position
            regs[regIndex] &= ~(mask << bitOffset);

            // Set the new value
            regs[regIndex] |= (static_cast<uint32_t>(value8) << bitOffset);
        }
    }

    template<Layout layout>
    void __device__ Store(T* buffer, uint element, uint stride)
    {
        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaStore<T, M, N, K, layout>(buffer + element, regs, stride);
    }

    template<Layout layout>
    static This __device__ Load(T* buffer, uint element, uint stride)
    {
        WmmaFragment<T, M, N, K, R, layout> fragment;

        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaLoad<T, M, N, K, R, layout>(fragment.regs, buffer + element, stride);

        return fragment;
    }

    static __device__ uint32_t GetLength() { return This::elements_per_thread; }

    // For referencing those template parameters outside the struct
    using ElementType = T;
    static constexpr int m_M = M;
    static constexpr int m_N = N;
    static constexpr int m_K = K;
    static constexpr Layout m_layout = MatrixLayout;

    // Maximum registers needed across all fragment types and data types
    static constexpr int MAX_REGS = 8;
    unsigned regs[MAX_REGS] = {};

    static constexpr uint32_t elements_per_warp = (R == MatrixUse::MatrixA)   ? (M * K)
                                                  : (R == MatrixUse::MatrixB) ? (K * N)
                                                                              : (M * N);

    static_assert(elements_per_warp % 32 == 0, "Total elements per warp must be divisible by 32");

    static constexpr uint32_t elements_per_thread = elements_per_warp / 32;
    static constexpr uint32_t bytes_per_thread = elements_per_thread * sizeof(T);
    static constexpr uint32_t registers_per_thread = (bytes_per_thread + 3) / 4;
};

// ====================================================================================
// FP16 MMA Helper - For half x half inputs
// Specialized on CType and DType (accumulator types)
//
// PTX Syntax: wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype d, a, b, c;
//   where:
//     dtype = type of d (output accumulator): {.f16, .f32}
//     ctype = type of c (input accumulator):  {.f16, .f32}
//
// Note: Types of a and b are implicitly f16 (not specified in PTX instruction).
//       Shape (M, N, K) is passed as template parameters, so one template handles all shapes.
//       We only need to specialize on CType and DType.
// ====================================================================================

template<typename CType, typename DType, int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper;

#if SLANG_CUDA_ENABLE_HALF
// Specialization: c=half, d=half (f16.f16)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<half, half, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<half, M, N, K, MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%4.%5.%6.%7.%8 "
                     "{%0, %1, %2, %3}, "
                     "{%9, %10, %11, %12, %13, %14, %15, %16}, "
                     "{%17, %18, %19, %20, %21, %22, %23, %24}, "
                     "{%25, %26, %27, %28};\n"
                     : "=r"(d.regs[0]), "=r"(d.regs[1]), "=r"(d.regs[2]), "=r"(d.regs[3])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<half>::name),
                       "C"(PtxTypeName<half>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]));
    }
};

// Specialization: c=float, d=half (f16.f32)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<float, half, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%4.%5.%6.%7.%8 "
                     "{%0, %1, %2, %3}, "
                     "{%9, %10, %11, %12, %13, %14, %15, %16}, "
                     "{%17, %18, %19, %20, %21, %22, %23, %24}, "
                     "{%25, %26, %27, %28, %29, %30, %31, %32};\n"
                     : "=r"(d.regs[0]), "=r"(d.regs[1]), "=r"(d.regs[2]), "=r"(d.regs[3])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<half>::name),
                       "C"(PtxTypeName<float>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: c=half, d=float (f32.f16)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<half, float, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.%11.%12 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%13, %14, %15, %16, %17, %18, %19, %20}, "
                     "{%21, %22, %23, %24, %25, %26, %27, %28}, "
                     "{%29, %30, %31, %32};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<float>::name),
                       "C"(PtxTypeName<half>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]));
    }
};

// Specialization: c=float, d=float (f32.f32)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<float, float, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.%11.%12 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%13, %14, %15, %16, %17, %18, %19, %20}, "
                     "{%21, %22, %23, %24, %25, %26, %27, %28}, "
                     "{%29, %30, %31, %32, %33, %34, %35, %36};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<float>::name),
                       "C"(PtxTypeName<float>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};
#endif // #if SLANG_CUDA_ENABLE_HALF

// ====================================================================================
// Integer MMA Helper - For int8/uint8 inputs
// Specialized on shape (register counts depend on shape)
//
// PTX Syntax: wmma.mma.sync.aligned.alayout.blayout.shape.s32.atype.btype.s32{.satfinite} d, a, b,
// c;
//   where:
//     atype = type of a (input matrix A): {.s8, .u8}
//     btype = type of b (input matrix B): {.s8, .u8}
//     C and D are always s32 (int32)
//
// Note: Unlike FP16, integer operations explicitly specify atype and btype in the instruction.
//       We must specialize on shape because register counts vary:
//         m16n16k16: a=2 regs, b=2 regs
//         m8n32k16:  a=1 reg,  b=4 regs
//         m32n8k16:  a=4 regs, b=1 reg
//       C and D always use 8 registers (int32).
// ====================================================================================

template<
    typename AType,
    typename BType,
    ShapeCombination shape,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper;

// Specialization: m16n16k16 (a=2 regs, b=2 regs)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m16n16k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 16, 16, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 16, 16, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 16, 16, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 16, 16, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14, %15}, "
                     "{%16, %17}, "
                     "{%18, %19, %20, %21, %22, %23, %24, %25};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<16, 16, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: m8n32k16 (a=1 reg, b=4 regs)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m8n32k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 8, 32, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 8, 32, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 8, 32, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 8, 32, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14}, "
                     "{%15, %16, %17, %18}, "
                     "{%19, %20, %21, %22, %23, %24, %25, %26};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<8, 32, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: m32n8k16 (a=4 regs, b=1 reg)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m32n8k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 32, 8, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 32, 8, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 32, 8, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 32, 8, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14, %15, %16, %17}, "
                     "{%18}, "
                     "{%19, %20, %21, %22, %23, %24, %25, %26};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<32, 8, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(b.regs[0]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};


// ====================================================================================
// MMA Helper - Primary Template (dispatcher)
// ====================================================================================

template<
    typename AType,
    typename BType,
    typename CType,
    typename DType,
    ShapeCombination shape,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct MMAHelper
{
    static constexpr int M = ShapeToMNK<shape>::M;
    static constexpr int N = ShapeToMNK<shape>::N;
    static constexpr int K = ShapeToMNK<shape>::K;

    __device__ static void eval(
        WmmaFragment<DType, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<CType, M, N, K, MatrixUse::MatrixC>& c,
        bool saturate = false)
    {
        // Dispatch to appropriate helper based on input types
        if constexpr (sizeof(AType) == 2 && sizeof(BType) == 2)
        {
            // FP16 inputs: dispatch to Fp16MMAHelper
            Fp16MMAHelper<CType, DType, M, N, K, LayoutA, LayoutB>::eval(d, a, b, c);
        }
        else
        {
            // Integer inputs (int8/uint8): dispatch to IntegerMMAHelper
            IntegerMMAHelper<AType, BType, shape, LayoutA, LayoutB, saturatingAccumulation>::eval(
                d,
                a,
                b,
                c);
        }
    }
};

//
template<
    typename AType,
    typename BType,
    typename CType,
    typename DType,
    int M,
    int N,
    int K,
    Layout layoutA,
    Layout layoutB,
    bool saturatingAccumulation>
WmmaFragment<DType, M, N, K, MatrixC> __device__ coopMatMulAdd(
    WmmaFragment<AType, M, N, K, MatrixUse::MatrixA, layoutA> matA,
    WmmaFragment<BType, M, N, K, MatrixUse::MatrixB, layoutB> matB,
    WmmaFragment<CType, M, N, K, MatrixUse::MatrixC> matC)
{
    constexpr ShapeCombination shape = (M == 16 && N == 16 && K == 16) ? ShapeCombination::m16n16k16
                                       : (M == 8 && N == 32 && K == 16)
                                           ? ShapeCombination::m8n32k16
                                           : ShapeCombination::m32n8k16;

    WmmaFragment<DType, M, N, K, MatrixC> matD;
    MMAHelper<AType, BType, CType, DType, shape, layoutA, layoutB, saturatingAccumulation>::eval(
        matD,
        matA,
        matB,
        matC);

    return matD;
}

} // namespace Slang_CUDA_WMMA
#endif // #if (((__CUDACC_VER_MAJOR__ >=12)&&(__CUDACC_VER_MINOR__>=5)) || (CUDA_VERSION >= 12050))


#line 44 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
struct Features_0
{
    float3  f0_0;
    float3  f1_0;
    float3  f2_0;
    float3  f3_0;
    float3  f4_0;
    float3  f5_0;
    float3  f6_0;
    float3  f7_0;
    float3  f8_0;
    float3  f9_0;
    float3  f10_0;
    float3  f11_0;
    float3  f12_0;
    float3  f13_0;
    float3  f14_0;
    float3  f15_0;
};


#line 1 "token paste"
__device__ Features_0 Features_x24_syn_dzero_0()
{

#line 1
    Features_0 result_0;

#line 2239 "core.meta.slang"
    float3  _S1 = make_float3 (0.0f);

#line 2239
    (&result_0)->f0_0 = _S1;

#line 2239
    (&result_0)->f1_0 = _S1;

#line 2239
    (&result_0)->f2_0 = _S1;

#line 2239
    (&result_0)->f3_0 = _S1;

#line 2239
    (&result_0)->f4_0 = _S1;

#line 2239
    (&result_0)->f5_0 = _S1;

#line 2239
    (&result_0)->f6_0 = _S1;

#line 2239
    (&result_0)->f7_0 = _S1;

#line 2239
    (&result_0)->f8_0 = _S1;

#line 2239
    (&result_0)->f9_0 = _S1;

#line 2239
    (&result_0)->f10_0 = _S1;

#line 2239
    (&result_0)->f11_0 = _S1;

#line 2239
    (&result_0)->f12_0 = _S1;

#line 2239
    (&result_0)->f13_0 = _S1;

#line 2239
    (&result_0)->f14_0 = _S1;

#line 2239
    (&result_0)->f15_0 = _S1;

#line 2239
    return result_0;
}


#line 2239
__device__ Features_0 Features_x24_syn_dadd_0(Features_0 * SLANG_anonymous_0_0, Features_0 * SLANG_anonymous_1_0)
{

#line 2239
    Features_0 result_1;

#line 2239
    (&result_1)->f0_0 = SLANG_anonymous_0_0->f0_0 + SLANG_anonymous_1_0->f0_0;

#line 2239
    (&result_1)->f1_0 = SLANG_anonymous_0_0->f1_0 + SLANG_anonymous_1_0->f1_0;

#line 2239
    (&result_1)->f2_0 = SLANG_anonymous_0_0->f2_0 + SLANG_anonymous_1_0->f2_0;

#line 2239
    (&result_1)->f3_0 = SLANG_anonymous_0_0->f3_0 + SLANG_anonymous_1_0->f3_0;

#line 2239
    (&result_1)->f4_0 = SLANG_anonymous_0_0->f4_0 + SLANG_anonymous_1_0->f4_0;

#line 2239
    (&result_1)->f5_0 = SLANG_anonymous_0_0->f5_0 + SLANG_anonymous_1_0->f5_0;

#line 2239
    (&result_1)->f6_0 = SLANG_anonymous_0_0->f6_0 + SLANG_anonymous_1_0->f6_0;

#line 2239
    (&result_1)->f7_0 = SLANG_anonymous_0_0->f7_0 + SLANG_anonymous_1_0->f7_0;

#line 2239
    (&result_1)->f8_0 = SLANG_anonymous_0_0->f8_0 + SLANG_anonymous_1_0->f8_0;

#line 2239
    (&result_1)->f9_0 = SLANG_anonymous_0_0->f9_0 + SLANG_anonymous_1_0->f9_0;

#line 2239
    (&result_1)->f10_0 = SLANG_anonymous_0_0->f10_0 + SLANG_anonymous_1_0->f10_0;

#line 2239
    (&result_1)->f11_0 = SLANG_anonymous_0_0->f11_0 + SLANG_anonymous_1_0->f11_0;

#line 2239
    (&result_1)->f12_0 = SLANG_anonymous_0_0->f12_0 + SLANG_anonymous_1_0->f12_0;

#line 2239
    (&result_1)->f13_0 = SLANG_anonymous_0_0->f13_0 + SLANG_anonymous_1_0->f13_0;

#line 2239
    (&result_1)->f14_0 = SLANG_anonymous_0_0->f14_0 + SLANG_anonymous_1_0->f14_0;

#line 2239
    (&result_1)->f15_0 = SLANG_anonymous_0_0->f15_0 + SLANG_anonymous_1_0->f15_0;

#line 2239
    return result_1;
}


#line 47 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
struct ControlPoint_0
{
    float t_0;
    float4  dirac_0;
};


#line 1641 "diff.meta.slang"
__device__ ControlPoint_0 ControlPoint_x24_syn_dzero_0()
{

#line 1641
    ControlPoint_0 result_2;

#line 1641
    (&result_2)->t_0 = 0.0f;

#line 1641
    (&result_2)->dirac_0 = make_float4 (0.0f);

#line 1641
    return result_2;
}


#line 20 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
struct SplineState_0
{
    float2  distortion_parts_0;
    float2  cum_sum_0;
    float3  padding_0;
    float t_1;
    float4  drgb_0;
    float logT_0;
    float3  C_0;
};


#line 20
__device__ SplineState_0 SplineState_x24_syn_dzero_0()
{

#line 20
    SplineState_0 result_3;

#line 2239 "core.meta.slang"
    float2  _S2 = make_float2 (0.0f);

#line 2239
    (&result_3)->distortion_parts_0 = _S2;

#line 2239
    (&result_3)->cum_sum_0 = _S2;

#line 2239
    float3  _S3 = make_float3 (0.0f);

#line 2239
    (&result_3)->padding_0 = _S3;

#line 2239
    (&result_3)->t_1 = 0.0f;

#line 2239
    (&result_3)->drgb_0 = make_float4 (0.0f);

#line 2239
    (&result_3)->logT_0 = 0.0f;

#line 2239
    (&result_3)->C_0 = _S3;

#line 2239
    return result_3;
}


#line 2239
__device__ SplineState_0 SplineState_x24_syn_dadd_0(SplineState_0 * SLANG_anonymous_0_1, SplineState_0 * SLANG_anonymous_1_1)
{

#line 2239
    SplineState_0 result_4;

#line 2239
    (&result_4)->distortion_parts_0 = SLANG_anonymous_0_1->distortion_parts_0 + SLANG_anonymous_1_1->distortion_parts_0;

#line 2239
    (&result_4)->cum_sum_0 = SLANG_anonymous_0_1->cum_sum_0 + SLANG_anonymous_1_1->cum_sum_0;

#line 2239
    (&result_4)->padding_0 = SLANG_anonymous_0_1->padding_0 + SLANG_anonymous_1_1->padding_0;

#line 2239
    (&result_4)->t_1 = SLANG_anonymous_0_1->t_1 + SLANG_anonymous_1_1->t_1;

#line 2239
    (&result_4)->drgb_0 = SLANG_anonymous_0_1->drgb_0 + SLANG_anonymous_1_1->drgb_0;

#line 2239
    (&result_4)->logT_0 = SLANG_anonymous_0_1->logT_0 + SLANG_anonymous_1_1->logT_0;

#line 2239
    (&result_4)->C_0 = SLANG_anonymous_0_1->C_0 + SLANG_anonymous_1_1->C_0;

#line 2239
    return result_4;
}


#line 20 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
__device__ SplineState_0 SplineState_x24init_0(float2  distortion_parts_1, float2  cum_sum_1, float3  padding_1, float t_2, float4  drgb_1, float logT_1, float3  C_1)
{

#line 20
    SplineState_0 _S4;

    (&_S4)->distortion_parts_0 = distortion_parts_1;
    (&_S4)->cum_sum_0 = cum_sum_1;
    (&_S4)->padding_0 = padding_1;

    (&_S4)->t_1 = t_2;
    (&_S4)->drgb_0 = drgb_1;


    (&_S4)->logT_0 = logT_1;
    (&_S4)->C_0 = C_1;

#line 20
    return _S4;
}


#line 101 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ SplineState_0 get_state_0(TensorView view_0, uint ind_0)
{
    float _S5 = ((view_0).load<float>((ind_0), (0U)));

#line 103
    float _S6 = ((view_0).load<float>((ind_0), (1U)));

#line 103
    float2  _S7 = make_float2 (_S5, _S6);
    float _S8 = ((view_0).load<float>((ind_0), (2U)));

#line 104
    float _S9 = ((view_0).load<float>((ind_0), (3U)));

#line 104
    float2  _S10 = make_float2 (_S8, _S9);
    float _S11 = ((view_0).load<float>((ind_0), (4U)));

#line 105
    float _S12 = ((view_0).load<float>((ind_0), (5U)));

#line 105
    float _S13 = ((view_0).load<float>((ind_0), (6U)));

#line 105
    float3  _S14 = make_float3 (_S11, _S12, _S13);

#line 102
    float _S15 = ((view_0).load<float>((ind_0), (7U)));

#line 107
    float _S16 = ((view_0).load<float>((ind_0), (8U)));

#line 107
    float _S17 = ((view_0).load<float>((ind_0), (9U)));

#line 107
    float _S18 = ((view_0).load<float>((ind_0), (10U)));

#line 107
    float _S19 = ((view_0).load<float>((ind_0), (11U)));

#line 107
    float4  _S20 = make_float4 (_S16, _S17, _S18, _S19);

#line 102
    float _S21 = ((view_0).load<float>((ind_0), (12U)));

#line 110
    float _S22 = ((view_0).load<float>((ind_0), (13U)));

#line 110
    float _S23 = ((view_0).load<float>((ind_0), (14U)));

#line 110
    float _S24 = ((view_0).load<float>((ind_0), (15U)));

#line 102
    return SplineState_x24init_0(_S7, _S10, _S14, _S15, _S20, _S21, make_float3 (_S22, _S23, _S24));
}


#line 22
__device__ float3  get_float3_0(TensorView view_1, uint ind_1)
{

#line 23
    float _S25 = ((view_1).load<float>((ind_1), (0U)));

#line 23
    float _S26 = ((view_1).load<float>((ind_1), (1U)));

#line 23
    float _S27 = ((view_1).load<float>((ind_1), (2U)));

#line 23
    return make_float3 (_S25, _S26, _S27);
}


#line 38
__device__ float4  get_float4_0(TensorView view_2, uint ind_2)
{

#line 39
    float _S28 = ((view_2).load<float>((ind_2), (0U)));

#line 39
    float _S29 = ((view_2).load<float>((ind_2), (1U)));

#line 39
    float _S30 = ((view_2).load<float>((ind_2), (2U)));

#line 39
    float _S31 = ((view_2).load<float>((ind_2), (3U)));

#line 39
    return make_float4 (_S28, _S29, _S30, _S31);
}


#line 11986 "hlsl.meta.slang"
struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};


#line 2137 "diff.meta.slang"
__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_0)
{
    DiffPair_float_0 _S32 = *dpx_0;

#line 2139
    float _S33;

#line 2139
    if(((*dpx_0).primal_0) > ((*dpy_0).primal_0))
    {

#line 2139
        _S33 = dOut_0;

#line 2139
    }
    else
    {

#line 2139
        if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
        {

#line 2139
            _S33 = 0.0f;

#line 2139
        }
        else
        {

#line 2139
            _S33 = 0.5f * dOut_0;

#line 2139
        }

#line 2139
    }

#line 2139
    dpx_0->primal_0 = _S32.primal_0;

#line 2139
    dpx_0->differential_0 = _S33;
    DiffPair_float_0 _S34 = *dpy_0;

#line 2140
    if(((*dpy_0).primal_0) > (_S32.primal_0))
    {

#line 2140
        _S33 = dOut_0;

#line 2140
    }
    else
    {

#line 2140
        if(((*dpy_0).primal_0) < ((*dpx_0).primal_0))
        {

#line 2140
            _S33 = 0.0f;

#line 2140
        }
        else
        {

#line 2140
            _S33 = 0.5f * dOut_0;

#line 2140
        }

#line 2140
    }

#line 2140
    dpy_0->primal_0 = _S34.primal_0;

#line 2140
    dpy_0->differential_0 = _S33;
    return;
}


#line 1 "token paste"
__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_1, float dOut_1)
{

#line 1915 "diff.meta.slang"
    float _S35 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_1).primal_0)))))) * dOut_1;

#line 1915
    dpx_1->primal_0 = (*dpx_1).primal_0;

#line 1915
    dpx_1->differential_0 = _S35;



    return;
}


#line 187 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ Matrix<float, 4, 4>  inverse_0(Matrix<float, 4, 4>  m_0)
{

#line 193
    float _S36 = m_0.rows[int(2)].y * m_0.rows[int(3)].z;

#line 193
    float _S37 = m_0.rows[int(3)].y * m_0.rows[int(2)].z;

#line 193
    float _S38 = m_0.rows[int(3)].y * m_0.rows[int(1)].z;

#line 193
    float _S39 = m_0.rows[int(1)].y * m_0.rows[int(3)].z;

#line 193
    float _S40 = m_0.rows[int(2)].y * m_0.rows[int(1)].z;

#line 193
    float _S41 = m_0.rows[int(1)].y * m_0.rows[int(2)].z;

#line 193
    float t11_0 = _S36 * m_0.rows[int(1)].w - _S37 * m_0.rows[int(1)].w + _S38 * m_0.rows[int(2)].w - _S39 * m_0.rows[int(2)].w - _S40 * m_0.rows[int(3)].w + _S41 * m_0.rows[int(3)].w;
    float _S42 = m_0.rows[int(3)].x * m_0.rows[int(2)].z;

#line 194
    float _S43 = m_0.rows[int(2)].x * m_0.rows[int(3)].z;

#line 194
    float _S44 = m_0.rows[int(3)].x * m_0.rows[int(1)].z;

#line 194
    float _S45 = m_0.rows[int(1)].x * m_0.rows[int(3)].z;

#line 194
    float _S46 = m_0.rows[int(2)].x * m_0.rows[int(1)].z;

#line 194
    float _S47 = m_0.rows[int(1)].x * m_0.rows[int(2)].z;

#line 194
    float t12_0 = _S42 * m_0.rows[int(1)].w - _S43 * m_0.rows[int(1)].w - _S44 * m_0.rows[int(2)].w + _S45 * m_0.rows[int(2)].w + _S46 * m_0.rows[int(3)].w - _S47 * m_0.rows[int(3)].w;
    float _S48 = m_0.rows[int(2)].x * m_0.rows[int(3)].y;

#line 195
    float _S49 = m_0.rows[int(3)].x * m_0.rows[int(2)].y;

#line 195
    float _S50 = m_0.rows[int(3)].x * m_0.rows[int(1)].y;

#line 195
    float _S51 = m_0.rows[int(1)].x * m_0.rows[int(3)].y;

#line 195
    float _S52 = m_0.rows[int(2)].x * m_0.rows[int(1)].y;

#line 195
    float _S53 = m_0.rows[int(1)].x * m_0.rows[int(2)].y;

#line 195
    float t13_0 = _S48 * m_0.rows[int(1)].w - _S49 * m_0.rows[int(1)].w + _S50 * m_0.rows[int(2)].w - _S51 * m_0.rows[int(2)].w - _S52 * m_0.rows[int(3)].w + _S53 * m_0.rows[int(3)].w;
    float t14_0 = _S49 * m_0.rows[int(1)].z - _S48 * m_0.rows[int(1)].z - _S50 * m_0.rows[int(2)].z + _S51 * m_0.rows[int(2)].z + _S52 * m_0.rows[int(3)].z - _S53 * m_0.rows[int(3)].z;


    float idet_0 = 1.0f / (m_0.rows[int(0)].x * t11_0 + m_0.rows[int(0)].y * t12_0 + m_0.rows[int(0)].z * t13_0 + m_0.rows[int(0)].w * t14_0);

    Matrix<float, 4, 4>  ret_0;

    *&(((&ret_0)->rows + (int(0)))->x) = t11_0 * idet_0;
    float _S54 = m_0.rows[int(3)].y * m_0.rows[int(0)].z;

#line 204
    float _S55 = m_0.rows[int(0)].y * m_0.rows[int(3)].z;

#line 204
    float _S56 = m_0.rows[int(2)].y * m_0.rows[int(0)].z;

#line 204
    float _S57 = m_0.rows[int(0)].y * m_0.rows[int(2)].z;

#line 204
    *&(((&ret_0)->rows + (int(0)))->y) = (_S37 * m_0.rows[int(0)].w - _S36 * m_0.rows[int(0)].w - _S54 * m_0.rows[int(2)].w + _S55 * m_0.rows[int(2)].w + _S56 * m_0.rows[int(3)].w - _S57 * m_0.rows[int(3)].w) * idet_0;
    float _S58 = m_0.rows[int(1)].y * m_0.rows[int(0)].z;

#line 205
    float _S59 = m_0.rows[int(0)].y * m_0.rows[int(1)].z;

#line 205
    *&(((&ret_0)->rows + (int(0)))->z) = (_S39 * m_0.rows[int(0)].w - _S38 * m_0.rows[int(0)].w + _S54 * m_0.rows[int(1)].w - _S55 * m_0.rows[int(1)].w - _S58 * m_0.rows[int(3)].w + _S59 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(0)))->w) = (_S40 * m_0.rows[int(0)].w - _S41 * m_0.rows[int(0)].w - _S56 * m_0.rows[int(1)].w + _S57 * m_0.rows[int(1)].w + _S58 * m_0.rows[int(2)].w - _S59 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(1)))->x) = t12_0 * idet_0;
    float _S60 = m_0.rows[int(3)].x * m_0.rows[int(0)].z;

#line 209
    float _S61 = m_0.rows[int(0)].x * m_0.rows[int(3)].z;

#line 209
    float _S62 = m_0.rows[int(2)].x * m_0.rows[int(0)].z;

#line 209
    float _S63 = m_0.rows[int(0)].x * m_0.rows[int(2)].z;

#line 209
    *&(((&ret_0)->rows + (int(1)))->y) = (_S43 * m_0.rows[int(0)].w - _S42 * m_0.rows[int(0)].w + _S60 * m_0.rows[int(2)].w - _S61 * m_0.rows[int(2)].w - _S62 * m_0.rows[int(3)].w + _S63 * m_0.rows[int(3)].w) * idet_0;
    float _S64 = m_0.rows[int(1)].x * m_0.rows[int(0)].z;

#line 210
    float _S65 = m_0.rows[int(0)].x * m_0.rows[int(1)].z;

#line 210
    *&(((&ret_0)->rows + (int(1)))->z) = (_S44 * m_0.rows[int(0)].w - _S45 * m_0.rows[int(0)].w - _S60 * m_0.rows[int(1)].w + _S61 * m_0.rows[int(1)].w + _S64 * m_0.rows[int(3)].w - _S65 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(1)))->w) = (_S47 * m_0.rows[int(0)].w - _S46 * m_0.rows[int(0)].w + _S62 * m_0.rows[int(1)].w - _S63 * m_0.rows[int(1)].w - _S64 * m_0.rows[int(2)].w + _S65 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(2)))->x) = t13_0 * idet_0;
    float _S66 = m_0.rows[int(3)].x * m_0.rows[int(0)].y;

#line 214
    float _S67 = m_0.rows[int(0)].x * m_0.rows[int(3)].y;

#line 214
    float _S68 = m_0.rows[int(2)].x * m_0.rows[int(0)].y;

#line 214
    float _S69 = m_0.rows[int(0)].x * m_0.rows[int(2)].y;

#line 214
    *&(((&ret_0)->rows + (int(2)))->y) = (_S49 * m_0.rows[int(0)].w - _S48 * m_0.rows[int(0)].w - _S66 * m_0.rows[int(2)].w + _S67 * m_0.rows[int(2)].w + _S68 * m_0.rows[int(3)].w - _S69 * m_0.rows[int(3)].w) * idet_0;
    float _S70 = m_0.rows[int(1)].x * m_0.rows[int(0)].y;

#line 215
    float _S71 = m_0.rows[int(0)].x * m_0.rows[int(1)].y;

#line 215
    *&(((&ret_0)->rows + (int(2)))->z) = (_S51 * m_0.rows[int(0)].w - _S50 * m_0.rows[int(0)].w + _S66 * m_0.rows[int(1)].w - _S67 * m_0.rows[int(1)].w - _S70 * m_0.rows[int(3)].w + _S71 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(2)))->w) = (_S52 * m_0.rows[int(0)].w - _S53 * m_0.rows[int(0)].w - _S68 * m_0.rows[int(1)].w + _S69 * m_0.rows[int(1)].w + _S70 * m_0.rows[int(2)].w - _S71 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(3)))->x) = t14_0 * idet_0;
    *&(((&ret_0)->rows + (int(3)))->y) = (_S48 * m_0.rows[int(0)].z - _S49 * m_0.rows[int(0)].z + _S66 * m_0.rows[int(2)].z - _S67 * m_0.rows[int(2)].z - _S68 * m_0.rows[int(3)].z + _S69 * m_0.rows[int(3)].z) * idet_0;
    *&(((&ret_0)->rows + (int(3)))->z) = (_S50 * m_0.rows[int(0)].z - _S51 * m_0.rows[int(0)].z - _S66 * m_0.rows[int(1)].z + _S67 * m_0.rows[int(1)].z + _S70 * m_0.rows[int(3)].z - _S71 * m_0.rows[int(3)].z) * idet_0;
    *&(((&ret_0)->rows + (int(3)))->w) = (_S53 * m_0.rows[int(0)].z - _S52 * m_0.rows[int(0)].z + _S68 * m_0.rows[int(1)].z - _S69 * m_0.rows[int(1)].z - _S70 * m_0.rows[int(2)].z + _S71 * m_0.rows[int(2)].z) * idet_0;

    return ret_0;
}


#line 30 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ float3  get_float3_1(TensorView view_3, uint ind_3, uint feat_ind_0)
{

#line 31
    float _S72 = ((view_3).load<float>((ind_3), (feat_ind_0), (0U)));

#line 31
    float _S73 = ((view_3).load<float>((ind_3), (feat_ind_0), (1U)));

#line 31
    float _S74 = ((view_3).load<float>((ind_3), (feat_ind_0), (2U)));

#line 31
    return make_float3 (_S72, _S73, _S74);
}


#line 46
__device__ Features_0 get_feats_0(TensorView features_0, uint prim_ind_0, uint sh_degree_0)
{

#line 47
    Features_0 feat_0;
    float3  _S75 = get_float3_1(features_0, prim_ind_0, 0U);

#line 48
    (&feat_0)->f0_0 = _S75;
    if(sh_degree_0 > 0U)
    {

#line 50
        float3  _S76 = get_float3_1(features_0, prim_ind_0, 1U);

#line 50
        (&feat_0)->f1_0 = _S76;
        float3  _S77 = get_float3_1(features_0, prim_ind_0, 2U);

#line 51
        (&feat_0)->f2_0 = _S77;
        float3  _S78 = get_float3_1(features_0, prim_ind_0, 3U);

#line 52
        (&feat_0)->f3_0 = _S78;
        if(sh_degree_0 > 1U)
        {

#line 54
            float3  _S79 = get_float3_1(features_0, prim_ind_0, 4U);

#line 54
            (&feat_0)->f4_0 = _S79;
            float3  _S80 = get_float3_1(features_0, prim_ind_0, 5U);

#line 55
            (&feat_0)->f5_0 = _S80;
            float3  _S81 = get_float3_1(features_0, prim_ind_0, 6U);

#line 56
            (&feat_0)->f6_0 = _S81;
            float3  _S82 = get_float3_1(features_0, prim_ind_0, 7U);

#line 57
            (&feat_0)->f7_0 = _S82;
            float3  _S83 = get_float3_1(features_0, prim_ind_0, 8U);

#line 58
            (&feat_0)->f8_0 = _S83;
            if(sh_degree_0 > 2U)
            {

#line 60
                float3  _S84 = get_float3_1(features_0, prim_ind_0, 9U);

#line 60
                (&feat_0)->f9_0 = _S84;
                float3  _S85 = get_float3_1(features_0, prim_ind_0, 10U);

#line 61
                (&feat_0)->f10_0 = _S85;
                float3  _S86 = get_float3_1(features_0, prim_ind_0, 11U);

#line 62
                (&feat_0)->f11_0 = _S86;
                float3  _S87 = get_float3_1(features_0, prim_ind_0, 12U);

#line 63
                (&feat_0)->f12_0 = _S87;
                float3  _S88 = get_float3_1(features_0, prim_ind_0, 13U);

#line 64
                (&feat_0)->f13_0 = _S88;
                float3  _S89 = get_float3_1(features_0, prim_ind_0, 14U);

#line 65
                (&feat_0)->f14_0 = _S89;
                float3  _S90 = get_float3_1(features_0, prim_ind_0, 15U);

#line 66
                (&feat_0)->f15_0 = _S90;

#line 59
            }

#line 53
        }

#line 49
    }

#line 70
    return feat_0;
}


#line 49 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
__device__ float3  eval_sh_col0_0(float3  dir_0, Features_0 * feat_1)
{

#line 50
    return make_float3 (0.282094806432724f) * feat_1->f0_0 + make_float3 (0.5f);
}


__device__ float3  eval_sh_col1_0(float3  dir_1, Features_0 * feat_2)
{



    return make_float3 (-0.48860251903533936f * dir_1.y) * feat_2->f1_0 + make_float3 (0.48860251903533936f * dir_1.z) * feat_2->f2_0 - make_float3 (0.48860251903533936f * dir_1.x) * feat_2->f3_0;
}


__device__ float3  eval_sh_col2_0(float3  dir_2, Features_0 * feat_3)
{

#line 64
    float x_0 = dir_2.x;
    float y_0 = dir_2.y;
    float z_0 = dir_2.z;
    float xx_0 = x_0 * x_0;

#line 67
    float yy_0 = y_0 * y_0;

    return make_float3 (1.09254848957061768f * (x_0 * y_0)) * feat_3->f4_0 + make_float3 (-1.09254848957061768f * (y_0 * z_0)) * feat_3->f5_0 + make_float3 (0.31539157032966614f * (2.0f * (z_0 * z_0) - xx_0 - yy_0)) * feat_3->f6_0 + make_float3 (-1.09254848957061768f * (x_0 * z_0)) * feat_3->f7_0 + make_float3 (0.54627424478530884f * (xx_0 - yy_0)) * feat_3->f8_0;
}




__device__ float3  eval_sh_col3_0(float3  dir_3, Features_0 * feat_4)
{

#line 76
    float x_1 = dir_3.x;
    float y_1 = dir_3.y;
    float z_1 = dir_3.z;
    float xx_1 = x_1 * x_1;

#line 79
    float yy_1 = y_1 * y_1;

#line 79
    float zz_0 = z_1 * z_1;

    float _S91 = 3.0f * xx_1;

    float _S92 = 4.0f * zz_0 - xx_1 - yy_1;
    float _S93 = 3.0f * yy_1;

#line 81
    return make_float3 (-0.59004360437393188f * y_1 * (_S91 - yy_1)) * feat_4->f9_0 + make_float3 (2.89061141014099121f * (x_1 * y_1) * z_1) * feat_4->f10_0 + make_float3 (-0.4570457935333252f * y_1 * _S92) * feat_4->f11_0 + make_float3 (0.37317633628845215f * z_1 * (2.0f * zz_0 - _S91 - _S93)) * feat_4->f12_0 + make_float3 (-0.4570457935333252f * x_1 * _S92) * feat_4->f13_0 + make_float3 (1.44530570507049561f * z_1 * (xx_1 - yy_1)) * feat_4->f14_0 + make_float3 (-0.59004360437393188f * x_1 * (xx_1 - _S93)) * feat_4->f15_0;
}


#line 91
__device__ float3  eval_color_0(float3  dir_4, Features_0 * feat_5, uint sh_degree_1)
{

#line 91
    float3  _S94 = eval_sh_col0_0(dir_4, feat_5);

#line 91
    float3  color_0;

    if(sh_degree_1 > 0U)
    {

#line 93
        float3  _S95 = eval_sh_col1_0(dir_4, feat_5);
        float3  color_1 = _S94 + _S95;
        if(sh_degree_1 > 1U)
        {

#line 95
            float3  _S96 = eval_sh_col2_0(dir_4, feat_5);
            float3  color_2 = color_1 + _S96;
            if(sh_degree_1 > 2U)
            {

#line 97
                float3  _S97 = eval_sh_col3_0(dir_4, feat_5);

#line 97
                color_0 = color_2 + _S97;

#line 97
            }
            else
            {

#line 97
                color_0 = color_2;

#line 97
            }

#line 95
        }
        else
        {

#line 95
            color_0 = color_1;

#line 95
        }

#line 93
    }
    else
    {

#line 93
        color_0 = _S94;

#line 93
    }

#line 102
    return color_0;
}


#line 102
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};


#line 1686 "diff.meta.slang"
__device__ void _d_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * a_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * b_0, float3  dOut_2)
{

#line 1693
    float _S98 = dOut_2.y;

#line 1693
    float _S99 = dOut_2.z;
    float _S100 = dOut_2.x;

#line 1694
    float _S101 = (*a_0).primal_0.z * _S98 + - (*a_0).primal_0.y * _S99;

#line 1694
    float _S102 = - (*a_0).primal_0.z * _S100 + (*a_0).primal_0.x * _S99;

#line 1694
    float _S103 = (*a_0).primal_0.y * _S100 + - (*a_0).primal_0.x * _S98;

#line 1701
    float3  _S104 = make_float3 (- (*b_0).primal_0.z * _S98 + (*b_0).primal_0.y * _S99, (*b_0).primal_0.z * _S100 + - (*b_0).primal_0.x * _S99, - (*b_0).primal_0.y * _S100 + (*b_0).primal_0.x * _S98);

#line 1701
    a_0->primal_0 = (*a_0).primal_0;

#line 1701
    a_0->differential_0 = _S104;
    float3  _S105 = make_float3 (_S101, _S102, _S103);

#line 1702
    b_0->primal_0 = (*b_0).primal_0;

#line 1702
    b_0->differential_0 = _S105;
    return;
}


#line 8883 "hlsl.meta.slang"
__device__ float3  cross_0(float3  left_0, float3  right_0)
{

#line 8897
    float _S106 = left_0.y;

#line 8897
    float _S107 = right_0.z;

#line 8897
    float _S108 = left_0.z;

#line 8897
    float _S109 = right_0.y;
    float _S110 = right_0.x;

#line 8898
    float _S111 = left_0.x;

#line 8896
    return make_float3 (_S106 * _S107 - _S108 * _S109, _S108 * _S110 - _S111 * _S107, _S111 * _S109 - _S106 * _S110);
}


#line 238 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ float3  rotate_vector_0(float3  v_0, float4  q_0)
{


    float3  _S112 = - float3 {q_0.y, q_0.z, q_0.w};

#line 242
    float3  _S113 = make_float3 (2.0f) * cross_0(_S112, v_0);
    return v_0 + make_float3 (q_0.x) * _S113 + cross_0(_S112, _S113);
}


#line 1639 "diff.meta.slang"
__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_3)
{
    float3  x_d_result_0;



    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_3;

#line 1641
    float3  y_d_result_0;

#line 1646
    *&((&y_d_result_0)->x) = (*dpx_2).primal_0.x * dOut_3;

#line 1645
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_3;
    *&((&y_d_result_0)->y) = (*dpx_2).primal_0.y * dOut_3;

#line 1645
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_3;
    *&((&y_d_result_0)->z) = (*dpx_2).primal_0.z * dOut_3;

#line 1646
    dpx_2->primal_0 = (*dpx_2).primal_0;

#line 1646
    dpx_2->differential_0 = x_d_result_0;

#line 1646
    dpy_1->primal_0 = (*dpy_1).primal_0;

#line 1646
    dpy_1->differential_0 = y_d_result_0;



    return;
}


#line 9438 "hlsl.meta.slang"
__device__ float dot_0(float3  x_2, float3  y_2)
{

#line 9438
    int i_0 = int(0);

#line 9438
    float result_5 = 0.0f;

#line 9457
    for(;;)
    {

#line 9457
        if(i_0 < int(3))
        {
        }
        else
        {

#line 9457
            break;
        }

#line 9458
        float result_6 = result_5 + _slang_vector_get_element(x_2, i_0) * _slang_vector_get_element(y_2, i_0);

#line 9457
        i_0 = i_0 + int(1);

#line 9457
        result_5 = result_6;

#line 9457
    }

    return result_5;
}


#line 9438
__device__ float dot_1(float4  x_3, float4  y_3)
{

#line 9438
    int i_1 = int(0);

#line 9438
    float result_7 = 0.0f;

#line 9457
    for(;;)
    {

#line 9457
        if(i_1 < int(4))
        {
        }
        else
        {

#line 9457
            break;
        }

#line 9458
        float result_8 = result_7 + _slang_vector_get_element(x_3, i_1) * _slang_vector_get_element(y_3, i_1);

#line 9457
        i_1 = i_1 + int(1);

#line 9457
        result_7 = result_8;

#line 9457
    }

    return result_7;
}


#line 1 "token paste"
__device__ void _d_abs_0(DiffPair_float_0 * dpx_3, float dOut_4)
{

#line 1915 "diff.meta.slang"
    float _S114 = _slang_select(((*dpx_3).primal_0) > 0.0f, 1.0f,_slang_select(((*dpx_3).primal_0) == 0.0f, 0.0f,-1.0f)) * dOut_4;

#line 1915
    dpx_3->primal_0 = (*dpx_3).primal_0;

#line 1915
    dpx_3->differential_0 = _S114;



    return;
}


#line 2162
__device__ void _d_min_0(DiffPair_float_0 * dpx_4, DiffPair_float_0 * dpy_2, float dOut_5)
{
    DiffPair_float_0 _S115 = *dpx_4;

#line 2164
    float _S116;

#line 2164
    if(((*dpx_4).primal_0) < ((*dpy_2).primal_0))
    {

#line 2164
        _S116 = dOut_5;

#line 2164
    }
    else
    {

#line 2164
        if(((*dpx_4).primal_0) > ((*dpy_2).primal_0))
        {

#line 2164
            _S116 = 0.0f;

#line 2164
        }
        else
        {

#line 2164
            _S116 = 0.5f * dOut_5;

#line 2164
        }

#line 2164
    }

#line 2164
    dpx_4->primal_0 = _S115.primal_0;

#line 2164
    dpx_4->differential_0 = _S116;
    DiffPair_float_0 _S117 = *dpy_2;

#line 2165
    if(((*dpy_2).primal_0) < (_S115.primal_0))
    {

#line 2165
        _S116 = dOut_5;

#line 2165
    }
    else
    {

#line 2165
        if(((*dpy_2).primal_0) > ((*dpx_4).primal_0))
        {

#line 2165
            _S116 = 0.0f;

#line 2165
        }
        else
        {

#line 2165
            _S116 = 0.5f * dOut_5;

#line 2165
        }

#line 2165
    }

#line 2165
    dpy_2->primal_0 = _S117.primal_0;

#line 2165
    dpy_2->differential_0 = _S116;
    return;
}


#line 45 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ float clip_0(float v_1, float minv_0, float maxv_0)
{

#line 46
    return (F32_max(((F32_min((v_1), (maxv_0)))), (minv_0)));
}


#line 68
__device__ void bw_safe_div_0(DiffPair_float_0 * a_1, DiffPair_float_0 * b_1, float R_0)
{

#line 69
    if((F32_abs(((*b_1).primal_0))) < 1.07549441632776457e-20f)
    {

#line 70
        float _S118 = clip_0(R_0 / 1.07549441632776457e-20f, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 70
        a_1->primal_0 = (*a_1).primal_0;

#line 70
        a_1->differential_0 = _S118;

#line 69
    }
    else
    {
        float _S119 = clip_0(R_0 / (*b_1).primal_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 72
        a_1->primal_0 = (*a_1).primal_0;

#line 72
        a_1->differential_0 = _S119;

#line 69
    }

#line 74
    float _S120 = (*b_1).primal_0 * (*b_1).primal_0;
    if(_S120 < 1.07549441632776457e-20f)
    {

#line 76
        float _S121 = clip_0(- (*a_1).primal_0 / 1.07549441632776457e-20f * R_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 76
        b_1->primal_0 = (*b_1).primal_0;

#line 76
        b_1->differential_0 = _S121;

#line 75
    }
    else
    {
        float _S122 = clip_0(- (*a_1).primal_0 / _S120 * R_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 78
        b_1->primal_0 = (*b_1).primal_0;

#line 78
        b_1->differential_0 = _S122;

#line 75
    }

#line 80
    return;
}



__device__ float safe_div_0(float a_2, float b_2)
{

#line 86
    if((F32_abs((b_2))) < 1.07549441632776457e-20f)
    {

#line 87
        return clip_0(a_2 / 1.07549441632776457e-20f, -1.00000002004087734e+20f, 1.00000002004087734e+20f);
    }
    else
    {

#line 89
        return clip_0(a_2 / b_2, -1.00000002004087734e+20f, 1.00000002004087734e+20f);
    }

#line 89
}




__device__ float3  safe_div_1(float3  a_3, float b_3)
{

#line 95
    return make_float3 (safe_div_0(a_3.x, b_3), safe_div_0(a_3.y, b_3), safe_div_0(a_3.z, b_3));
}


#line 137
__device__ float3  l2_normalize_0(float3  x_4)
{

#line 138
    return safe_div_1(x_4, (F32_sqrt(((F32_max((dot_0(x_4, x_4)), (1.07549441632776457e-20f)))))));
}


#line 113
__device__ float3  safe_div_2(float3  a_4, float3  b_4)
{

#line 114
    return make_float3 (safe_div_0(a_4.x, b_4.x), safe_div_0(a_4.y, b_4.y), safe_div_0(a_4.z, b_4.z));
}


#line 123
__device__ float safe_sqrt_0(float a_5)
{

#line 124
    if(a_5 < 1.07549441632776457e-20f)
    {

#line 125
        return 0.0f;
    }
    else
    {

#line 127
        return (F32_sqrt((a_5)));
    }

#line 127
}


#line 133 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
__device__ float2  safe_eliIntersect_0(float3  ro_0, float3  rd_0, float3  ra_0)
{



    float3  ocn_0 = safe_div_2(ro_0, ra_0);
    float3  rdn_0 = safe_div_2(rd_0, ra_0);
    float a_6 = dot_0(rdn_0, rdn_0);
    float bp_0 = - dot_0(ocn_0, rdn_0);
    float3  l_0 = ocn_0 + make_float3 (safe_div_0(bp_0, a_6)) * rdn_0;
    float h_0 = a_6 * (1.0f - dot_0(l_0, l_0));
    float c_0 = dot_0(ocn_0, ocn_0) - 1.0f;
    if(h_0 < 0.0f)
    {

#line 145
        return make_float2 (-1.0f);
    }

#line 146
    float _S123 = bp_0 + float((F32_sign((bp_0)))) * safe_sqrt_0(h_0);
    return make_float2 (safe_div_0(c_0, _S123), safe_div_0(_S123, a_6));
}


#line 179
__device__ float2  safe_ray_intersect_ellipsoid_0(float3  rayo_0, float3  rayd_0, float3  scales_0, float4  quat_0)
{

#line 188
    float2  fminmaxt_0 = safe_eliIntersect_0(rotate_vector_0(rayo_0, quat_0), l2_normalize_0(rotate_vector_0(rayd_0, quat_0)), scales_0);
    float _S124 = fminmaxt_0.x;

#line 189
    float _S125 = fminmaxt_0.y;

#line 189
    return make_float2 ((F32_min((_S124), (_S125))), (F32_max((_S124), (_S125))));
}


#line 290
__device__ ControlPoint_0 safe_intersect_0(float3  rayo_1, float3  rayd_1, float3  scales_1, float3  mean_0, float4  quat_1, float3  color_3, float density_0, uint face_id_0, bool skip_close_0)
{

    float2  minmaxt_0 = safe_ray_intersect_ellipsoid_0(rayo_1 - mean_0, rayd_1, scales_1, quat_1);

    bool _S126 = face_id_0 == 1U;

#line 295
    float t_3;

#line 295
    if(_S126)
    {

#line 295
        t_3 = minmaxt_0.x;

#line 295
    }
    else
    {

#line 295
        t_3 = minmaxt_0.y;

#line 295
    }

#line 295
    float dirac_multi_0;
    if(_S126)
    {

#line 296
        dirac_multi_0 = density_0;

#line 296
    }
    else
    {

#line 296
        dirac_multi_0 = - density_0;

#line 296
    }

    ControlPoint_0 out_0 = { t_3, make_float4 (dirac_multi_0, dirac_multi_0 * color_3.x, dirac_multi_0 * color_3.y, dirac_multi_0 * color_3.z) };


    return out_0;
}


#line 114 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
struct DualModel_0
{
    TensorView means_0;
    TensorView scales_2;
    TensorView quats_0;
    TensorView densities_0;
    TensorView features_1;
    TensorView dL_dmeans_0;
    TensorView dL_dscales_0;
    TensorView dL_dquats_0;
    TensorView dL_ddensities_0;
    TensorView dL_dfeatures_0;
    TensorView dL_drayos_0;
    TensorView dL_drayds_0;
    TensorView dL_dmeans2D_0;
};


#line 223
__device__ ControlPoint_0 load_ctrl_pt_0(uint older_tri_ind_0, DualModel_0 * model_0, float3  origin_0, float3  direction_0, uint sh_degree_2, bool skip_close_1)
{
    uint _S127 = uint((F32_floor((float(older_tri_ind_0 / 2U)))));
    uint _S128 = ((older_tri_ind_0) % (2U));


    float3  _S129 = get_float3_0(model_0->means_0, _S127);

    float3  _S130 = get_float3_0(model_0->scales_2, _S127);
    float4  _S131 = get_float4_0(model_0->quats_0, _S127);

    float _S132 = ((model_0->densities_0).load<float>((_S127)));

    Features_0 older_feat_0 = get_feats_0(model_0->features_1, _S127, sh_degree_2);

#line 236
    Features_0 _S133 = older_feat_0;

#line 236
    float3  _S134 = eval_color_0(direction_0, &_S133, sh_degree_2);


    return safe_intersect_0(origin_0, direction_0, _S130, _S129, _S131, _S134, _S132, _S128, skip_close_1);
}


#line 1 "token paste"
__device__ void _d_exp_0(DiffPair_float_0 * dpx_5, float dOut_6)
{

#line 1915 "diff.meta.slang"
    float _S135 = (F32_exp(((*dpx_5).primal_0))) * dOut_6;

#line 1915
    dpx_5->primal_0 = (*dpx_5).primal_0;

#line 1915
    dpx_5->differential_0 = _S135;



    return;
}


#line 142 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ float safe_exp_0(float v_2)
{

#line 143
    return (F32_exp((clip_0(v_2, -1.00000002004087734e+20f, (F32_log((1.00000002004087734e+20f)))))));
}


#line 164
__device__ void bw_expm1_0(DiffPair_float_0 * v_3, float R_1)
{

#line 165
    float _S136 = (F32_exp(((*v_3).primal_0))) * R_1;

#line 165
    v_3->primal_0 = (*v_3).primal_0;

#line 165
    v_3->differential_0 = _S136;
    return;
}


#line 2098 "diff.meta.slang"
__device__ void _d_pow_0(DiffPair_float_0 * dpx_6, DiffPair_float_0 * dpy_3, float dOut_7)
{

    if(((*dpx_6).primal_0) < 9.99999997475242708e-07f)
    {

#line 2101
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 2101
        dpx_6->differential_0 = 0.0f;

#line 2101
        dpy_3->primal_0 = (*dpy_3).primal_0;

#line 2101
        dpy_3->differential_0 = 0.0f;

#line 2101
    }
    else
    {

#line 2108
        float val_0 = (F32_pow(((*dpx_6).primal_0), ((*dpy_3).primal_0)));

        DiffPair_float_0 _S137 = *dpx_6;

#line 2110
        float _S138 = val_0 * (*dpy_3).primal_0 / (*dpx_6).primal_0 * dOut_7;

#line 2110
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 2110
        dpx_6->differential_0 = _S138;

#line 2110
        float _S139 = val_0 * (F32_log((_S137.primal_0))) * dOut_7;

#line 2110
        dpy_3->primal_0 = (*dpy_3).primal_0;

#line 2110
        dpy_3->differential_0 = _S139;

#line 2101
    }

#line 2116
    return;
}


#line 247 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ float tukey_power_ladder_0(float x_5, float p_0)
{

    float _S140 = (F32_abs((p_0 - 1.0f)));

    return float((F32_sign((x_5)))) * _S140 / p_0 * ((F32_pow(((F32_abs((x_5))) / (F32_max((1.07549441632776457e-20f), (_S140))) + 1.0f), (p_0))) - 1.0f);
}


#line 71 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
__device__ SplineState_0 inverse_update_dual_0(SplineState_0 * new_state_0, ControlPoint_0 * new_ctrl_pt_0, ControlPoint_0 * ctrl_pt_0, float t_min_0, float t_max_0)
{

#line 71
    SplineState_0 _S141 = *new_state_0;

#line 79
    float dt_0 = (F32_max(((&_S141)->t_1 - ctrl_pt_0->t_0), (0.0f)));

#line 22
    float2  _S142 = make_float2 (0.0f);

    float3  _S143 = make_float3 (0.0f);

#line 81
    SplineState_0 state_0 = SplineState_x24init_0(_S142, _S142, _S143, 0.0f, make_float4 (0.0f), 0.0f, _S143);
    float4  _S144 = (&_S141)->drgb_0 - new_ctrl_pt_0->dirac_0;

#line 82
    (&state_0)->drgb_0 = _S144;

    (&state_0)->t_1 = ctrl_pt_0->t_0;

#line 89
    float _S145 = _S144.x;

#line 89
    float area_0 = (F32_max((_S145 * dt_0), (0.0f)));
    float3  _S146 = safe_div_1(make_float3 (_S144.y, _S144.z, _S144.w), _S145);

    float _S147 = (F32_max(((&_S141)->logT_0 - area_0), (0.0f)));

#line 92
    (&state_0)->logT_0 = _S147;
    float _S148 = - area_0;

#line 93
    float _S149 = safe_exp_0(_S148);

#line 93
    float weight_0 = clip_0((1.0f - _S149) * safe_exp_0(- _S147), 0.0f, 1.0f);

    (&state_0)->C_0 = (&_S141)->C_0 - make_float3 (weight_0) * _S146;
    float _S150 = (expm1((_S148)));

#line 96
    float alpha_0 = - _S150;

#line 96
    float segment_depth_val_0;



    if(_S145 < 9.99999997475242708e-07f)
    {

#line 100
        segment_depth_val_0 = alpha_0 * ctrl_pt_0->t_0 + (1.0f - alpha_0) * (&state_0)->t_1;

#line 100
    }
    else
    {
        float _S151 = safe_div_0(1.0f, _S145);

#line 103
        float _S152 = (expm1((_S148)));

#line 103
        segment_depth_val_0 = _S151 * - _S152 - (ctrl_pt_0->t_0 + t_min_0) * _S149 + ((&state_0)->t_1 + t_min_0);

#line 100
    }

#line 109
    *&((&(&_S141)->padding_0)->x) = *&((&(&state_0)->padding_0)->x) - safe_exp_0(- (&state_0)->logT_0) * (F32_max((segment_depth_val_0), (0.0f)));



    float _S153 = tukey_power_ladder_0(((&_S141)->t_1 + (&state_0)->t_1) / 2.0f * 1000.0f, -0.10000000149011612f);
    *&((&(&state_0)->cum_sum_0)->x) = (&_S141)->cum_sum_0.x - weight_0;
    *&((&(&state_0)->cum_sum_0)->y) = (&_S141)->cum_sum_0.y - weight_0 * _S153;
    float _S154 = 2.0f * weight_0;

#line 116
    *&((&(&state_0)->distortion_parts_0)->x) = (&_S141)->distortion_parts_0.x - _S154 * _S153 * (&state_0)->cum_sum_0.x;
    *&((&(&state_0)->distortion_parts_0)->y) = (&_S141)->distortion_parts_0.y - _S154 * (&state_0)->cum_sum_0.y;

    return state_0;
}


#line 65
__device__ SplineState_0 from_dual_0(SplineState_0 * state_1, ControlPoint_0 * ctrl_pt_1)
{

    return *state_1;
}


#line 47
__device__ ControlPoint_0 ControlPoint_x24init_0(float t_4, float4  dirac_1)
{

#line 47
    ControlPoint_0 _S155;

    (&_S155)->t_0 = t_4;
    (&_S155)->dirac_0 = dirac_1;

#line 47
    return _S155;
}


#line 86 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ void atomic_add_float3_0(TensorView view_4, uint ind_4, float3  val_1)
{

#line 87
    float temp_0;
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 0U))), (val_1.x));
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 1U))), (val_1.y));
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 2U))), (val_1.z));
    return;
}


#line 93
__device__ void atomic_add_float4_0(TensorView view_5, uint ind_5, float4  val_2)
{

#line 94
    float temp_1;
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 0U))), (val_2.x));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 1U))), (val_2.y));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 2U))), (val_2.z));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 3U))), (val_2.w));
    return;
}


#line 44 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
__device__ Features_0 Features_x24init_0(float3  f0_1, float3  f1_1, float3  f2_1, float3  f3_1, float3  f4_1, float3  f5_1, float3  f6_1, float3  f7_1, float3  f8_1, float3  f9_1, float3  f10_1, float3  f11_1, float3  f12_1, float3  f13_1, float3  f14_1, float3  f15_1)
{

#line 44
    Features_0 _S156;
    (&_S156)->f0_0 = f0_1;

#line 45
    (&_S156)->f1_0 = f1_1;

#line 45
    (&_S156)->f2_0 = f2_1;

#line 45
    (&_S156)->f3_0 = f3_1;

#line 45
    (&_S156)->f4_0 = f4_1;

#line 45
    (&_S156)->f5_0 = f5_1;

#line 45
    (&_S156)->f6_0 = f6_1;

#line 45
    (&_S156)->f7_0 = f7_1;

#line 45
    (&_S156)->f8_0 = f8_1;

#line 45
    (&_S156)->f9_0 = f9_1;

#line 45
    (&_S156)->f10_0 = f10_1;

#line 45
    (&_S156)->f11_0 = f11_1;

#line 45
    (&_S156)->f12_0 = f12_1;

#line 45
    (&_S156)->f13_0 = f13_1;

#line 45
    (&_S156)->f14_0 = f14_1;

#line 45
    (&_S156)->f15_0 = f15_1;

#line 44
    return _S156;
}


#line 73 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ void atomic_add_float3_1(TensorView view_6, uint ind_6, uint feat_ind_1, float3  val_3)
{

#line 74
    float temp_2;
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 0U))), (val_3.x));
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 1U))), (val_3.y));
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 2U))), (val_3.z));
    return;
}


#line 78
struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_0;
    float4  differential_0;
};


#line 171
struct DiffPair_matrixx3Cfloatx2C4x2C4x3E_0
{
    Matrix<float, 4, 4>  primal_0;
    Matrix<float, 4, 4>  differential_0;
};


#line 1508 "diff.meta.slang"
__device__ void _d_mul_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * left_1, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * right_1, float4  dOut_8)
{

#line 1508
    float _S157 = (*right_1).primal_0.rows[int(0)].x * dOut_8.x;


    Matrix<float, 4, 4>  right_d_result_0;

#line 1520
    *&(((&right_d_result_0)->rows + (int(0)))->x) = (*left_1).primal_0.x * dOut_8.x;

#line 1519
    float sum_0 = _S157 + (*right_1).primal_0.rows[int(0)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = (*left_1).primal_0.x * dOut_8.y;

#line 1519
    float sum_1 = sum_0 + (*right_1).primal_0.rows[int(0)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = (*left_1).primal_0.x * dOut_8.z;

#line 1519
    float sum_2 = sum_1 + (*right_1).primal_0.rows[int(0)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = (*left_1).primal_0.x * dOut_8.w;

#line 1510
    float4  left_d_result_0;

#line 1522
    *&((&left_d_result_0)->x) = sum_2;

#line 1522
    float _S158 = (*right_1).primal_0.rows[int(1)].x * dOut_8.x;

#line 1520
    *&(((&right_d_result_0)->rows + (int(1)))->x) = (*left_1).primal_0.y * dOut_8.x;

#line 1519
    float sum_3 = _S158 + (*right_1).primal_0.rows[int(1)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = (*left_1).primal_0.y * dOut_8.y;

#line 1519
    float sum_4 = sum_3 + (*right_1).primal_0.rows[int(1)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = (*left_1).primal_0.y * dOut_8.z;

#line 1519
    float sum_5 = sum_4 + (*right_1).primal_0.rows[int(1)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = (*left_1).primal_0.y * dOut_8.w;

    *&((&left_d_result_0)->y) = sum_5;

#line 1522
    float _S159 = (*right_1).primal_0.rows[int(2)].x * dOut_8.x;

#line 1520
    *&(((&right_d_result_0)->rows + (int(2)))->x) = (*left_1).primal_0.z * dOut_8.x;

#line 1519
    float sum_6 = _S159 + (*right_1).primal_0.rows[int(2)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = (*left_1).primal_0.z * dOut_8.y;

#line 1519
    float sum_7 = sum_6 + (*right_1).primal_0.rows[int(2)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = (*left_1).primal_0.z * dOut_8.z;

#line 1519
    float sum_8 = sum_7 + (*right_1).primal_0.rows[int(2)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = (*left_1).primal_0.z * dOut_8.w;

    *&((&left_d_result_0)->z) = sum_8;

#line 1522
    float _S160 = (*right_1).primal_0.rows[int(3)].x * dOut_8.x;

#line 1520
    *&(((&right_d_result_0)->rows + (int(3)))->x) = (*left_1).primal_0.w * dOut_8.x;

#line 1519
    float sum_9 = _S160 + (*right_1).primal_0.rows[int(3)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = (*left_1).primal_0.w * dOut_8.y;

#line 1519
    float sum_10 = sum_9 + (*right_1).primal_0.rows[int(3)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = (*left_1).primal_0.w * dOut_8.z;

#line 1519
    float sum_11 = sum_10 + (*right_1).primal_0.rows[int(3)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = (*left_1).primal_0.w * dOut_8.w;

    *&((&left_d_result_0)->w) = sum_11;

#line 1522
    left_1->primal_0 = (*left_1).primal_0;

#line 1522
    left_1->differential_0 = left_d_result_0;

#line 1522
    right_1->primal_0 = (*right_1).primal_0;

#line 1522
    right_1->differential_0 = right_d_result_0;



    return;
}


#line 12721 "hlsl.meta.slang"
__device__ float4  mul_0(float4  left_2, Matrix<float, 4, 4>  right_2)
{

#line 12733
    float4  result_9;

#line 12733
    int j_0 = int(0);
    for(;;)
    {

#line 12734
        if(j_0 < int(4))
        {
        }
        else
        {

#line 12734
            break;
        }

#line 12734
        int i_2 = int(0);

#line 12734
        float sum_12 = 0.0f;


        for(;;)
        {

#line 12737
            if(i_2 < int(4))
            {
            }
            else
            {

#line 12737
                break;
            }
            float sum_13 = sum_12 + _slang_vector_get_element(left_2, i_2) * _slang_vector_get_element(right_2.rows[i_2], j_0);

#line 12737
            i_2 = i_2 + int(1);

#line 12737
            sum_12 = sum_13;

#line 12737
        }



        *_slang_vector_get_element_ptr(&result_9, j_0) = sum_12;

#line 12734
        j_0 = j_0 + int(1);

#line 12734
    }

#line 12743
    return result_9;
}


#line 12721
__device__ float3  mul_1(float3  left_3, Matrix<float, 3, 3>  right_3)
{

#line 12733
    float3  result_10;

#line 12733
    int j_1 = int(0);
    for(;;)
    {

#line 12734
        if(j_1 < int(3))
        {
        }
        else
        {

#line 12734
            break;
        }

#line 12734
        int i_3 = int(0);

#line 12734
        float sum_14 = 0.0f;


        for(;;)
        {

#line 12737
            if(i_3 < int(3))
            {
            }
            else
            {

#line 12737
                break;
            }
            float sum_15 = sum_14 + _slang_vector_get_element(left_3, i_3) * _slang_vector_get_element(right_3.rows[i_3], j_1);

#line 12737
            i_3 = i_3 + int(1);

#line 12737
            sum_14 = sum_15;

#line 12737
        }



        *_slang_vector_get_element_ptr(&result_10, j_1) = sum_14;

#line 12734
        j_1 = j_1 + int(1);

#line 12734
    }

#line 12743
    return result_10;
}


#line 244 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ float3  project_0(float3  xyz_0, Matrix<float, 4, 4>  wct_0)
{
    float4  _S161 = mul_0(make_float4 (xyz_0.x, xyz_0.y, xyz_0.z, 1.0f), wct_0);
    float _S162 = _S161.z;
    return make_float3 (safe_div_0(_S161.x, _S162), safe_div_0(_S161.y, _S162), _S162);
}


#line 80
__device__ void atomic_add_float2_0(TensorView view_7, uint ind_7, float2  val_4)
{

#line 81
    float temp_3;
    *((&temp_3)) = atomicAdd((view_7).data_ptr_at<float>((make_uint2 (ind_7, 0U))), (val_4.x));
    *((&temp_3)) = atomicAdd((view_7).data_ptr_at<float>((make_uint2 (ind_7, 1U))), (val_4.y));
    return;
}


#line 84
struct DiffPair_SplineState_0
{
    SplineState_0 primal_0;
    SplineState_0 differential_0;
};


#line 132
struct DiffPair_ControlPoint_0
{
    ControlPoint_0 primal_0;
    ControlPoint_0 differential_0;
};


#line 124 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
struct s_bwd_prop_update_Intermediates_0
{
    float _S163;
    float _S164;
};


#line 124
__device__ float s_primal_ctx_max_0(float _S165, float _S166)
{

#line 124
    return (F32_max((_S165), (_S166)));
}


#line 124
__device__ float s_primal_ctx_safe_div_0(float _S167, float _S168)
{

#line 124
    return safe_div_0(_S167, _S168);
}


#line 124
__device__ float3  s_primal_ctx_safe_div_1(float3  dpa_0, float dpb_0)
{

#line 94 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    return make_float3 (s_primal_ctx_safe_div_0(dpa_0.x, dpb_0), s_primal_ctx_safe_div_0(dpa_0.y, dpb_0), s_primal_ctx_safe_div_0(dpa_0.z, dpb_0));
}


#line 94
__device__ float s_primal_ctx_safe_expm1_0(float _S169)
{

#line 94
    float _S170 = (expm1((_S169)));

#line 94
    return _S170;
}


#line 94
__device__ float s_primal_ctx_log_0(float _S171)
{

#line 94
    return (F32_log((_S171)));
}


#line 94
__device__ float s_primal_ctx_min_0(float _S172, float _S173)
{

#line 94
    return (F32_min((_S172), (_S173)));
}


#line 94
__device__ float s_primal_ctx_clip_0(float dpv_0, float dpminv_0, float dpmaxv_0)
{

#line 45
    return s_primal_ctx_max_0(s_primal_ctx_min_0(dpv_0, dpmaxv_0), dpminv_0);
}


#line 45
__device__ float s_primal_ctx_exp_0(float _S174)
{

#line 45
    return (F32_exp((_S174)));
}


#line 45
__device__ float s_primal_ctx_safe_exp_0(float dpv_1)
{

#line 142
    return s_primal_ctx_exp_0(s_primal_ctx_clip_0(dpv_1, -1.00000002004087734e+20f, s_primal_ctx_log_0(1.00000002004087734e+20f)));
}


#line 142
__device__ float s_primal_ctx_abs_0(float _S175)
{

#line 142
    return (F32_abs((_S175)));
}


#line 142
__device__ float s_primal_ctx_pow_0(float _S176, float _S177)
{

#line 142
    return (F32_pow((_S176), (_S177)));
}


#line 142
__device__ float s_primal_ctx_tukey_power_ladder_0(float dpx_7, float dpp_0)
{

#line 247
    float _S178 = s_primal_ctx_abs_0(dpp_0 - 1.0f);

#line 247
    return float((F32_sign((dpx_7)))) * _S178 / dpp_0 * (s_primal_ctx_pow_0(s_primal_ctx_abs_0(dpx_7) / s_primal_ctx_max_0(1.07549441632776457e-20f, _S178) + 1.0f, dpp_0) - 1.0f);
}


#line 247
__device__ SplineState_0 s_primal_ctx_update_0(SplineState_0 * dpstate_0, ControlPoint_0 * dpctrl_pt_0, float t_min_1, float t_max_1, float max_prim_size_0, s_bwd_prop_update_Intermediates_0 * _s_diff_ctx_0)
{

#line 129 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    _s_diff_ctx_0->_S163 = 0.0f;

#line 129
    _s_diff_ctx_0->_S164 = 0.0f;

#line 155
    _s_diff_ctx_0->_S164 = 0.0f;

#line 134
    float2  _S179 = make_float2 (0.0f);

#line 134
    float3  _S180 = make_float3 (0.0f);
    float4  _S181 = dpstate_0->drgb_0 + dpctrl_pt_0->dirac_0;

#line 141
    float _S182 = dpstate_0->drgb_0.x;

#line 141
    float _S183 = s_primal_ctx_max_0(_S182 * s_primal_ctx_max_0(dpctrl_pt_0->t_0 - dpstate_0->t_1, 0.0f), 0.0f);

#line 141
    float3  _S184 = s_primal_ctx_safe_div_1(make_float3 (dpstate_0->drgb_0.y, dpstate_0->drgb_0.z, dpstate_0->drgb_0.w), _S182);

#line 141
    float _S185 = s_primal_ctx_max_0(_S183 + dpstate_0->logT_0, 0.0f);

#line 146
    float _S186 = - _S183;

#line 146
    float _S187 = s_primal_ctx_safe_expm1_0(_S186);

#line 146
    _s_diff_ctx_0->_S163 = _S187;

#line 146
    float alpha_1 = - _S187;

#line 146
    float _S188 = s_primal_ctx_safe_exp_0(- dpstate_0->logT_0);

#line 146
    float _S189 = s_primal_ctx_clip_0(alpha_1 * _S188, 0.0f, 1.0f);

    float3  _S190 = dpstate_0->C_0 + make_float3 (_S189) * _S184;

#line 148
    float _S191 = dpctrl_pt_0->t_0;

#line 148
    float segment_depth_val_1;



    if(_S182 < 9.99999997475242708e-07f)
    {

#line 152
        segment_depth_val_1 = alpha_1 * dpctrl_pt_0->t_0 + (1.0f - alpha_1) * dpstate_0->t_1;

#line 152
    }
    else
    {

#line 152
        float _S192 = s_primal_ctx_safe_div_0(1.0f, _S182);

#line 152
        float _S193 = s_primal_ctx_safe_expm1_0(_S186);


        _s_diff_ctx_0->_S164 = _S193;

#line 155
        segment_depth_val_1 = _S192 * - _S193 - (dpctrl_pt_0->t_0 + t_min_1) * s_primal_ctx_safe_exp_0(_S186) + (dpstate_0->t_1 + t_min_1);

#line 152
    }

#line 161
    float _S194 = *&((&dpstate_0->padding_0)->x) + _S188 * s_primal_ctx_max_0(segment_depth_val_1, 0.0f);

#line 161
    SplineState_0 _S195;

#line 161
    (&_S195)->distortion_parts_0 = _S179;

#line 161
    (&_S195)->cum_sum_0 = _S179;

#line 161
    (&_S195)->padding_0 = _S180;

#line 161
    (&_S195)->t_1 = _S191;

#line 161
    (&_S195)->drgb_0 = _S181;

#line 161
    (&_S195)->logT_0 = _S185;

#line 161
    (&_S195)->C_0 = _S190;

#line 161
    *&((&(&_S195)->padding_0)->x) = _S194;

#line 161
    float _S196 = s_primal_ctx_tukey_power_ladder_0((_S195.t_1 + dpstate_0->t_1) / 2.0f * 1000.0f, -0.10000000149011612f);

#line 167
    float _S197 = 2.0f * _S189;

#line 167
    float _S198 = dpstate_0->cum_sum_0.x;

#line 167
    *&((&(&_S195)->distortion_parts_0)->x) = dpstate_0->distortion_parts_0.x + _S197 * _S196 * _S198;
    float _S199 = dpstate_0->cum_sum_0.y;

#line 168
    *&((&(&_S195)->distortion_parts_0)->y) = dpstate_0->distortion_parts_0.y + _S197 * _S199;

#line 168
    *&((&(&_S195)->cum_sum_0)->x) = _S198 + _S189;

#line 168
    *&((&(&_S195)->cum_sum_0)->y) = _S199 + _S189 * _S196;

#line 168
    return _S195;
}


#line 169 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_pow_0(DiffPair_float_0 * _S200, DiffPair_float_0 * _S201, float _S202)
{

#line 169
    _d_pow_0(_S200, _S201, _S202);

#line 169
    return;
}


#line 169
__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S203, DiffPair_float_0 * _S204, float _S205)
{

#line 169
    _d_max_0(_S203, _S204, _S205);

#line 169
    return;
}


#line 169
__device__ void s_bwd_prop_abs_0(DiffPair_float_0 * _S206, float _S207)
{

#line 169
    _d_abs_0(_S206, _S207);

#line 169
    return;
}


#line 247 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_tukey_power_ladder_0(DiffPair_float_0 * dpx_8, DiffPair_float_0 * dpp_1, float _s_dOut_0)
{

#line 247
    float _S208 = s_primal_ctx_abs_0((*dpx_8).primal_0);


    float _S209 = (*dpp_1).primal_0 - 1.0f;

#line 250
    float _S210 = s_primal_ctx_abs_0(_S209);

#line 250
    float _S211 = s_primal_ctx_max_0(1.07549441632776457e-20f, _S210);

#line 250
    float _S212 = _S211 * _S211;
    float _S213 = float((F32_sign(((*dpx_8).primal_0))));

#line 251
    float _S214 = _S213 * _S210;

#line 251
    float _S215 = (*dpp_1).primal_0 * (*dpp_1).primal_0;

#line 251
    float _S216 = _S208 / _S211 + 1.0f;

#line 251
    float _S217 = _S214 / (*dpp_1).primal_0 * _s_dOut_0;

#line 251
    float _S218 = (s_primal_ctx_pow_0(_S216, (*dpp_1).primal_0) - 1.0f) * _s_dOut_0;

#line 251
    DiffPair_float_0 _S219;

#line 251
    (&_S219)->primal_0 = _S216;

#line 251
    (&_S219)->differential_0 = 0.0f;

#line 251
    DiffPair_float_0 _S220;

#line 251
    (&_S220)->primal_0 = (*dpp_1).primal_0;

#line 251
    (&_S220)->differential_0 = 0.0f;

#line 251
    s_bwd_prop_pow_0(&_S219, &_S220, _S217);

#line 251
    float _S221 = _S218 / _S215;

#line 251
    float _S222 = _S214 * - _S221;

#line 251
    float _S223 = _S213 * ((*dpp_1).primal_0 * _S221);

#line 250
    float _S224 = _S219.differential_0 / _S212;

#line 250
    float _S225 = _S208 * - _S224;

#line 250
    float _S226 = _S211 * _S224;

#line 250
    DiffPair_float_0 _S227;

#line 250
    (&_S227)->primal_0 = 1.07549441632776457e-20f;

#line 250
    (&_S227)->differential_0 = 0.0f;

#line 250
    DiffPair_float_0 _S228;

#line 250
    (&_S228)->primal_0 = _S210;

#line 250
    (&_S228)->differential_0 = 0.0f;

#line 250
    s_bwd_prop_max_0(&_S227, &_S228, _S225);

#line 250
    float _S229 = _S223 + _S228.differential_0;

#line 250
    DiffPair_float_0 _S230;

#line 250
    (&_S230)->primal_0 = _S209;

#line 250
    (&_S230)->differential_0 = 0.0f;

#line 250
    s_bwd_prop_abs_0(&_S230, _S229);

#line 249
    DiffPair_float_0 _S231;

#line 249
    (&_S231)->primal_0 = (*dpx_8).primal_0;

#line 249
    (&_S231)->differential_0 = 0.0f;

#line 249
    s_bwd_prop_abs_0(&_S231, _S226);

#line 1230 "core.meta.slang"
    float _S232 = _S220.differential_0 + _S222 + _S230.differential_0;

#line 1230
    dpp_1->primal_0 = (*dpp_1).primal_0;

#line 1230
    dpp_1->differential_0 = _S232;

#line 1230
    dpx_8->primal_0 = (*dpx_8).primal_0;

#line 1230
    dpx_8->differential_0 = _S231.differential_0;

#line 247 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 247
__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S233, float _S234)
{

#line 247
    _d_exp_0(_S233, _S234);

#line 247
    return;
}


#line 247
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S235, DiffPair_float_0 * _S236, float _S237)
{

#line 247
    _d_min_0(_S235, _S236, _S237);

#line 247
    return;
}


#line 45
__device__ void s_bwd_prop_clip_0(DiffPair_float_0 * dpv_2, DiffPair_float_0 * dpminv_1, DiffPair_float_0 * dpmaxv_1, float _s_dOut_1)
{

#line 46
    DiffPair_float_0 _S238;

#line 46
    (&_S238)->primal_0 = s_primal_ctx_min_0((*dpv_2).primal_0, (*dpmaxv_1).primal_0);

#line 46
    (&_S238)->differential_0 = 0.0f;

#line 46
    DiffPair_float_0 _S239;

#line 46
    (&_S239)->primal_0 = (*dpminv_1).primal_0;

#line 46
    (&_S239)->differential_0 = 0.0f;

#line 46
    s_bwd_prop_max_0(&_S238, &_S239, _s_dOut_1);

#line 46
    DiffPair_float_0 _S240;

#line 46
    (&_S240)->primal_0 = (*dpv_2).primal_0;

#line 46
    (&_S240)->differential_0 = 0.0f;

#line 46
    DiffPair_float_0 _S241;

#line 46
    (&_S241)->primal_0 = (*dpmaxv_1).primal_0;

#line 46
    (&_S241)->differential_0 = 0.0f;

#line 46
    s_bwd_prop_min_0(&_S240, &_S241, _S238.differential_0);

#line 46
    dpmaxv_1->primal_0 = (*dpmaxv_1).primal_0;

#line 46
    dpmaxv_1->differential_0 = _S241.differential_0;

#line 46
    dpminv_1->primal_0 = (*dpminv_1).primal_0;

#line 46
    dpminv_1->differential_0 = _S239.differential_0;

#line 46
    dpv_2->primal_0 = (*dpv_2).primal_0;

#line 46
    dpv_2->differential_0 = _S240.differential_0;

#line 45
    return;
}


#line 142
__device__ void s_bwd_prop_safe_exp_0(DiffPair_float_0 * dpv_3, float _s_dOut_2)
{

#line 142
    float _S242 = s_primal_ctx_log_0(1.00000002004087734e+20f);
    DiffPair_float_0 _S243;

#line 143
    (&_S243)->primal_0 = s_primal_ctx_clip_0((*dpv_3).primal_0, -1.00000002004087734e+20f, _S242);

#line 143
    (&_S243)->differential_0 = 0.0f;

#line 143
    s_bwd_prop_exp_0(&_S243, _s_dOut_2);

#line 143
    DiffPair_float_0 _S244;

#line 143
    (&_S244)->primal_0 = (*dpv_3).primal_0;

#line 143
    (&_S244)->differential_0 = 0.0f;

#line 143
    DiffPair_float_0 _S245;

#line 143
    (&_S245)->primal_0 = -1.00000002004087734e+20f;

#line 143
    (&_S245)->differential_0 = 0.0f;

#line 143
    DiffPair_float_0 _S246;

#line 143
    (&_S246)->primal_0 = _S242;

#line 143
    (&_S246)->differential_0 = 0.0f;

#line 143
    s_bwd_prop_clip_0(&_S244, &_S245, &_S246, _S243.differential_0);

#line 143
    dpv_3->primal_0 = (*dpv_3).primal_0;

#line 143
    dpv_3->differential_0 = _S244.differential_0;

#line 142
    return;
}


#line 142
__device__ void s_bwd_prop_safe_expm1_0(DiffPair_float_0 * _S247, float _S248)
{

#line 142
    bw_expm1_0(_S247, _S248);

#line 142
    return;
}


#line 142
__device__ void s_bwd_prop_safe_div_0(DiffPair_float_0 * _S249, DiffPair_float_0 * _S250, float _S251)
{

#line 142
    bw_safe_div_0(_S249, _S250, _S251);

#line 142
    return;
}


#line 94
__device__ void s_bwd_prop_safe_div_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpa_1, DiffPair_float_0 * dpb_1, float3  _s_dOut_3)
{
    float _S252 = (*dpa_1).primal_0.x;
    float _S253 = (*dpa_1).primal_0.y;
    DiffPair_float_0 _S254;

#line 98
    (&_S254)->primal_0 = (*dpa_1).primal_0.z;

#line 98
    (&_S254)->differential_0 = 0.0f;

#line 98
    DiffPair_float_0 _S255;

#line 98
    (&_S255)->primal_0 = (*dpb_1).primal_0;

#line 98
    (&_S255)->differential_0 = 0.0f;

#line 98
    s_bwd_prop_safe_div_0(&_S254, &_S255, _s_dOut_3.z);

#line 97
    DiffPair_float_0 _S256;

#line 97
    (&_S256)->primal_0 = _S253;

#line 97
    (&_S256)->differential_0 = 0.0f;

#line 97
    DiffPair_float_0 _S257;

#line 97
    (&_S257)->primal_0 = (*dpb_1).primal_0;

#line 97
    (&_S257)->differential_0 = 0.0f;

#line 97
    s_bwd_prop_safe_div_0(&_S256, &_S257, _s_dOut_3.y);

#line 96
    DiffPair_float_0 _S258;

#line 96
    (&_S258)->primal_0 = _S252;

#line 96
    (&_S258)->differential_0 = 0.0f;

#line 96
    DiffPair_float_0 _S259;

#line 96
    (&_S259)->primal_0 = (*dpb_1).primal_0;

#line 96
    (&_S259)->differential_0 = 0.0f;

#line 96
    s_bwd_prop_safe_div_0(&_S258, &_S259, _s_dOut_3.x);

#line 1230 "core.meta.slang"
    float _S260 = _S255.differential_0 + _S257.differential_0 + _S259.differential_0;

#line 1230
    dpb_1->primal_0 = (*dpb_1).primal_0;

#line 1230
    dpb_1->differential_0 = _S260;

#line 1230
    float3  _S261 = make_float3 (_S258.differential_0, _S256.differential_0, _S254.differential_0);

#line 1230
    dpa_1->primal_0 = (*dpa_1).primal_0;

#line 1230
    dpa_1->differential_0 = _S261;

#line 94 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 124 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
__device__ void s_bwd_prop_update_0(DiffPair_SplineState_0 * dpstate_1, DiffPair_ControlPoint_0 * dpctrl_pt_1, float t_min_2, float t_max_2, float max_prim_size_1, SplineState_0 * _s_dOut_4, s_bwd_prop_update_Intermediates_0 * _s_diff_ctx_1)
{

#line 124
    DiffPair_SplineState_0 _S262 = *dpstate_1;

#line 124
    DiffPair_ControlPoint_0 _S263 = *dpctrl_pt_1;

#line 132
    float _S264 = (*dpctrl_pt_1).primal_0.t_0 - (*dpstate_1).primal_0.t_1;

#line 132
    float _S265 = s_primal_ctx_max_0(_S264, 0.0f);

    float2  _S266 = make_float2 (0.0f);

#line 134
    float3  _S267 = make_float3 (0.0f);
    float4  _S268 = (*dpstate_1).primal_0.drgb_0 + (*dpctrl_pt_1).primal_0.dirac_0;

#line 141
    float _S269 = (*dpstate_1).primal_0.drgb_0.x;

#line 141
    float _S270 = _S269 * _S265;

#line 141
    float _S271 = s_primal_ctx_max_0(_S270, 0.0f);

    float3  _S272 = make_float3 ((*dpstate_1).primal_0.drgb_0.y, (*dpstate_1).primal_0.drgb_0.z, (*dpstate_1).primal_0.drgb_0.w);

#line 143
    float3  _S273 = s_primal_ctx_safe_div_1(_S272, _S269);

    float _S274 = _S271 + (*dpstate_1).primal_0.logT_0;

#line 145
    float _S275 = s_primal_ctx_max_0(_S274, 0.0f);
    float _S276 = - _S271;

#line 146
    float alpha_2 = - _s_diff_ctx_1->_S163;
    float _S277 = - (*dpstate_1).primal_0.logT_0;

#line 147
    float _S278 = s_primal_ctx_safe_exp_0(_S277);

#line 147
    float _S279 = alpha_2 * _S278;

#line 147
    float _S280 = s_primal_ctx_clip_0(_S279, 0.0f, 1.0f);
    float3  _S281 = make_float3 (_S280);

#line 148
    float3  _S282 = (*dpstate_1).primal_0.C_0 + make_float3 (_S280) * _S273;



    bool _S283 = _S269 < 9.99999997475242708e-07f;

#line 152
    float segment_depth_val_2;

#line 152
    float _S284;

#line 152
    float _S285;

#line 152
    float _S286;

#line 152
    float _S287;

#line 152
    float _S288;

#line 152
    if(_S283)
    {

#line 153
        float _S289 = 1.0f - alpha_2;

#line 153
        segment_depth_val_2 = alpha_2 * _S263.primal_0.t_0 + _S289 * _S262.primal_0.t_1;

#line 153
        _S284 = 0.0f;

#line 153
        _S285 = 0.0f;

#line 153
        _S286 = 0.0f;

#line 153
        _S287 = 0.0f;

#line 153
        _S288 = _S289;

#line 153
    }
    else
    {

#line 153
        float _S290 = s_primal_ctx_safe_div_0(1.0f, _S269);

        float _S291 = - _s_diff_ctx_1->_S164;
        float _S292 = _S263.primal_0.t_0 + t_min_2;

#line 156
        float _S293 = s_primal_ctx_safe_exp_0(_S276);

#line 156
        segment_depth_val_2 = _S290 * _S291 - _S292 * _S293 + (_S262.primal_0.t_1 + t_min_2);

#line 156
        _S284 = _S292;

#line 156
        _S285 = _S293;

#line 156
        _S286 = _S290;

#line 156
        _S287 = _S291;

#line 156
        _S288 = 0.0f;

#line 156
    }

#line 156
    float _S294 = s_primal_ctx_max_0(segment_depth_val_2, 0.0f);

#line 161
    float _S295 = _S262.primal_0.padding_0.x + _S278 * _S294;

#line 161
    SplineState_0 _S296;

#line 161
    (&_S296)->distortion_parts_0 = _S266;

#line 161
    (&_S296)->cum_sum_0 = _S266;

#line 161
    (&_S296)->padding_0 = _S267;

#line 161
    (&_S296)->t_1 = _S263.primal_0.t_0;

#line 161
    (&_S296)->drgb_0 = _S268;

#line 161
    (&_S296)->logT_0 = _S275;

#line 161
    (&_S296)->C_0 = _S282;

#line 161
    *&((&(&_S296)->padding_0)->x) = _S295;

#line 166
    float _S297 = (_S296.t_1 + _S262.primal_0.t_1) / 2.0f * 1000.0f;

#line 166
    float _S298 = s_primal_ctx_tukey_power_ladder_0(_S297, -0.10000000149011612f);
    float _S299 = 2.0f * _S280;

#line 167
    float _S300 = _S299 * _S298;

#line 167
    float _S301 = _S262.primal_0.cum_sum_0.x;
    float _S302 = _S262.primal_0.cum_sum_0.y;

#line 148
    SplineState_0 _S303 = SplineState_x24_syn_dzero_0();

#line 148
    _S296 = *_s_dOut_4;

#line 148
    *&((&(&_S296)->cum_sum_0)->y) = 0.0f;

#line 170
    float _S304 = _S280 * *&((&_s_dOut_4->cum_sum_0)->y);

#line 170
    float _S305 = _S298 * *&((&_s_dOut_4->cum_sum_0)->y);

#line 170
    *&((&(&_S296)->cum_sum_0)->x) = 0.0f;

#line 170
    *&((&(&_S296)->distortion_parts_0)->y) = 0.0f;

#line 168
    float _S306 = _S302 * *&((&_s_dOut_4->distortion_parts_0)->y);

#line 168
    float _S307 = *&((&_s_dOut_4->cum_sum_0)->y) + _S299 * *&((&_s_dOut_4->distortion_parts_0)->y);

#line 168
    *&((&(&_S296)->distortion_parts_0)->x) = 0.0f;

#line 167
    float _S308 = _S301 * *&((&_s_dOut_4->distortion_parts_0)->x);

#line 167
    float2  _S309 = make_float2 (*&((&_s_dOut_4->cum_sum_0)->x) + _S300 * *&((&_s_dOut_4->distortion_parts_0)->x), _S307);

#line 167
    float _S310 = 2.0f * (_S306 + _S298 * _S308);

#line 167
    float2  _S311 = make_float2 (*&((&_s_dOut_4->distortion_parts_0)->x), *&((&_s_dOut_4->distortion_parts_0)->y));

#line 166
    float _S312 = _S304 + _S299 * _S308;

#line 166
    DiffPair_float_0 _S313;

#line 166
    (&_S313)->primal_0 = _S297;

#line 166
    (&_S313)->differential_0 = 0.0f;

#line 166
    DiffPair_float_0 _S314;

#line 166
    (&_S314)->primal_0 = -0.10000000149011612f;

#line 166
    (&_S314)->differential_0 = 0.0f;

#line 166
    s_bwd_prop_tukey_power_ladder_0(&_S313, &_S314, _S312);

#line 166
    float _S315 = 0.5f * (1000.0f * _S313.differential_0);

#line 161
    SplineState_0 _S316 = _S303;

#line 161
    (&_S316)->t_1 = _S315;

#line 161
    SplineState_0 _S317 = _S296;

#line 161
    SplineState_0 _S318 = _S316;

#line 161
    SplineState_0 _S319 = SplineState_x24_syn_dadd_0(&_S317, &_S318);

#line 161
    _S296 = _S319;

#line 161
    *&((&(&_S296)->padding_0)->x) = 0.0f;

#line 161
    float _S320 = _S278 * _S319.padding_0.x;

#line 161
    float _S321 = _S294 * _S319.padding_0.x;

#line 161
    float3  _S322 = _S267;

#line 161
    *&((&_S322)->x) = _S319.padding_0.x;

#line 160
    DiffPair_float_0 _S323;

#line 160
    (&_S323)->primal_0 = segment_depth_val_2;

#line 160
    (&_S323)->differential_0 = 0.0f;

#line 160
    DiffPair_float_0 _S324;

#line 160
    (&_S324)->primal_0 = 0.0f;

#line 160
    (&_S324)->differential_0 = 0.0f;

#line 160
    s_bwd_prop_max_0(&_S323, &_S324, _S320);

#line 160
    DiffPair_float_0 _S325 = _S323;

#line 147
    float _S326 = _S305 + *&((&_s_dOut_4->cum_sum_0)->x) + _S310;

#line 147
    SplineState_0 _S327 = _S303;

#line 147
    (&_S327)->cum_sum_0 = _S309;

#line 147
    (&_S327)->distortion_parts_0 = _S311;

#line 147
    (&_S327)->padding_0 = _S322;

#line 147
    SplineState_0 _S328 = _S303;

#line 147
    SplineState_0 _S329 = _S327;

#line 147
    SplineState_0 _S330 = SplineState_x24_syn_dadd_0(&_S328, &_S329);

#line 147
    SplineState_0 _S331 = _S296;

#line 147
    SplineState_0 _S332 = _S303;

#line 147
    SplineState_0 _S333 = SplineState_x24_syn_dadd_0(&_S331, &_S332);

#line 147
    if(_S283)
    {

#line 153
        float _S334 = alpha_2 * _S325.differential_0;

#line 1230 "core.meta.slang"
        float _S335 = _S288 * _S325.differential_0 + _S315;

#line 1230
        segment_depth_val_2 = - (_S262.primal_0.t_1 * _S325.differential_0) + _S263.primal_0.t_0 * _S325.differential_0;

#line 1230
        _S284 = 0.0f;

#line 1230
        _S285 = 0.0f;

#line 1230
        _S286 = _S335;

#line 1230
        _S287 = _S334;

#line 1230
    }
    else
    {

#line 156 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
        float _S336 = - _S325.differential_0;

#line 156
        float _S337 = _S284 * _S336;

#line 156
        float _S338 = _S285 * _S336;

#line 156
        DiffPair_float_0 _S339;

#line 156
        (&_S339)->primal_0 = _S276;

#line 156
        (&_S339)->differential_0 = 0.0f;

#line 156
        s_bwd_prop_safe_exp_0(&_S339, _S337);

#line 155
        float _S340 = _S287 * _S325.differential_0;

#line 155
        float _S341 = - (_S286 * _S325.differential_0);

#line 155
        DiffPair_float_0 _S342;

#line 155
        (&_S342)->primal_0 = _S276;

#line 155
        (&_S342)->differential_0 = 0.0f;

#line 155
        s_bwd_prop_safe_expm1_0(&_S342, _S341);

#line 155
        DiffPair_float_0 _S343;

#line 155
        (&_S343)->primal_0 = 1.0f;

#line 155
        (&_S343)->differential_0 = 0.0f;

#line 155
        DiffPair_float_0 _S344;

#line 155
        (&_S344)->primal_0 = _S269;

#line 155
        (&_S344)->differential_0 = 0.0f;

#line 155
        s_bwd_prop_safe_div_0(&_S343, &_S344, _S340);

#line 1230 "core.meta.slang"
        float _S345 = _S325.differential_0 + _S315;

#line 146 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
        float _S346 = _S339.differential_0 + _S342.differential_0;

#line 146
        segment_depth_val_2 = 0.0f;

#line 146
        _S284 = _S346;

#line 146
        _S285 = _S344.differential_0;

#line 146
        _S286 = _S345;

#line 146
        _S287 = _S338;

#line 146
    }

    float3  _S347 = _S281 * _S333.C_0;

#line 148
    float3  _S348 = _S273 * _S333.C_0;

#line 147
    float _S349 = _S348.x + _S348.y + _S348.z + _S326;

#line 147
    DiffPair_float_0 _S350;

#line 147
    (&_S350)->primal_0 = _S279;

#line 147
    (&_S350)->differential_0 = 0.0f;

#line 147
    DiffPair_float_0 _S351;

#line 147
    (&_S351)->primal_0 = 0.0f;

#line 147
    (&_S351)->differential_0 = 0.0f;

#line 147
    DiffPair_float_0 _S352;

#line 147
    (&_S352)->primal_0 = 1.0f;

#line 147
    (&_S352)->differential_0 = 0.0f;

#line 147
    s_bwd_prop_clip_0(&_S350, &_S351, &_S352, _S349);

#line 147
    float _S353 = _S278 * _S350.differential_0;

#line 147
    float _S354 = alpha_2 * _S350.differential_0 + _S321;

#line 147
    DiffPair_float_0 _S355;

#line 147
    (&_S355)->primal_0 = _S277;

#line 147
    (&_S355)->differential_0 = 0.0f;

#line 147
    s_bwd_prop_safe_exp_0(&_S355, _S354);

#line 147
    float _S356 = - _S355.differential_0;

#line 146
    float _S357 = - (_S353 + segment_depth_val_2);

#line 146
    DiffPair_float_0 _S358;

#line 146
    (&_S358)->primal_0 = _S276;

#line 146
    (&_S358)->differential_0 = 0.0f;

#line 146
    s_bwd_prop_safe_expm1_0(&_S358, _S357);

#line 146
    float _S359 = - (_S358.differential_0 + _S284);

#line 145
    DiffPair_float_0 _S360;

#line 145
    (&_S360)->primal_0 = _S274;

#line 145
    (&_S360)->differential_0 = 0.0f;

#line 145
    DiffPair_float_0 _S361;

#line 145
    (&_S361)->primal_0 = 0.0f;

#line 145
    (&_S361)->differential_0 = 0.0f;

#line 145
    s_bwd_prop_max_0(&_S360, &_S361, _S333.logT_0);

#line 1230 "core.meta.slang"
    float _S362 = _S356 + _S360.differential_0;

#line 143 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S363;

#line 143
    (&_S363)->primal_0 = _S272;

#line 143
    (&_S363)->differential_0 = _S267;

#line 143
    DiffPair_float_0 _S364;

#line 143
    (&_S364)->primal_0 = _S269;

#line 143
    (&_S364)->differential_0 = 0.0f;

#line 143
    s_bwd_prop_safe_div_1(&_S363, &_S364, _S347);

#line 141
    float _S365 = _S359 + _S360.differential_0;

#line 141
    DiffPair_float_0 _S366;

#line 141
    (&_S366)->primal_0 = _S270;

#line 141
    (&_S366)->differential_0 = 0.0f;

#line 141
    DiffPair_float_0 _S367;

#line 141
    (&_S367)->primal_0 = 0.0f;

#line 141
    (&_S367)->differential_0 = 0.0f;

#line 141
    s_bwd_prop_max_0(&_S366, &_S367, _S365);

#line 141
    float _S368 = _S269 * _S366.differential_0;

#line 2246 "core.meta.slang"
    float4  _S369 = _S333.drgb_0 + make_float4 (_S364.differential_0 + _S265 * _S366.differential_0 + _S285, _S363.differential_0.x, _S363.differential_0.y, _S363.differential_0.z);

#line 132 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    DiffPair_float_0 _S370;

#line 132
    (&_S370)->primal_0 = _S264;

#line 132
    (&_S370)->differential_0 = 0.0f;

#line 132
    DiffPair_float_0 _S371;

#line 132
    (&_S371)->primal_0 = 0.0f;

#line 132
    (&_S371)->differential_0 = 0.0f;

#line 132
    s_bwd_prop_max_0(&_S370, &_S371, _S368);

#line 1230 "core.meta.slang"
    float _S372 = - _S370.differential_0 + _S286;

#line 1230
    float _S373 = _S333.t_1 + _S370.differential_0 + _S287;

#line 1230
    ControlPoint_0 _S374 = ControlPoint_x24_syn_dzero_0();

#line 1230
    (&_S374)->dirac_0 = _S333.drgb_0;

#line 1230
    (&_S374)->t_0 = _S373;

#line 1230
    dpctrl_pt_1->primal_0 = (*dpctrl_pt_1).primal_0;

#line 1230
    dpctrl_pt_1->differential_0 = _S374;

#line 1230
    SplineState_0 _S375 = _S303;

#line 1230
    (&_S375)->C_0 = _S333.C_0;

#line 1230
    (&_S375)->logT_0 = _S362;

#line 1230
    (&_S375)->drgb_0 = _S369;

#line 1230
    (&_S375)->t_1 = _S372;

#line 1230
    SplineState_0 _S376 = _S330;

#line 1230
    SplineState_0 _S377 = _S375;

#line 1230
    SplineState_0 _S378 = SplineState_x24_syn_dadd_0(&_S376, &_S377);

#line 1230
    dpstate_1->primal_0 = (*dpstate_1).primal_0;

#line 1230
    dpstate_1->differential_0 = _S378;

#line 124 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    return;
}


#line 124
__device__ void s_bwd_update_0(DiffPair_SplineState_0 * _S379, DiffPair_ControlPoint_0 * _S380, float _S381, float _S382, float _S383, SplineState_0 * _S384)
{



    SplineState_0 _S385 = (*_S379).primal_0;

#line 129
    ControlPoint_0 _S386 = (*_S380).primal_0;

#line 129
    s_bwd_prop_update_Intermediates_0 _S387;

#line 129
    SplineState_0 _S388 = s_primal_ctx_update_0(&_S385, &_S386, _S381, _S382, _S383, &_S387);

#line 129
    s_bwd_prop_update_Intermediates_0 _S389 = _S387;

#line 129
    s_bwd_prop_update_0(_S379, _S380, _S381, _S382, _S383, _S384, &_S389);

#line 129
    return;
}


#line 167 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ float3  s_primal_ctx_cross_0(float3  _S390, float3  _S391)
{

#line 167
    return cross_0(_S390, _S391);
}


#line 167
__device__ float3  s_primal_ctx_rotate_vector_0(float3  dpv_4, float4  dpq_0)
{

#line 242 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    float3  _S392 = - float3 {dpq_0.y, dpq_0.z, dpq_0.w};

#line 242
    float3  _S393 = make_float3 (2.0f) * s_primal_ctx_cross_0(_S392, dpv_4);

#line 242
    return dpv_4 + make_float3 (dpq_0.x) * _S393 + s_primal_ctx_cross_0(_S392, _S393);
}


#line 242
__device__ float s_primal_ctx_dot_0(float3  _S394, float3  _S395)
{

#line 242
    return dot_0(_S394, _S395);
}


#line 242
__device__ float s_primal_ctx_sqrt_0(float _S396)
{

#line 242
    return (F32_sqrt((_S396)));
}


#line 242
__device__ float3  s_primal_ctx_l2_normalize_0(float3  dpx_9)
{

#line 137
    return s_primal_ctx_safe_div_1(dpx_9, s_primal_ctx_sqrt_0(s_primal_ctx_max_0(s_primal_ctx_dot_0(dpx_9, dpx_9), 1.07549441632776457e-20f)));
}


#line 137
__device__ float3  s_primal_ctx_safe_div_2(float3  dpa_2, float3  dpb_2)
{

#line 113
    return make_float3 (s_primal_ctx_safe_div_0(dpa_2.x, dpb_2.x), s_primal_ctx_safe_div_0(dpa_2.y, dpb_2.y), s_primal_ctx_safe_div_0(dpa_2.z, dpb_2.z));
}


#line 113
__device__ float s_primal_ctx_safe_sqrt_0(float dpa_3)
{

#line 123
    float _S397;
    if(dpa_3 < 1.07549441632776457e-20f)
    {

#line 124
        _S397 = 0.0f;

#line 124
    }
    else
    {

#line 124
        _S397 = s_primal_ctx_sqrt_0(dpa_3);

#line 124
    }

#line 124
    return _S397;
}


#line 124
__device__ float2  s_primal_ctx_safe_eliIntersect_0(float3  dpro_0, float3  dprd_0, float3  dpra_0)
{

#line 133 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
    float3  _S398 = s_primal_ctx_safe_div_2(dpro_0, dpra_0);

#line 133
    float3  _S399 = s_primal_ctx_safe_div_2(dprd_0, dpra_0);

#line 133
    float _S400 = s_primal_ctx_dot_0(_S399, _S399);

#line 141
    float bp_1 = - s_primal_ctx_dot_0(_S398, _S399);
    float3  l_1 = _S398 + make_float3 (s_primal_ctx_safe_div_0(bp_1, _S400)) * _S399;
    float h_1 = _S400 * (1.0f - s_primal_ctx_dot_0(l_1, l_1));
    float c_1 = s_primal_ctx_dot_0(_S398, _S398) - 1.0f;
    bool _S401 = h_1 < 0.0f;

#line 145
    float2  _S402;

#line 145
    if(_S401)
    {

#line 145
        _S402 = make_float2 (-1.0f);

#line 145
    }

#line 145
    bool _S403 = !_S401;

#line 145
    if(_S403)
    {

#line 146
        float _S404 = bp_1 + float((F32_sign((bp_1)))) * s_primal_ctx_safe_sqrt_0(h_1);

#line 146
        _S402 = make_float2 (s_primal_ctx_safe_div_0(c_1, _S404), s_primal_ctx_safe_div_0(_S404, _S400));

#line 146
    }

#line 146
    return _S402;
}


#line 146
__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S405, float _S406)
{

#line 146
    _d_sqrt_0(_S405, _S406);

#line 146
    return;
}


#line 123 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_safe_sqrt_0(DiffPair_float_0 * dpa_4, float _s_dOut_5)
{

#line 123
    DiffPair_float_0 _S407 = *dpa_4;

#line 123
    float _S408;

#line 123
    if(((*dpa_4).primal_0) < 1.07549441632776457e-20f)
    {

#line 123
        _S408 = 0.0f;

#line 123
    }
    else
    {

        DiffPair_float_0 _S409;

#line 127
        (&_S409)->primal_0 = _S407.primal_0;

#line 127
        (&_S409)->differential_0 = 0.0f;

#line 127
        s_bwd_prop_sqrt_0(&_S409, _s_dOut_5);

#line 127
        _S408 = _S409.differential_0;

#line 127
    }

#line 127
    dpa_4->primal_0 = (*dpa_4).primal_0;

#line 127
    dpa_4->differential_0 = _S408;

#line 123
    return;
}


#line 123
__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S410, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S411, float _S412)
{

#line 123
    _d_dot_0(_S410, _S411, _S412);

#line 123
    return;
}


#line 113
__device__ void s_bwd_prop_safe_div_2(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpa_5, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpb_3, float3  _s_dOut_6)
{
    float _S413 = (*dpa_5).primal_0.x;

#line 115
    float _S414 = (*dpb_3).primal_0.x;
    float _S415 = (*dpa_5).primal_0.y;

#line 116
    float _S416 = (*dpb_3).primal_0.y;
    float _S417 = (*dpb_3).primal_0.z;

#line 117
    DiffPair_float_0 _S418;

#line 117
    (&_S418)->primal_0 = (*dpa_5).primal_0.z;

#line 117
    (&_S418)->differential_0 = 0.0f;

#line 117
    DiffPair_float_0 _S419;

#line 117
    (&_S419)->primal_0 = _S417;

#line 117
    (&_S419)->differential_0 = 0.0f;

#line 117
    s_bwd_prop_safe_div_0(&_S418, &_S419, _s_dOut_6.z);

#line 116
    DiffPair_float_0 _S420;

#line 116
    (&_S420)->primal_0 = _S415;

#line 116
    (&_S420)->differential_0 = 0.0f;

#line 116
    DiffPair_float_0 _S421;

#line 116
    (&_S421)->primal_0 = _S416;

#line 116
    (&_S421)->differential_0 = 0.0f;

#line 116
    s_bwd_prop_safe_div_0(&_S420, &_S421, _s_dOut_6.y);

#line 115
    DiffPair_float_0 _S422;

#line 115
    (&_S422)->primal_0 = _S413;

#line 115
    (&_S422)->differential_0 = 0.0f;

#line 115
    DiffPair_float_0 _S423;

#line 115
    (&_S423)->primal_0 = _S414;

#line 115
    (&_S423)->differential_0 = 0.0f;

#line 115
    s_bwd_prop_safe_div_0(&_S422, &_S423, _s_dOut_6.x);

#line 115
    float3  _S424 = make_float3 (_S423.differential_0, _S421.differential_0, _S419.differential_0);

#line 115
    dpb_3->primal_0 = (*dpb_3).primal_0;

#line 115
    dpb_3->differential_0 = _S424;

#line 115
    float3  _S425 = make_float3 (_S422.differential_0, _S420.differential_0, _S418.differential_0);

#line 115
    dpa_5->primal_0 = (*dpa_5).primal_0;

#line 115
    dpa_5->differential_0 = _S425;

#line 113
    return;
}


#line 133 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
__device__ void s_bwd_prop_safe_eliIntersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpro_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprd_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpra_1, float2  _s_dOut_7)
{

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S426 = *dpro_1;

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S427 = *dprd_1;

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S428 = *dpra_1;

#line 133
    float3  _S429 = s_primal_ctx_safe_div_2((*dpro_1).primal_0, (*dpra_1).primal_0);

#line 133
    float3  _S430 = s_primal_ctx_safe_div_2((*dprd_1).primal_0, (*dpra_1).primal_0);

#line 133
    float _S431 = s_primal_ctx_dot_0(_S430, _S430);

#line 141
    float bp_2 = - s_primal_ctx_dot_0(_S429, _S430);

#line 141
    float _S432 = s_primal_ctx_safe_div_0(bp_2, _S431);
    float3  _S433 = make_float3 (_S432);

#line 142
    float3  l_2 = _S429 + make_float3 (_S432) * _S430;
    float _S434 = 1.0f - s_primal_ctx_dot_0(l_2, l_2);

#line 143
    float h_2 = _S431 * _S434;
    float c_2 = s_primal_ctx_dot_0(_S429, _S429) - 1.0f;

#line 144
    bool _S435 = !(h_2 < 0.0f);

#line 144
    float _S436;

#line 144
    float _S437;

#line 144
    if(_S435)
    {
        float _S438 = float((F32_sign((bp_2))));

#line 146
        _S436 = bp_2 + _S438 * s_primal_ctx_safe_sqrt_0(h_2);

#line 146
        _S437 = _S438;

#line 146
    }
    else
    {

#line 146
        _S436 = 0.0f;

#line 146
        _S437 = 0.0f;

#line 146
    }

#line 146
    float _S439;

#line 146
    float _S440;

#line 146
    if(_S435)
    {

#line 147
        DiffPair_float_0 _S441;

#line 147
        (&_S441)->primal_0 = _S436;

#line 147
        (&_S441)->differential_0 = 0.0f;

#line 147
        DiffPair_float_0 _S442;

#line 147
        (&_S442)->primal_0 = _S431;

#line 147
        (&_S442)->differential_0 = 0.0f;

#line 147
        s_bwd_prop_safe_div_0(&_S441, &_S442, _s_dOut_7.y);

#line 147
        DiffPair_float_0 _S443;

#line 147
        (&_S443)->primal_0 = c_2;

#line 147
        (&_S443)->differential_0 = 0.0f;

#line 147
        DiffPair_float_0 _S444;

#line 147
        (&_S444)->primal_0 = _S436;

#line 147
        (&_S444)->differential_0 = 0.0f;

#line 147
        s_bwd_prop_safe_div_0(&_S443, &_S444, _s_dOut_7.x);

#line 146
        float _S445 = _S441.differential_0 + _S444.differential_0;

#line 146
        float _S446 = _S437 * _S445;

#line 146
        DiffPair_float_0 _S447;

#line 146
        (&_S447)->primal_0 = h_2;

#line 146
        (&_S447)->differential_0 = 0.0f;

#line 146
        s_bwd_prop_safe_sqrt_0(&_S447, _S446);

#line 146
        _S436 = _S443.differential_0;

#line 146
        _S437 = _S447.differential_0;

#line 146
        _S439 = _S445;

#line 146
        _S440 = _S442.differential_0;

#line 146
    }
    else
    {

#line 146
        _S436 = 0.0f;

#line 146
        _S437 = 0.0f;

#line 146
        _S439 = 0.0f;

#line 146
        _S440 = 0.0f;

#line 146
    }

#line 144
    float3  _S448 = make_float3 (0.0f);

#line 144
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S449;

#line 144
    (&_S449)->primal_0 = _S429;

#line 144
    (&_S449)->differential_0 = _S448;

#line 144
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S450;

#line 144
    (&_S450)->primal_0 = _S429;

#line 144
    (&_S450)->differential_0 = _S448;

#line 144
    s_bwd_prop_dot_0(&_S449, &_S450, _S436);

#line 143
    float _S451 = _S434 * _S437;

#line 143
    float _S452 = - (_S431 * _S437);

#line 143
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S453;

#line 143
    (&_S453)->primal_0 = l_2;

#line 143
    (&_S453)->differential_0 = _S448;

#line 143
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S454;

#line 143
    (&_S454)->primal_0 = l_2;

#line 143
    (&_S454)->differential_0 = _S448;

#line 143
    s_bwd_prop_dot_0(&_S453, &_S454, _S452);

#line 142
    float3  _S455 = _S454.differential_0 + _S453.differential_0;

#line 142
    float3  _S456 = _S433 * _S455;

#line 142
    float3  _S457 = _S430 * _S455;

#line 142
    float _S458 = _S457.x + _S457.y + _S457.z;

#line 142
    DiffPair_float_0 _S459;

#line 142
    (&_S459)->primal_0 = bp_2;

#line 142
    (&_S459)->differential_0 = 0.0f;

#line 142
    DiffPair_float_0 _S460;

#line 142
    (&_S460)->primal_0 = _S431;

#line 142
    (&_S460)->differential_0 = 0.0f;

#line 142
    s_bwd_prop_safe_div_0(&_S459, &_S460, _S458);

#line 141
    float _S461 = - (_S459.differential_0 + _S439);

#line 141
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S462;

#line 141
    (&_S462)->primal_0 = _S429;

#line 141
    (&_S462)->differential_0 = _S448;

#line 141
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S463;

#line 141
    (&_S463)->primal_0 = _S430;

#line 141
    (&_S463)->differential_0 = _S448;

#line 141
    s_bwd_prop_dot_0(&_S462, &_S463, _S461);

#line 140
    float _S464 = _S451 + _S460.differential_0 + _S440;

#line 140
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S465;

#line 140
    (&_S465)->primal_0 = _S430;

#line 140
    (&_S465)->differential_0 = _S448;

#line 140
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S466;

#line 140
    (&_S466)->primal_0 = _S430;

#line 140
    (&_S466)->differential_0 = _S448;

#line 140
    s_bwd_prop_dot_0(&_S465, &_S466, _S464);

#line 139
    float3  _S467 = _S456 + _S463.differential_0 + _S466.differential_0 + _S465.differential_0;

#line 139
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S468;

#line 139
    (&_S468)->primal_0 = _S427.primal_0;

#line 139
    (&_S468)->differential_0 = _S448;

#line 139
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S469;

#line 139
    (&_S469)->primal_0 = _S428.primal_0;

#line 139
    (&_S469)->differential_0 = _S448;

#line 139
    s_bwd_prop_safe_div_2(&_S468, &_S469, _S467);

#line 138
    float3  _S470 = _S450.differential_0 + _S449.differential_0 + _S455 + _S462.differential_0;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S471;

#line 138
    (&_S471)->primal_0 = _S426.primal_0;

#line 138
    (&_S471)->differential_0 = _S448;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S472;

#line 138
    (&_S472)->primal_0 = _S428.primal_0;

#line 138
    (&_S472)->differential_0 = _S448;

#line 138
    s_bwd_prop_safe_div_2(&_S471, &_S472, _S470);

#line 2246 "core.meta.slang"
    float3  _S473 = _S469.differential_0 + _S472.differential_0;

#line 2246
    dpra_1->primal_0 = (*dpra_1).primal_0;

#line 2246
    dpra_1->differential_0 = _S473;

#line 2246
    dprd_1->primal_0 = (*dprd_1).primal_0;

#line 2246
    dprd_1->differential_0 = _S468.differential_0;

#line 2246
    dpro_1->primal_0 = (*dpro_1).primal_0;

#line 2246
    dpro_1->differential_0 = _S471.differential_0;

#line 133 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
    return;
}


#line 133
__device__ void s_bwd_prop_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S474, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S475, float3  _S476)
{

#line 133
    _d_cross_0(_S474, _S475, _S476);

#line 133
    return;
}


#line 238 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_rotate_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpv_5, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpq_1, float3  _s_dOut_8)
{


    float3  _S477 = - float3 {(*dpq_1).primal_0.y, (*dpq_1).primal_0.z, (*dpq_1).primal_0.w};

#line 242
    float3  _S478 = make_float3 (2.0f) * s_primal_ctx_cross_0(_S477, (*dpv_5).primal_0);
    float3  _S479 = make_float3 ((*dpq_1).primal_0.x);

#line 243
    float3  _S480 = make_float3 (0.0f);

#line 243
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S481;

#line 243
    (&_S481)->primal_0 = _S477;

#line 243
    (&_S481)->differential_0 = _S480;

#line 243
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S482;

#line 243
    (&_S482)->primal_0 = _S478;

#line 243
    (&_S482)->differential_0 = _S480;

#line 243
    s_bwd_prop_cross_0(&_S481, &_S482, _s_dOut_8);

#line 243
    float3  _S483 = _S478 * _s_dOut_8;

#line 243
    float _S484 = _S483.x + _S483.y + _S483.z;

#line 242
    float3  _S485 = make_float3 (2.0f) * (_S482.differential_0 + _S479 * _s_dOut_8);

#line 242
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S486;

#line 242
    (&_S486)->primal_0 = _S477;

#line 242
    (&_S486)->differential_0 = _S480;

#line 242
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S487;

#line 242
    (&_S487)->primal_0 = (*dpv_5).primal_0;

#line 242
    (&_S487)->differential_0 = _S480;

#line 242
    s_bwd_prop_cross_0(&_S486, &_S487, _S485);

#line 242
    float3  _S488 = - (_S481.differential_0 + _S486.differential_0);

#line 242
    float4  _S489 = make_float4 (_S484, _S488.x, _S488.y, _S488.z);

#line 242
    dpq_1->primal_0 = (*dpq_1).primal_0;

#line 242
    dpq_1->differential_0 = _S489;

#line 2246 "core.meta.slang"
    float3  _S490 = _s_dOut_8 + _S487.differential_0;

#line 2246
    dpv_5->primal_0 = (*dpv_5).primal_0;

#line 2246
    dpv_5->differential_0 = _S490;

#line 238 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 137
__device__ void s_bwd_prop_l2_normalize_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_10, float3  _s_dOut_9)
{

#line 137
    float3  _S491 = (*dpx_10).primal_0;

#line 137
    float _S492 = s_primal_ctx_dot_0(_S491, _S491);

#line 137
    float _S493 = s_primal_ctx_max_0(_S492, 1.07549441632776457e-20f);

#line 137
    float _S494 = s_primal_ctx_sqrt_0(_S493);
    float3  _S495 = make_float3 (0.0f);

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S496;

#line 138
    (&_S496)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S496)->differential_0 = _S495;

#line 138
    DiffPair_float_0 _S497;

#line 138
    (&_S497)->primal_0 = _S494;

#line 138
    (&_S497)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_safe_div_1(&_S496, &_S497, _s_dOut_9);

#line 138
    DiffPair_float_0 _S498;

#line 138
    (&_S498)->primal_0 = _S493;

#line 138
    (&_S498)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_sqrt_0(&_S498, _S497.differential_0);

#line 138
    DiffPair_float_0 _S499;

#line 138
    (&_S499)->primal_0 = _S492;

#line 138
    (&_S499)->differential_0 = 0.0f;

#line 138
    DiffPair_float_0 _S500;

#line 138
    (&_S500)->primal_0 = 1.07549441632776457e-20f;

#line 138
    (&_S500)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_max_0(&_S499, &_S500, _S498.differential_0);

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S501;

#line 138
    (&_S501)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S501)->differential_0 = _S495;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S502;

#line 138
    (&_S502)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S502)->differential_0 = _S495;

#line 138
    s_bwd_prop_dot_0(&_S501, &_S502, _S499.differential_0);

#line 2246 "core.meta.slang"
    float3  _S503 = _S496.differential_0 + _S502.differential_0 + _S501.differential_0;

#line 2246
    dpx_10->primal_0 = (*dpx_10).primal_0;

#line 2246
    dpx_10->differential_0 = _S503;

#line 137 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 179 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
__device__ void s_bwd_prop_safe_ray_intersect_ellipsoid_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayo_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayd_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpscales_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquat_0, float2  _s_dOut_10)
{


    float3  _S504 = s_primal_ctx_rotate_vector_0((*dprayd_0).primal_0, (*dpquat_0).primal_0);

#line 183
    float3  _S505 = s_primal_ctx_l2_normalize_0(_S504);

#line 183
    float3  _S506 = s_primal_ctx_rotate_vector_0((*dprayo_0).primal_0, (*dpquat_0).primal_0);

#line 183
    float2  _S507 = s_primal_ctx_safe_eliIntersect_0(_S506, _S505, (*dpscales_0).primal_0);

#line 189
    float _S508 = _S507.x;

#line 189
    float _S509 = _S507.y;

#line 189
    DiffPair_float_0 _S510;

#line 189
    (&_S510)->primal_0 = _S508;

#line 189
    (&_S510)->differential_0 = 0.0f;

#line 189
    DiffPair_float_0 _S511;

#line 189
    (&_S511)->primal_0 = _S509;

#line 189
    (&_S511)->differential_0 = 0.0f;

#line 189
    s_bwd_prop_max_0(&_S510, &_S511, _s_dOut_10.y);

#line 189
    DiffPair_float_0 _S512;

#line 189
    (&_S512)->primal_0 = _S508;

#line 189
    (&_S512)->differential_0 = 0.0f;

#line 189
    DiffPair_float_0 _S513;

#line 189
    (&_S513)->primal_0 = _S509;

#line 189
    (&_S513)->differential_0 = 0.0f;

#line 189
    s_bwd_prop_min_0(&_S512, &_S513, _s_dOut_10.x);

#line 188
    float2  _S514 = make_float2 (_S510.differential_0 + _S512.differential_0, _S511.differential_0 + _S513.differential_0);

#line 188
    float3  _S515 = make_float3 (0.0f);

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S516;

#line 188
    (&_S516)->primal_0 = _S506;

#line 188
    (&_S516)->differential_0 = _S515;

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S517;

#line 188
    (&_S517)->primal_0 = _S505;

#line 188
    (&_S517)->differential_0 = _S515;

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S518;

#line 188
    (&_S518)->primal_0 = (*dpscales_0).primal_0;

#line 188
    (&_S518)->differential_0 = _S515;

#line 188
    s_bwd_prop_safe_eliIntersect_0(&_S516, &_S517, &_S518, _S514);

#line 186
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S519;

#line 186
    (&_S519)->primal_0 = (*dprayo_0).primal_0;

#line 186
    (&_S519)->differential_0 = _S515;

#line 186
    float4  _S520 = make_float4 (0.0f);

#line 186
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S521;

#line 186
    (&_S521)->primal_0 = (*dpquat_0).primal_0;

#line 186
    (&_S521)->differential_0 = _S520;

#line 186
    s_bwd_prop_rotate_vector_0(&_S519, &_S521, _S516.differential_0);

#line 185
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S522;

#line 185
    (&_S522)->primal_0 = _S504;

#line 185
    (&_S522)->differential_0 = _S515;

#line 185
    s_bwd_prop_l2_normalize_0(&_S522, _S517.differential_0);

#line 185
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S523;

#line 185
    (&_S523)->primal_0 = (*dprayd_0).primal_0;

#line 185
    (&_S523)->differential_0 = _S515;

#line 185
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S524;

#line 185
    (&_S524)->primal_0 = (*dpquat_0).primal_0;

#line 185
    (&_S524)->differential_0 = _S520;

#line 185
    s_bwd_prop_rotate_vector_0(&_S523, &_S524, _S522.differential_0);

#line 2246 "core.meta.slang"
    float4  _S525 = _S521.differential_0 + _S524.differential_0;

#line 2246
    dpquat_0->primal_0 = (*dpquat_0).primal_0;

#line 2246
    dpquat_0->differential_0 = _S525;

#line 2246
    dpscales_0->primal_0 = (*dpscales_0).primal_0;

#line 2246
    dpscales_0->differential_0 = _S518.differential_0;

#line 2246
    dprayd_0->primal_0 = (*dprayd_0).primal_0;

#line 2246
    dprayd_0->differential_0 = _S523.differential_0;

#line 2246
    dprayo_0->primal_0 = (*dprayo_0).primal_0;

#line 2246
    dprayo_0->differential_0 = _S519.differential_0;

#line 179 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
    return;
}


#line 290
__device__ void s_bwd_prop_safe_intersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayo_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayd_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpscales_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpmean_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquat_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcolor_0, DiffPair_float_0 * dpdensity_0, uint face_id_1, bool skip_close_2, ControlPoint_0 * s_diff_out_T_0)
{

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S526 = *dprayd_1;

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S527 = *dpscales_1;

#line 291
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S528 = *dpquat_1;

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S529 = *dpcolor_0;

#line 291
    DiffPair_float_0 _S530 = *dpdensity_0;

    float3  _S531 = (*dprayo_1).primal_0 - (*dpmean_0).primal_0;

    bool _S532 = face_id_1 == 1U;

#line 295
    float dirac_multi_1;
    if(_S532)
    {

#line 296
        dirac_multi_1 = _S530.primal_0;

#line 296
    }
    else
    {

#line 296
        dirac_multi_1 = - _S530.primal_0;

#line 296
    }


    float _S533 = _S529.primal_0.x;

#line 299
    float _S534 = _S529.primal_0.y;

#line 299
    float _S535 = _S529.primal_0.z;

#line 296
    float _S536 = *&((&s_diff_out_T_0->dirac_0)->x) + _S535 * *&((&s_diff_out_T_0->dirac_0)->w) + _S534 * *&((&s_diff_out_T_0->dirac_0)->z) + _S533 * *&((&s_diff_out_T_0->dirac_0)->y);

#line 296
    float3  _S537 = make_float3 (dirac_multi_1 * *&((&s_diff_out_T_0->dirac_0)->y), dirac_multi_1 * *&((&s_diff_out_T_0->dirac_0)->z), dirac_multi_1 * *&((&s_diff_out_T_0->dirac_0)->w));

#line 296
    if(_S532)
    {

#line 296
        dirac_multi_1 = _S536;

#line 296
    }
    else
    {

#line 296
        dirac_multi_1 = - _S536;

#line 296
    }

#line 296
    float2  _S538;

#line 296
    if(_S532)
    {

#line 296
        _S538 = make_float2 (s_diff_out_T_0->t_0, 0.0f);

#line 296
    }
    else
    {

#line 296
        _S538 = make_float2 (0.0f, s_diff_out_T_0->t_0);

#line 296
    }

#line 293
    float3  _S539 = make_float3 (0.0f);

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S540;

#line 293
    (&_S540)->primal_0 = _S531;

#line 293
    (&_S540)->differential_0 = _S539;

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S541;

#line 293
    (&_S541)->primal_0 = _S526.primal_0;

#line 293
    (&_S541)->differential_0 = _S539;

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S542;

#line 293
    (&_S542)->primal_0 = _S527.primal_0;

#line 293
    (&_S542)->differential_0 = _S539;

#line 293
    float4  _S543 = make_float4 (0.0f);

#line 293
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S544;

#line 293
    (&_S544)->primal_0 = _S528.primal_0;

#line 293
    (&_S544)->differential_0 = _S543;

#line 293
    s_bwd_prop_safe_ray_intersect_ellipsoid_0(&_S540, &_S541, &_S542, &_S544, _S538);

#line 293
    float3  _S545 = - _S540.differential_0;

#line 293
    dpdensity_0->primal_0 = (*dpdensity_0).primal_0;

#line 293
    dpdensity_0->differential_0 = dirac_multi_1;

#line 293
    dpcolor_0->primal_0 = (*dpcolor_0).primal_0;

#line 293
    dpcolor_0->differential_0 = _S537;

#line 293
    dpquat_1->primal_0 = (*dpquat_1).primal_0;

#line 293
    dpquat_1->differential_0 = _S544.differential_0;

#line 293
    dpmean_0->primal_0 = (*dpmean_0).primal_0;

#line 293
    dpmean_0->differential_0 = _S545;

#line 293
    dpscales_1->primal_0 = (*dpscales_1).primal_0;

#line 293
    dpscales_1->differential_0 = _S542.differential_0;

#line 293
    dprayd_1->primal_0 = (*dprayd_1).primal_0;

#line 293
    dprayd_1->differential_0 = _S541.differential_0;

#line 293
    dprayo_1->primal_0 = (*dprayo_1).primal_0;

#line 293
    dprayo_1->differential_0 = _S540.differential_0;

#line 290
    return;
}


#line 290
__device__ void s_bwd_safe_intersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S546, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S547, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S548, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S549, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S550, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S551, DiffPair_float_0 * _S552, uint _S553, bool _S554, ControlPoint_0 * _S555)
{

#line 291
    s_bwd_prop_safe_intersect_0(_S546, _S547, _S548, _S549, _S550, _S551, _S552, _S553, _S554, _S555);

#line 291
    return;
}


#line 291
struct DiffPair_Features_0
{
    Features_0 primal_0;
    Features_0 differential_0;
};


#line 75 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
__device__ void s_bwd_prop_eval_sh_col3_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_0, DiffPair_Features_0 * dpfeat_0, float3  _s_dOut_11)
{

#line 76
    float x_6 = (*dpdir_0).primal_0.x;
    float y_4 = (*dpdir_0).primal_0.y;
    float z_2 = (*dpdir_0).primal_0.z;
    float xx_2 = x_6 * x_6;

#line 79
    float yy_2 = y_4 * y_4;

#line 79
    float zz_1 = z_2 * z_2;

    float _S556 = -0.59004360437393188f * y_4;

#line 81
    float _S557 = 3.0f * xx_2;

#line 81
    float _S558 = _S557 - yy_2;
    float _S559 = 2.89061141014099121f * (x_6 * y_4);
    float _S560 = -0.4570457935333252f * y_4;

#line 83
    float _S561 = 4.0f * zz_1 - xx_2 - yy_2;
    float _S562 = 0.37317633628845215f * z_2;

#line 84
    float _S563 = 3.0f * yy_2;

#line 84
    float _S564 = 2.0f * zz_1 - _S557 - _S563;
    float _S565 = -0.4570457935333252f * x_6;
    float _S566 = 1.44530570507049561f * z_2;

#line 86
    float _S567 = xx_2 - yy_2;
    float _S568 = -0.59004360437393188f * x_6;

#line 87
    float _S569 = xx_2 - _S563;

#line 87
    float3  _S570 = make_float3 (_S568 * _S569) * _s_dOut_11;

#line 87
    float3  _S571 = (&dpfeat_0->primal_0)->f15_0 * _s_dOut_11;

#line 87
    float _S572 = _S571.x + _S571.y + _S571.z;

#line 87
    float _S573 = _S568 * _S572;

#line 86
    float3  _S574 = make_float3 (_S566 * _S567) * _s_dOut_11;

#line 86
    float3  _S575 = (&dpfeat_0->primal_0)->f14_0 * _s_dOut_11;

#line 86
    float _S576 = _S575.x + _S575.y + _S575.z;

#line 86
    float _S577 = _S566 * _S576;

#line 85
    float3  _S578 = make_float3 (_S565 * _S561) * _s_dOut_11;

#line 85
    float3  _S579 = (&dpfeat_0->primal_0)->f13_0 * _s_dOut_11;

#line 85
    float _S580 = _S579.x + _S579.y + _S579.z;

#line 84
    float3  _S581 = make_float3 (_S562 * _S564) * _s_dOut_11;

#line 84
    float3  _S582 = (&dpfeat_0->primal_0)->f12_0 * _s_dOut_11;

#line 84
    float _S583 = _S582.x + _S582.y + _S582.z;

#line 84
    float _S584 = _S562 * _S583;

#line 84
    float _S585 = - _S584;

#line 83
    float3  _S586 = make_float3 (_S560 * _S561) * _s_dOut_11;

#line 83
    float3  _S587 = (&dpfeat_0->primal_0)->f11_0 * _s_dOut_11;

#line 83
    float _S588 = _S587.x + _S587.y + _S587.z;

#line 83
    float _S589 = _S565 * _S580 + _S560 * _S588;

#line 83
    float _S590 = - _S589;

#line 82
    float3  _S591 = make_float3 (_S559 * z_2) * _s_dOut_11;

#line 82
    float3  _S592 = (&dpfeat_0->primal_0)->f10_0 * _s_dOut_11;

#line 82
    float _S593 = _S592.x + _S592.y + _S592.z;

#line 82
    float s_diff_xy_T_0 = 2.89061141014099121f * (z_2 * _S593);

#line 81
    float3  _S594 = make_float3 (_S556 * _S558) * _s_dOut_11;

#line 81
    float3  _S595 = (&dpfeat_0->primal_0)->f9_0 * _s_dOut_11;

#line 81
    float _S596 = _S595.x + _S595.y + _S595.z;

#line 81
    float _S597 = _S556 * _S596;

#line 79
    float _S598 = z_2 * (2.0f * _S584 + 4.0f * _S589);

#line 79
    float _S599 = y_4 * (- _S577 + 3.0f * (- _S573 + _S585) + _S590 + - _S597);

#line 79
    float _S600 = x_6 * (_S573 + _S577 + _S590 + 3.0f * (_S585 + _S597));

#line 78
    float _S601 = 1.44530570507049561f * (_S567 * _S576) + 0.37317633628845215f * (_S564 * _S583) + _S559 * _S593 + _S598 + _S598;

#line 77
    float _S602 = -0.4570457935333252f * (_S561 * _S588) + -0.59004360437393188f * (_S558 * _S596) + x_6 * s_diff_xy_T_0 + _S599 + _S599;

#line 76
    float _S603 = -0.59004360437393188f * (_S569 * _S572) + -0.4570457935333252f * (_S561 * _S580) + y_4 * s_diff_xy_T_0 + _S600 + _S600;

#line 76
    Features_0 _S604 = Features_x24_syn_dzero_0();

#line 76
    (&_S604)->f15_0 = _S570;

#line 76
    (&_S604)->f14_0 = _S574;

#line 76
    (&_S604)->f13_0 = _S578;

#line 76
    (&_S604)->f12_0 = _S581;

#line 76
    (&_S604)->f11_0 = _S586;

#line 76
    (&_S604)->f10_0 = _S591;

#line 76
    (&_S604)->f9_0 = _S594;

#line 76
    dpfeat_0->primal_0 = dpfeat_0->primal_0;

#line 76
    dpfeat_0->differential_0 = _S604;

#line 76
    float3  _S605 = make_float3 (_S603, _S602, _S601);

#line 76
    dpdir_0->primal_0 = (*dpdir_0).primal_0;

#line 76
    dpdir_0->differential_0 = _S605;

#line 75
    return;
}


#line 63
__device__ void s_bwd_prop_eval_sh_col2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_1, DiffPair_Features_0 * dpfeat_1, float3  _s_dOut_12)
{

#line 64
    float x_7 = (*dpdir_1).primal_0.x;
    float y_5 = (*dpdir_1).primal_0.y;
    float z_3 = (*dpdir_1).primal_0.z;
    float xx_3 = x_7 * x_7;

#line 67
    float yy_3 = y_5 * y_5;



    float3  _S606 = make_float3 (0.54627424478530884f * (xx_3 - yy_3)) * _s_dOut_12;

#line 71
    float3  _S607 = (&dpfeat_1->primal_0)->f8_0 * _s_dOut_12;

#line 71
    float _S608 = 0.54627424478530884f * (_S607.x + _S607.y + _S607.z);

#line 71
    float3  _S609 = make_float3 (-1.09254848957061768f * (x_7 * z_3)) * _s_dOut_12;

#line 71
    float3  _S610 = (&dpfeat_1->primal_0)->f7_0 * _s_dOut_12;

#line 71
    float s_diff_xz_T_0 = -1.09254848957061768f * (_S610.x + _S610.y + _S610.z);

#line 70
    float3  _S611 = make_float3 (0.31539157032966614f * (2.0f * (z_3 * z_3) - xx_3 - yy_3)) * _s_dOut_12;

#line 70
    float3  _S612 = (&dpfeat_1->primal_0)->f6_0 * _s_dOut_12;

#line 70
    float _S613 = 0.31539157032966614f * (_S612.x + _S612.y + _S612.z);

#line 70
    float _S614 = - _S613;

#line 69
    float3  _S615 = make_float3 (-1.09254848957061768f * (y_5 * z_3)) * _s_dOut_12;

#line 69
    float3  _S616 = (&dpfeat_1->primal_0)->f5_0 * _s_dOut_12;

#line 69
    float s_diff_yz_T_0 = -1.09254848957061768f * (_S616.x + _S616.y + _S616.z);

#line 69
    float3  _S617 = make_float3 (1.09254848957061768f * (x_7 * y_5)) * _s_dOut_12;

#line 69
    float3  _S618 = (&dpfeat_1->primal_0)->f4_0 * _s_dOut_12;

#line 69
    float s_diff_xy_T_1 = 1.09254848957061768f * (_S618.x + _S618.y + _S618.z);

#line 67
    float _S619 = z_3 * (2.0f * _S613);

#line 67
    float _S620 = y_5 * (- _S608 + _S614);

#line 67
    float _S621 = x_7 * (_S608 + _S614);

#line 66
    float _S622 = x_7 * s_diff_xz_T_0 + y_5 * s_diff_yz_T_0 + _S619 + _S619;

#line 65
    float _S623 = z_3 * s_diff_yz_T_0 + x_7 * s_diff_xy_T_1 + _S620 + _S620;

#line 64
    float _S624 = z_3 * s_diff_xz_T_0 + y_5 * s_diff_xy_T_1 + _S621 + _S621;

#line 64
    Features_0 _S625 = Features_x24_syn_dzero_0();

#line 64
    (&_S625)->f8_0 = _S606;

#line 64
    (&_S625)->f7_0 = _S609;

#line 64
    (&_S625)->f6_0 = _S611;

#line 64
    (&_S625)->f5_0 = _S615;

#line 64
    (&_S625)->f4_0 = _S617;

#line 64
    dpfeat_1->primal_0 = dpfeat_1->primal_0;

#line 64
    dpfeat_1->differential_0 = _S625;

#line 64
    float3  _S626 = make_float3 (_S624, _S623, _S622);

#line 64
    dpdir_1->primal_0 = (*dpdir_1).primal_0;

#line 64
    dpdir_1->differential_0 = _S626;

#line 63
    return;
}


#line 54
__device__ void s_bwd_prop_eval_sh_col1_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_2, DiffPair_Features_0 * dpfeat_2, float3  s_diff_color_T_0)
{


    float3  _S627 = - s_diff_color_T_0;

#line 58
    float3  _S628 = make_float3 (0.48860251903533936f * (*dpdir_2).primal_0.x) * _S627;

#line 58
    float3  _S629 = (&dpfeat_2->primal_0)->f3_0 * _S627;

#line 58
    float s_diff_x_T_0 = 0.48860251903533936f * (_S629.x + _S629.y + _S629.z);

#line 58
    float3  _S630 = make_float3 (0.48860251903533936f * (*dpdir_2).primal_0.z) * s_diff_color_T_0;

#line 58
    float3  _S631 = (&dpfeat_2->primal_0)->f2_0 * s_diff_color_T_0;

#line 58
    float s_diff_z_T_0 = 0.48860251903533936f * (_S631.x + _S631.y + _S631.z);

#line 58
    float3  _S632 = make_float3 (-0.48860251903533936f * (*dpdir_2).primal_0.y) * s_diff_color_T_0;

#line 58
    float3  _S633 = (&dpfeat_2->primal_0)->f1_0 * s_diff_color_T_0;

#line 58
    float s_diff_y_T_0 = -0.48860251903533936f * (_S633.x + _S633.y + _S633.z);

#line 58
    Features_0 _S634 = Features_x24_syn_dzero_0();

#line 58
    (&_S634)->f3_0 = _S628;

#line 58
    (&_S634)->f2_0 = _S630;

#line 58
    (&_S634)->f1_0 = _S632;

#line 58
    dpfeat_2->primal_0 = dpfeat_2->primal_0;

#line 58
    dpfeat_2->differential_0 = _S634;

#line 58
    float3  _S635 = make_float3 (s_diff_x_T_0, s_diff_y_T_0, s_diff_z_T_0);

#line 58
    dpdir_2->primal_0 = (*dpdir_2).primal_0;

#line 58
    dpdir_2->differential_0 = _S635;

#line 54
    return;
}


#line 49
__device__ void s_bwd_prop_eval_sh_col0_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_3, DiffPair_Features_0 * dpfeat_3, float3  _s_dOut_13)
{

#line 50
    float3  _S636 = make_float3 (0.282094806432724f) * _s_dOut_13;

#line 50
    Features_0 _S637 = Features_x24_syn_dzero_0();

#line 50
    (&_S637)->f0_0 = _S636;

#line 50
    dpfeat_3->primal_0 = dpfeat_3->primal_0;

#line 50
    dpfeat_3->differential_0 = _S637;

#line 2239 "core.meta.slang"
    float3  _S638 = make_float3 (0.0f);

#line 2239
    dpdir_3->primal_0 = (*dpdir_3).primal_0;

#line 2239
    dpdir_3->differential_0 = _S638;

#line 49 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
    return;
}


#line 91
__device__ void s_bwd_prop_eval_color_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_4, DiffPair_Features_0 * dpfeat_4, uint sh_degree_3, float3  _s_dOut_14)
{

#line 91
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S639 = *dpdir_4;

#line 91
    Features_0 _S640 = dpfeat_4->primal_0;

    bool _S641 = sh_degree_3 > 0U;

#line 93
    bool _S642;

#line 93
    bool _S643;

#line 93
    if(_S641)
    {
        bool _S644 = sh_degree_3 > 1U;

#line 95
        if(_S644)
        {

#line 95
            _S642 = sh_degree_3 > 2U;

#line 95
        }
        else
        {

#line 95
            _S642 = false;

#line 95
        }

        bool _S645 = _S642;

#line 97
        _S642 = _S644;

#line 97
        _S643 = _S645;

#line 97
    }
    else
    {

#line 97
        _S642 = false;

#line 97
        _S643 = false;

#line 97
    }

#line 2239 "core.meta.slang"
    float3  _S646 = make_float3 (0.0f);

#line 2239
    Features_0 _S647 = Features_x24_syn_dzero_0();

#line 2239
    Features_0 _S648;

#line 2239
    float3  _S649;

#line 2239
    if(_S641)
    {

#line 2239
        if(_S642)
        {

#line 2239
            if(_S643)
            {

#line 98 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S650;

#line 98
                (&_S650)->primal_0 = _S639.primal_0;

#line 98
                (&_S650)->differential_0 = _S646;

#line 98
                DiffPair_Features_0 _S651;

#line 98
                (&_S651)->primal_0 = _S640;

#line 98
                (&_S651)->differential_0 = _S647;

#line 98
                s_bwd_prop_eval_sh_col3_0(&_S650, &_S651, _s_dOut_14);

#line 98
                Features_0 _S652 = (&_S651)->differential_0;

#line 98
                Features_0 _S653 = _S647;

#line 98
                Features_0 _S654 = Features_x24_syn_dadd_0(&_S652, &_S653);

#line 98
                _S648 = _S654;

#line 98
                _S649 = _S650.differential_0;

#line 98
            }
            else
            {

#line 98
                _S648 = _S647;

#line 98
                _S649 = _S646;

#line 98
            }

#line 96
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S655;

#line 96
            (&_S655)->primal_0 = _S639.primal_0;

#line 96
            (&_S655)->differential_0 = _S646;

#line 96
            DiffPair_Features_0 _S656;

#line 96
            (&_S656)->primal_0 = _S640;

#line 96
            (&_S656)->differential_0 = _S647;

#line 96
            s_bwd_prop_eval_sh_col2_0(&_S655, &_S656, _s_dOut_14);

#line 96
            Features_0 _S657 = (&_S656)->differential_0;

#line 96
            Features_0 _S658 = _S648;

#line 96
            Features_0 _S659 = Features_x24_syn_dadd_0(&_S657, &_S658);

#line 2246 "core.meta.slang"
            float3  _S660 = _S655.differential_0 + _S649;

#line 2246
            _S648 = _S659;

#line 2246
            _S649 = _S660;

#line 2246
        }
        else
        {

#line 2246
            _S648 = _S647;

#line 2246
            _S649 = _S646;

#line 2246
        }

#line 94 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S661;

#line 94
        (&_S661)->primal_0 = _S639.primal_0;

#line 94
        (&_S661)->differential_0 = _S646;

#line 94
        DiffPair_Features_0 _S662;

#line 94
        (&_S662)->primal_0 = _S640;

#line 94
        (&_S662)->differential_0 = _S647;

#line 94
        s_bwd_prop_eval_sh_col1_0(&_S661, &_S662, _s_dOut_14);

#line 94
        Features_0 _S663 = (&_S662)->differential_0;

#line 94
        Features_0 _S664 = _S648;

#line 94
        Features_0 _S665 = Features_x24_syn_dadd_0(&_S663, &_S664);

#line 2246 "core.meta.slang"
        float3  _S666 = _S661.differential_0 + _S649;

#line 2246
        _S648 = _S665;

#line 2246
        _S649 = _S666;

#line 2246
    }
    else
    {

#line 2246
        _S648 = _S647;

#line 2246
        _S649 = _S646;

#line 2246
    }

#line 92 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S667;

#line 92
    (&_S667)->primal_0 = _S639.primal_0;

#line 92
    (&_S667)->differential_0 = _S646;

#line 92
    DiffPair_Features_0 _S668;

#line 92
    (&_S668)->primal_0 = _S640;

#line 92
    (&_S668)->differential_0 = _S647;

#line 92
    s_bwd_prop_eval_sh_col0_0(&_S667, &_S668, _s_dOut_14);

#line 92
    Features_0 _S669 = (&_S668)->differential_0;

#line 92
    Features_0 _S670 = _S648;

#line 92
    Features_0 _S671 = Features_x24_syn_dadd_0(&_S669, &_S670);

#line 92
    dpfeat_4->primal_0 = dpfeat_4->primal_0;

#line 92
    dpfeat_4->differential_0 = _S671;

#line 2246 "core.meta.slang"
    float3  _S672 = _S667.differential_0 + _S649;

#line 2246
    dpdir_4->primal_0 = (*dpdir_4).primal_0;

#line 2246
    dpdir_4->differential_0 = _S672;

#line 91 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/sh.slang"
    return;
}


#line 91
__device__ void s_bwd_eval_color_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S673, DiffPair_Features_0 * _S674, uint _S675, float3  _S676)
{

#line 91
    s_bwd_prop_eval_color_0(_S673, _S674, _S675, _S676);

#line 91
    return;
}


#line 91
struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
    float2  primal_0;
    float2  differential_0;
};


#line 218 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_mul_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * _S677, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S678, float4  _S679)
{

#line 218
    _d_mul_0(_S677, _S678, _S679);

#line 218
    return;
}


#line 252
__device__ void s_bwd_prop_inv_project_0(DiffPair_vectorx3Cfloatx2C2x3E_0 * dpxy_0, DiffPair_float_0 * dpdist_0, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dpinv_wvt_0, float3  _s_dOut_15)
{

#line 253
    float2  _S680 = make_float2 ((*dpdist_0).primal_0);
    float4  _S681 = make_float4 (_s_dOut_15.x, _s_dOut_15.y, _s_dOut_15.z, 0.0f);

#line 254
    float4  _S682 = make_float4 (0.0f);

#line 254
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S683;

#line 254
    (&_S683)->primal_0 = make_float4 (((*dpxy_0).primal_0 * make_float2 ((*dpdist_0).primal_0)).x, ((*dpxy_0).primal_0 * make_float2 ((*dpdist_0).primal_0)).y, (*dpdist_0).primal_0, 1.0f);

#line 254
    (&_S683)->differential_0 = _S682;

#line 254
    Matrix<float, 4, 4>  _S684 = makeMatrix<float, 4, 4> (0.0f);

#line 254
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S685;

#line 254
    (&_S685)->primal_0 = (*dpinv_wvt_0).primal_0;

#line 254
    (&_S685)->differential_0 = _S684;

#line 254
    s_bwd_prop_mul_0(&_S683, &_S685, _S681);

#line 253
    float2  _S686 = float2 {_S683.differential_0.x, _S683.differential_0.y};

#line 253
    float2  _S687 = (*dpxy_0).primal_0 * _S686;

#line 253
    float2  _S688 = _S680 * _S686;

#line 253
    dpinv_wvt_0->primal_0 = (*dpinv_wvt_0).primal_0;

#line 253
    dpinv_wvt_0->differential_0 = _S685.differential_0;

#line 1230 "core.meta.slang"
    float _S689 = _S683.differential_0.z + _S687.x + _S687.y;

#line 1230
    dpdist_0->primal_0 = (*dpdist_0).primal_0;

#line 1230
    dpdist_0->differential_0 = _S689;

#line 1230
    dpxy_0->primal_0 = (*dpxy_0).primal_0;

#line 1230
    dpxy_0->differential_0 = _S688;

#line 252 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
    return;
}


#line 252
__device__ void s_bwd_inv_project_0(DiffPair_vectorx3Cfloatx2C2x3E_0 * _S690, DiffPair_float_0 * _S691, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S692, float3  _S693)
{

#line 252
    s_bwd_prop_inv_project_0(_S690, _S691, _S692, _S693);

#line 252
    return;
}


#line 132
__device__ DiffPair_SplineState_0 run_update_0(SplineState_0 * old_dual_state_0, ControlPoint_0 * old_ctrl_pt_0, ControlPoint_0 * ctrl_pt_2, uint prim_ind_1, uint face_id_2, uint ray_ind_0, DiffPair_SplineState_0 * deriv_state_0, float3  origin_1, float3  direction_1, float tmin_0, float tmax_0, uint sh_degree_4, float max_prim_size_2, Matrix<float, 4, 4>  wct_1, Matrix<float, 4, 4>  inv_wct_0, DualModel_0 * model_1)
{

#line 147
    SplineState_0 _S694 = from_dual_0(old_dual_state_0, old_ctrl_pt_0);

#line 22 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    float2  _S695 = make_float2 (0.0f);

    float3  _S696 = make_float3 (0.0f);


    float4  _S697 = make_float4 (0.0f);

#line 149 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
    SplineState_0 _S698 = SplineState_x24init_0(_S695, _S695, _S696, 0.0f, _S697, 0.0f, _S696);

#line 149
    DiffPair_SplineState_0 old_deriv_state_0;

#line 149
    (&old_deriv_state_0)->primal_0 = _S694;

#line 149
    (&old_deriv_state_0)->differential_0 = _S698;
    ControlPoint_0 _S699 = ControlPoint_x24init_0(0.0f, _S697);

#line 150
    DiffPair_ControlPoint_0 deriv_ctrl_pt_0;

#line 150
    (&deriv_ctrl_pt_0)->primal_0 = *ctrl_pt_2;

#line 150
    (&deriv_ctrl_pt_0)->differential_0 = _S699;

#line 150
    s_bwd_update_0(&old_deriv_state_0, &deriv_ctrl_pt_0, tmin_0, tmax_0, max_prim_size_2, &deriv_state_0->differential_0);

#line 156
    float3  _S700 = get_float3_0(model_1->means_0, prim_ind_1);
    float3  _S701 = get_float3_0(model_1->scales_2, prim_ind_1);
    float4  _S702 = get_float4_0(model_1->quats_0, prim_ind_1);
    float _S703 = ((model_1->densities_0).load<float>((prim_ind_1)));
    Features_0 feat_6 = get_feats_0(model_1->features_1, prim_ind_1, sh_degree_4);

#line 160
    Features_0 _S704 = feat_6;

#line 160
    float3  _S705 = eval_color_0(direction_1, &_S704, sh_degree_4);


    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_origin_0;

#line 163
    (&deriv_origin_0)->primal_0 = origin_1;

#line 163
    (&deriv_origin_0)->differential_0 = _S696;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_direction_0;

#line 164
    (&deriv_direction_0)->primal_0 = direction_1;

#line 164
    (&deriv_direction_0)->differential_0 = _S696;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_scales_0;

#line 165
    (&deriv_scales_0)->primal_0 = _S701;

#line 165
    (&deriv_scales_0)->differential_0 = _S696;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_mean_0;

#line 166
    (&deriv_mean_0)->primal_0 = _S700;

#line 166
    (&deriv_mean_0)->differential_0 = _S696;
    DiffPair_vectorx3Cfloatx2C4x3E_0 deriv_quat_0;

#line 167
    (&deriv_quat_0)->primal_0 = _S702;

#line 167
    (&deriv_quat_0)->differential_0 = _S697;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_color_0;

#line 168
    (&deriv_color_0)->primal_0 = _S705;

#line 168
    (&deriv_color_0)->differential_0 = _S696;
    DiffPair_float_0 deriv_density_0;

#line 169
    (&deriv_density_0)->primal_0 = _S703;

#line 169
    (&deriv_density_0)->differential_0 = 0.0f;

#line 169
    ControlPoint_0 _S706 = deriv_ctrl_pt_0.differential_0;

#line 169
    s_bwd_safe_intersect_0(&deriv_origin_0, &deriv_direction_0, &deriv_scales_0, &deriv_mean_0, &deriv_quat_0, &deriv_color_0, &deriv_density_0, face_id_2, false, &_S706);

#line 174
    atomic_add_float3_0(model_1->dL_dmeans_0, prim_ind_1, deriv_mean_0.differential_0);
    atomic_add_float3_0(model_1->dL_dscales_0, prim_ind_1, deriv_scales_0.differential_0);
    atomic_add_float4_0(model_1->dL_dquats_0, prim_ind_1, deriv_quat_0.differential_0);
    float temp_4;
    *((&temp_4)) = atomicAdd((model_1->dL_ddensities_0).data_ptr_at<float>((prim_ind_1)), (deriv_density_0.differential_0));

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S707 = deriv_direction_0;

#line 180
    (&deriv_direction_0)->primal_0 = direction_1;

#line 180
    (&deriv_direction_0)->differential_0 = _S696;

    Features_0 _S708 = Features_x24init_0(_S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696, _S696);

#line 182
    DiffPair_Features_0 d_feat_0;

#line 182
    (&d_feat_0)->primal_0 = feat_6;

#line 182
    (&d_feat_0)->differential_0 = _S708;
    s_bwd_eval_color_0(&deriv_direction_0, &d_feat_0, sh_degree_4, deriv_color_0.differential_0);
    float3  d_rayd_0 = _S707.differential_0 + deriv_direction_0.differential_0;

    atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 0U, (&(&d_feat_0)->differential_0)->f0_0);
    if(sh_degree_4 > 0U)
    {

#line 188
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 1U, (&(&d_feat_0)->differential_0)->f1_0);
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 2U, (&(&d_feat_0)->differential_0)->f2_0);
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 3U, (&(&d_feat_0)->differential_0)->f3_0);
        if(sh_degree_4 > 1U)
        {

#line 192
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 4U, (&(&d_feat_0)->differential_0)->f4_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 5U, (&(&d_feat_0)->differential_0)->f5_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 6U, (&(&d_feat_0)->differential_0)->f6_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 7U, (&(&d_feat_0)->differential_0)->f7_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 8U, (&(&d_feat_0)->differential_0)->f8_0);
            if(sh_degree_4 > 2U)
            {

#line 198
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 9U, (&(&d_feat_0)->differential_0)->f9_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 10U, (&(&d_feat_0)->differential_0)->f10_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 11U, (&(&d_feat_0)->differential_0)->f11_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 12U, (&(&d_feat_0)->differential_0)->f12_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 13U, (&(&d_feat_0)->differential_0)->f13_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 14U, (&(&d_feat_0)->differential_0)->f14_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 15U, (&(&d_feat_0)->differential_0)->f15_0);

#line 197
            }

#line 191
        }

#line 187
    }

#line 209
    atomic_add_float3_0(model_1->dL_drayos_0, ray_ind_0, deriv_origin_0.differential_0);
    atomic_add_float3_0(model_1->dL_drayds_0, ray_ind_0, d_rayd_0);

    float3  xyd_0 = project_0(_S700, wct_1);

    DiffPair_vectorx3Cfloatx2C2x3E_0 d_xy_0;

#line 214
    (&d_xy_0)->primal_0 = make_float2 (xyd_0.x, xyd_0.y);

#line 214
    (&d_xy_0)->differential_0 = _S695;
    DiffPair_float_0 d_dist_0;

#line 215
    (&d_dist_0)->primal_0 = xyd_0.z;

#line 215
    (&d_dist_0)->differential_0 = 0.0f;
    Matrix<float, 4, 4>  _S709 = makeMatrix<float, 4, 4> (0.0f);

#line 216
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 d_inv_wct_0;

#line 216
    (&d_inv_wct_0)->primal_0 = inv_wct_0;

#line 216
    (&d_inv_wct_0)->differential_0 = _S709;

    s_bwd_inv_project_0(&d_xy_0, &d_dist_0, &d_inv_wct_0, deriv_mean_0.differential_0);
    atomic_add_float2_0(model_1->dL_dmeans2D_0, prim_ind_1, d_xy_0.differential_0);
    return old_deriv_state_0;
}


#line 175 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
struct SplineOutput_0
{
    float3  C_2;
    float depth_0;
    float distortion_loss_0;
};


#line 175
__device__ void s_bwd_prop_SplineOutput_x24init_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpC_0, DiffPair_float_0 * dpdepth_0, DiffPair_float_0 * dpdistortion_loss_0, SplineOutput_0 * _s_dOut_16)
{

#line 175
    float _S710 = _s_dOut_16->distortion_loss_0;

#line 175
    dpdistortion_loss_0->primal_0 = (*dpdistortion_loss_0).primal_0;

#line 175
    dpdistortion_loss_0->differential_0 = _S710;

#line 175
    float _S711 = _s_dOut_16->depth_0;

#line 175
    dpdepth_0->primal_0 = (*dpdepth_0).primal_0;

#line 175
    dpdepth_0->differential_0 = _S711;

#line 175
    float3  _S712 = _s_dOut_16->C_2;

#line 175
    dpC_0->primal_0 = (*dpC_0).primal_0;

#line 175
    dpC_0->differential_0 = _S712;

#line 175
    return;
}


#line 182
__device__ void s_bwd_prop_extract_color_0(DiffPair_SplineState_0 * dpstate_2, DiffPair_float_0 * dptmin_0, SplineOutput_0 * _s_dOut_17)
{



    float _S713 = (*dpstate_2).primal_0.distortion_parts_0.x - (*dpstate_2).primal_0.distortion_parts_0.y;

#line 184
    float3  _S714 = make_float3 (0.0f);

#line 184
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S715;

#line 184
    (&_S715)->primal_0 = (*dpstate_2).primal_0.C_0;

#line 184
    (&_S715)->differential_0 = _S714;

#line 184
    DiffPair_float_0 _S716;

#line 184
    (&_S716)->primal_0 = (*dpstate_2).primal_0.padding_0.x;

#line 184
    (&_S716)->differential_0 = 0.0f;

#line 184
    DiffPair_float_0 _S717;

#line 184
    (&_S717)->primal_0 = _S713;

#line 184
    (&_S717)->differential_0 = 0.0f;

#line 184
    s_bwd_prop_SplineOutput_x24init_0(&_S715, &_S716, &_S717, _s_dOut_17);

#line 184
    float2  _S718 = make_float2 (_S717.differential_0, - _S717.differential_0);

#line 184
    float3  _S719 = _S714;

#line 184
    *&((&_S719)->x) = _S716.differential_0;

#line 184
    dptmin_0->primal_0 = (*dptmin_0).primal_0;

#line 184
    dptmin_0->differential_0 = 0.0f;

#line 184
    SplineState_0 _S720 = SplineState_x24_syn_dzero_0();

#line 184
    (&_S720)->distortion_parts_0 = _S718;

#line 184
    (&_S720)->padding_0 = _S719;

#line 184
    (&_S720)->C_0 = _S715.differential_0;

#line 184
    dpstate_2->primal_0 = (*dpstate_2).primal_0;

#line 184
    dpstate_2->differential_0 = _S720;

#line 182
    return;
}


#line 182
__device__ void s_bwd_extract_color_0(DiffPair_SplineState_0 * _S721, DiffPair_float_0 * _S722, SplineOutput_0 * _S723)
{

#line 182
    s_bwd_prop_extract_color_0(_S721, _S722, _S723);

#line 182
    return;
}


#line 260 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__global__ void __kernel__backwards_kernel(TensorView last_state_0, TensorView last_dirac_0, TensorView iters_0, TensorView tri_collection_0, TensorView ray_origins_0, TensorView ray_directions_0, DualModel_0 model_2, TensorView initial_drgb_0, TensorView dL_dinital_drgb_0, TensorView touch_count_0, TensorView dL_doutputs_0, TensorView wcts_0, float tmin_1, float tmax_1, float max_prim_size_3, uint max_iters_0)
{

#line 260
    DualModel_0 _S724 = model_2;

#line 282
    uint ray_ind_1 = (((threadIdx)) + ((blockIdx)) * ((blockDim))).x;
    uint _S725 = ((ray_origins_0).sizes[(0U)]);

#line 283
    if(ray_ind_1 >= _S725)
    {

#line 284
        return;
    }
    SplineState_0 dual_state_0 = get_state_0(last_state_0, ray_ind_1);
    float3  _S726 = get_float3_0(ray_directions_0, ray_ind_1);
    float3  _S727 = get_float3_0(ray_origins_0, ray_ind_1);

#line 288
    float3  _S728 = _S727 + make_float3 (tmin_1) * _S726;

#line 22 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/spline-machine.slang"
    float2  _S729 = make_float2 (0.0f);

    float3  _S730 = make_float3 (0.0f);

#line 291 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
    SplineState_0 _S731 = SplineState_x24init_0(_S729, _S729, _S730, 0.0f, make_float4 (0.0f), 0.0f, _S730);

#line 291
    DiffPair_SplineState_0 deriv_state_1;

#line 291
    (&deriv_state_1)->primal_0 = dual_state_0;

#line 291
    (&deriv_state_1)->differential_0 = _S731;



    float4  _S732 = get_float4_0(dL_doutputs_0, ray_ind_1);
    float _S733 = ((dL_doutputs_0).load<float>((ray_ind_1), (4U)));
    SplineOutput_0 dL_doutput_0;
    (&dL_doutput_0)->C_2 = make_float3 (_S732.x, _S732.y, _S732.z);
    (&dL_doutput_0)->depth_0 = _S732.w;
    (&dL_doutput_0)->distortion_loss_0 = _S733;

    int _S734 = ((iters_0).load<int>((ray_ind_1)));

#line 302
    uint num_iters_0 = (U32_max(((U32_min((uint(_S734)), (max_iters_0)))), (0U)));
    int _S735 = ((iters_0).load<int>((ray_ind_1)));

#line 303
    bool _S736;

#line 303
    if(uint(_S735) >= (max_iters_0 - 1U))
    {

#line 303
        _S736 = true;

#line 303
    }
    else
    {

#line 303
        int _S737 = ((iters_0).load<int>((ray_ind_1)));

#line 303
        _S736 = _S737 <= int(0);

#line 303
    }

#line 303
    if(_S736)
    {

#line 303
        return;
    }

#line 304
    DiffPair_float_0 dtmin_0;

#line 304
    (&dtmin_0)->primal_0 = tmin_1;

#line 304
    (&dtmin_0)->differential_0 = 0.0f;

#line 304
    SplineOutput_0 _S738 = dL_doutput_0;

#line 304
    s_bwd_extract_color_0(&deriv_state_1, &dtmin_0, &_S738);



    uint _S739 = (((&_S724)->features_1).sizes[(1U)]);
    int _S740 = int((F32_sqrt((float(_S739))))) - int(1);


    uint _S741 = ((wcts_0).sizes[(0U)]);

#line 312
    uint tri_ind_0;

#line 312
    if(ray_ind_1 < _S741)
    {

#line 312
        tri_ind_0 = ray_ind_1;

#line 312
    }
    else
    {

#line 312
        tri_ind_0 = 0U;

#line 312
    }
    float _S742 = ((wcts_0).load<float>((tri_ind_0), (0U), (0U)));

#line 313
    float _S743 = ((wcts_0).load<float>((tri_ind_0), (0U), (1U)));

#line 313
    float _S744 = ((wcts_0).load<float>((tri_ind_0), (0U), (2U)));

#line 313
    float _S745 = ((wcts_0).load<float>((tri_ind_0), (0U), (3U)));

#line 313
    float _S746 = ((wcts_0).load<float>((tri_ind_0), (1U), (0U)));

#line 313
    float _S747 = ((wcts_0).load<float>((tri_ind_0), (1U), (1U)));

#line 313
    float _S748 = ((wcts_0).load<float>((tri_ind_0), (1U), (2U)));

#line 313
    float _S749 = ((wcts_0).load<float>((tri_ind_0), (1U), (3U)));

#line 313
    float _S750 = ((wcts_0).load<float>((tri_ind_0), (2U), (0U)));

#line 313
    float _S751 = ((wcts_0).load<float>((tri_ind_0), (2U), (1U)));

#line 313
    float _S752 = ((wcts_0).load<float>((tri_ind_0), (2U), (2U)));

#line 313
    float _S753 = ((wcts_0).load<float>((tri_ind_0), (2U), (3U)));

#line 313
    float _S754 = ((wcts_0).load<float>((tri_ind_0), (3U), (0U)));

#line 313
    float _S755 = ((wcts_0).load<float>((tri_ind_0), (3U), (1U)));

#line 313
    float _S756 = ((wcts_0).load<float>((tri_ind_0), (3U), (2U)));

#line 313
    float _S757 = ((wcts_0).load<float>((tri_ind_0), (3U), (3U)));

#line 313
    Matrix<float, 4, 4>  wct_2 = makeMatrix<float, 4, 4> (_S742, _S743, _S744, _S745, _S746, _S747, _S748, _S749, _S750, _S751, _S752, _S753, _S754, _S755, _S756, _S757);

#line 320
    Matrix<float, 4, 4>  _S758 = inverse_0(wct_2);

    uint _S759 = (U32_max((num_iters_0 - 1U), (0U)));

#line 322
    uint _S760 = ((ray_origins_0).sizes[(0U)]);

#line 322
    int _S761 = ((tri_collection_0).load<int>((ray_ind_1 + _S759 * _S760)));

#line 322
    uint tri_ind_1 = uint(_S761);
    uint _S762 = uint(_S740);

#line 323
    DualModel_0 _S763 = _S724;

#line 323
    ControlPoint_0 _S764 = load_ctrl_pt_0(tri_ind_1, &_S763, _S728, _S726, _S762, false);


    int _S765 = int(num_iters_0);

#line 326
    SplineState_0 dual_state_1 = dual_state_0;

#line 326
    ControlPoint_0 ctrl_pt_3 = _S764;

#line 326
    tri_ind_0 = tri_ind_1;

#line 326
    int i_4 = _S765;

#line 326
    for(;;)
    {

#line 326
        int i_5 = i_4 - int(1);

#line 326
        if(i_4 > int(0))
        {
        }
        else
        {

#line 326
            break;
        }

        ControlPoint_0 old_ctrl_pt_1;
        int _S766 = i_5 - int(1);

#line 330
        uint old_tri_ind_0;

#line 330
        if(_S766 >= int(0))
        {

#line 331
            uint _S767 = uint(_S766);

#line 331
            uint _S768 = ((ray_origins_0).sizes[(0U)]);

#line 331
            int _S769 = ((tri_collection_0).load<int>((ray_ind_1 + _S767 * _S768)));

#line 331
            uint old_tri_ind_1 = uint(_S769);

#line 331
            DualModel_0 _S770 = _S724;

#line 331
            ControlPoint_0 _S771 = load_ctrl_pt_0(old_tri_ind_1, &_S770, _S728, _S726, _S762, false);
            old_ctrl_pt_1 = _S771;

#line 332
            old_tri_ind_0 = old_tri_ind_1;

#line 330
        }
        else
        {

            (&old_ctrl_pt_1)->t_0 = 0.0f;
            (&old_ctrl_pt_1)->dirac_0 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);

#line 330
        }

#line 330
        SplineState_0 _S772 = dual_state_1;

#line 330
        ControlPoint_0 _S773 = ctrl_pt_3;

#line 330
        ControlPoint_0 _S774 = old_ctrl_pt_1;

#line 330
        SplineState_0 _S775 = inverse_update_dual_0(&_S772, &_S773, &_S774, tmin_1, tmax_1);

#line 345
        uint _S776 = uint((F32_floor((float(tri_ind_0 / 2U)))));
        uint _S777 = ((tri_ind_0) % (2U));

#line 346
        SplineState_0 _S778 = _S775;

#line 346
        ControlPoint_0 _S779 = old_ctrl_pt_1;

#line 346
        ControlPoint_0 _S780 = ctrl_pt_3;

#line 346
        DiffPair_SplineState_0 _S781 = deriv_state_1;

#line 346
        DiffPair_SplineState_0 _S782 = run_update_0(&_S778, &_S779, &_S780, _S776, _S777, ray_ind_1, &_S781, _S728, _S726, tmin_1, tmax_1, _S762, max_prim_size_3, wct_2, _S758, &_S724);

#line 351
        int itemp_0;
        *((&itemp_0)) = atomicAdd((touch_count_0).data_ptr_at<int>((uint((F32_floor((float(tri_ind_0 / 2U))))))), (int(1)));



        ControlPoint_0 _S783 = old_ctrl_pt_1;

#line 356
        (&deriv_state_1)->primal_0 = _S775;

#line 356
        (&deriv_state_1)->differential_0 = _S782.differential_0;

#line 326
        dual_state_1 = _S775;

#line 326
        ctrl_pt_3 = _S783;

#line 326
        tri_ind_0 = old_tri_ind_0;

#line 326
        i_4 = i_5;

#line 326
    }

#line 362
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (0U), (deriv_state_1.differential_0.drgb_0.x));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (1U), (deriv_state_1.differential_0.drgb_0.y));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (2U), (deriv_state_1.differential_0.drgb_0.z));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (3U), (deriv_state_1.differential_0.drgb_0.w));
    return;
}


#line 12087 "hlsl.meta.slang"
__device__ float3  max_0(float3  x_8, float3  y_6)
{

#line 6802
    float3  result_11;

#line 6802
    int i_6 = int(0);

#line 6802
    for(;;)
    {

#line 6802
        if(i_6 < int(3))
        {
        }
        else
        {

#line 6802
            break;
        }

#line 6802
        *_slang_vector_get_element_ptr(&result_11, i_6) = (F32_max((_slang_vector_get_element(x_8, i_6)), (_slang_vector_get_element(y_6, i_6))));

#line 6802
        i_6 = i_6 + int(1);

#line 6802
    }

#line 6802
    return result_11;
}


#line 11590
__device__ float length_0(float4  x_9)
{

#line 11602
    return (F32_sqrt((dot_1(x_9, x_9))));
}


#line 11590
__device__ float length_1(float3  x_10)
{

#line 11602
    return (F32_sqrt((dot_0(x_10, x_10))));
}


#line 103 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/safe-math.slang"
__device__ float4  safe_div_3(float4  a_7, float b_5)
{

#line 104
    return make_float4 (safe_div_0(a_7.x, b_5), safe_div_0(a_7.y, b_5), safe_div_0(a_7.z, b_5), safe_div_0(a_7.w, b_5));
}


#line 22 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/tri-intersect.slang"
__device__ Matrix<float, 3, 3>  quat2mat_0(float4  quat_2)
{

#line 22
    float _S784 = quat_2.z;

#line 29
    float _S785 = _S784 * _S784;

#line 29
    float _S786 = quat_2.w * quat_2.w;
    float _S787 = quat_2.y * quat_2.z;

#line 30
    float _S788 = quat_2.x * quat_2.w;
    float _S789 = quat_2.y * quat_2.w;

#line 31
    float _S790 = quat_2.x * quat_2.z;


    float _S791 = quat_2.y * quat_2.y;
    float _S792 = quat_2.z * quat_2.w;

#line 35
    float _S793 = quat_2.x * quat_2.y;

#line 41
    return makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S785 + _S786), 2.0f * (_S787 - _S788), 2.0f * (_S789 + _S790), 2.0f * (_S787 + _S788), 1.0f - 2.0f * (_S791 + _S786), 2.0f * (_S792 - _S793), 2.0f * (_S789 - _S790), 2.0f * (_S792 + _S793), 1.0f - 2.0f * (_S791 + _S785));
}


#line 369 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_mix_drgb_0(DiffPair_float_0 * dpdensity_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcolor_1, float4  _s_dOut_18)
{

#line 370
    float _S794 = (*dpcolor_1).primal_0.z * _s_dOut_18.w;

#line 370
    float _S795 = (*dpcolor_1).primal_0.y * _s_dOut_18.z;

#line 370
    float _S796 = (*dpcolor_1).primal_0.x * _s_dOut_18.y;

#line 370
    float3  _S797 = make_float3 ((*dpdensity_1).primal_0 * _s_dOut_18.y, (*dpdensity_1).primal_0 * _s_dOut_18.z, (*dpdensity_1).primal_0 * _s_dOut_18.w);

#line 370
    dpcolor_1->primal_0 = (*dpcolor_1).primal_0;

#line 370
    dpcolor_1->differential_0 = _S797;

#line 1230 "core.meta.slang"
    float _S798 = _s_dOut_18.x + _S794 + _S795 + _S796;

#line 1230
    dpdensity_1->primal_0 = (*dpdensity_1).primal_0;

#line 1230
    dpdensity_1->differential_0 = _S798;

#line 369 "E:/projects/multi-view-hair/code/new-ever/splinetracers/slang/backwards_kernel.slang"
    return;
}


#line 369
__device__ void s_bwd_mix_drgb_0(DiffPair_float_0 * _S799, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S800, float4  _S801)
{

#line 369
    s_bwd_prop_mix_drgb_0(_S799, _S800, _S801);

#line 369
    return;
}




__global__ void __kernel__backwards_initial_drgb_kernel(TensorView ray_origins_1, TensorView ray_directions_1, DualModel_0 model_3, TensorView initial_drgb_1, TensorView initial_inds_0, TensorView dL_dinital_drgb_1, TensorView touch_count_1, float tmin_2)
{

#line 386
    uint3  _S802 = ((threadIdx));

#line 386
    uint3  _S803 = ((blockIdx));

#line 386
    uint3  _S804 = ((blockDim));

#line 386
    uint thread_j_0 = _S802.x + _S803.x * _S804.x;
    uint thread_i_0 = _S802.y + _S803.y * _S804.y;
    uint _S805 = ((initial_inds_0).sizes[(0U)]);

#line 388
    bool _S806;

#line 388
    if(thread_i_0 >= _S805)
    {

#line 388
        _S806 = true;

#line 388
    }
    else
    {

#line 388
        uint _S807 = ((ray_directions_1).sizes[(0U)]);

#line 388
        _S806 = thread_j_0 >= _S807;

#line 388
    }

#line 388
    if(_S806)
    {

#line 389
        return;
    }
    int _S808 = ((initial_inds_0).load<int>((thread_i_0)));

#line 391
    uint prim_ind_2 = uint(_S808);


    float3  mean_1 = get_float3_0(model_3.means_0, prim_ind_2);
    float4  quat_3 = get_float4_0(model_3.quats_0, prim_ind_2);
    float3  scales_3 = get_float3_0(model_3.scales_2, prim_ind_2);
    float3  rayd_2 = get_float3_0(ray_directions_1, thread_j_0);
    float3  _S809 = get_float3_0(ray_origins_1, 0U);

#line 404
    if((length_1(safe_div_2(mul_1(_S809 + make_float3 (tmin_2) * rayd_2 - mean_1, quat2mat_0(safe_div_3(quat_3, length_0(quat_3)))), max_0(scales_3, make_float3 (9.99999993922529029e-09f))))) <= 1.0f)
    {
        float _S810 = ((model_3.densities_0).load<float>((prim_ind_2)));

        Features_0 feat_7 = get_feats_0(model_3.features_1, prim_ind_2, 0U);

#line 408
        Features_0 _S811 = feat_7;

#line 408
        float3  _S812 = eval_color_0(rayd_2, &_S811, 0U);

        float3  _S813 = make_float3 (0.0f);

#line 410
        DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_color_1;

#line 410
        (&deriv_color_1)->primal_0 = _S812;

#line 410
        (&deriv_color_1)->differential_0 = _S813;
        DiffPair_float_0 deriv_density_1;

#line 411
        (&deriv_density_1)->primal_0 = _S810;

#line 411
        (&deriv_density_1)->differential_0 = 0.0f;
        float4  vdL_dinital_drgb_0 = get_float4_0(dL_dinital_drgb_1, thread_j_0);
        s_bwd_mix_drgb_0(&deriv_density_1, &deriv_color_1, vdL_dinital_drgb_0);

#line 405
        float temp_5;

#line 415
        *((&temp_5)) = atomicAdd((model_3.dL_ddensities_0).data_ptr_at<float>((prim_ind_2)), (deriv_density_1.differential_0));

        DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_direction_1;

#line 417
        (&deriv_direction_1)->primal_0 = rayd_2;

#line 417
        (&deriv_direction_1)->differential_0 = _S813;
        Features_0 _S814 = Features_x24init_0(_S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813, _S813);

#line 418
        DiffPair_Features_0 d_feat_1;

#line 418
        (&d_feat_1)->primal_0 = feat_7;

#line 418
        (&d_feat_1)->differential_0 = _S814;
        s_bwd_eval_color_0(&deriv_direction_1, &d_feat_1, 0U, deriv_color_1.differential_0);

#line 425
        atomic_add_float3_1(model_3.dL_dfeatures_0, prim_ind_2, 0U, make_float3 ((&(&d_feat_1)->differential_0)->f0_0.x, (&(&d_feat_1)->differential_0)->f0_0.y, (&(&d_feat_1)->differential_0)->f0_0.z));

#line 404
    }

#line 427
    return;
}

