#ifndef SIMD_H
#define SIMD_H

#include <array>
#include <type_traits>
#include <x86intrin.h>
#include <immintrin.h>

#define T_IS(typ) (std::is_same<T,typ>::value)

// set1
template<typename T,typename V>
inline V simd_set1(const T x);

template<>
inline __m128i simd_set1(const int8_t x)
{
    return _mm_set1_epi8(x);
}

template<>
inline __m128i simd_set1(const int16_t x)
{
    return _mm_set1_epi16(x);
}

template<>
inline __m128i simd_set1(const int32_t x)
{
    return _mm_set1_epi32(x);
}

template<>
inline __m256i simd_set1(const int8_t x)
{
    return _mm256_set1_epi8(x);
}

template<>
inline __m256i simd_set1(const int16_t x)
{
    return _mm256_set1_epi16(x);
}

template<>
inline __m256i simd_set1(const int32_t x)
{
    return _mm256_set1_epi32(x);
}


// set
template<typename T,size_t n,typename V>
inline V simd_set(const std::array<T,n>& xs);

template<>
inline __m128i simd_set(const std::array<int8_t,16>& xs)
{
    return _mm_set_epi8(
        xs[15], xs[14], xs[13], xs[12],
        xs[11], xs[10], xs[ 9], xs[ 8],
        xs[ 7], xs[ 6], xs[ 5], xs[ 4],
        xs[ 3], xs[ 2], xs[ 1], xs[ 0]
    );
}

template<>
inline __m128i simd_set(const std::array<int16_t,8>& xs)
{
    return _mm_set_epi16(
        xs[7], xs[6], xs[5], xs[4],
        xs[3], xs[2], xs[1], xs[0]
    );
}

template<>
inline __m128i simd_set(const std::array<int32_t,4>& xs)
{
    return _mm_set_epi32(
        xs[3], xs[2], xs[1], xs[0]
    );
}

template<>
inline __m256i simd_set(const std::array<int8_t,32>& xs)
{
    return _mm256_set_epi8(
        xs[31], xs[30], xs[29], xs[28],
        xs[27], xs[26], xs[25], xs[24],
        xs[23], xs[22], xs[21], xs[20],
        xs[19], xs[18], xs[17], xs[16],
        xs[15], xs[14], xs[13], xs[12],
        xs[11], xs[10], xs[ 9], xs[ 8],
        xs[ 7], xs[ 6], xs[ 5], xs[ 4],
        xs[ 3], xs[ 2], xs[ 1], xs[ 0]
    );
}

template<>
inline __m256i simd_set(const std::array<int16_t,16>& xs)
{
    return _mm256_set_epi16(
        xs[15], xs[14], xs[13], xs[12],
        xs[11], xs[10], xs[ 9], xs[ 8],
        xs[ 7], xs[ 6], xs[ 5], xs[ 4],
        xs[ 3], xs[ 2], xs[ 1], xs[ 0]
    );
}

template<>
inline __m256i simd_set(const std::array<int32_t,8>& xs)
{
    return _mm256_set_epi32(
        xs[7], xs[6], xs[5], xs[4],
        xs[3], xs[2], xs[1], xs[0]
    );
}

// max
template<typename T,typename V>
inline V simd_max(const V x, const V y);

template<typename T>
inline __m128i simd_max(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_max_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_max_epi16(x, y);
    if (T_IS(int32_t))
        return _mm_max_epi32(x, y);
}

template<typename T>
inline __m256i simd_max(const __m256i x, const __m256i y)
{
    if (T_IS(int8_t))
        return _mm256_max_epi8(x, y);
    if (T_IS(int16_t))
        return _mm256_max_epi16(x, y);
    if (T_IS(int32_t))
        return _mm256_max_epi32(x, y);
}

// add (saturated)
template<typename T,typename V>
inline V simd_adds(const V x, const V y);

template<typename T>
inline __m128i simd_adds(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_adds_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_adds_epi16(x, y);
}

template<typename T>
inline __m256i simd_adds(const __m256i x, const __m256i y)
{
    if (T_IS(int8_t))
        return _mm256_adds_epi8(x, y);
    if (T_IS(int16_t))
        return _mm256_adds_epi16(x, y);
}

// add
template<typename T,typename V>
inline V simd_add(const V x, const V y);

template<typename T>
inline __m128i simd_add(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_add_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_add_epi16(x, y);
    if (T_IS(int32_t))
        return _mm_add_epi32(x, y);
}

template<typename T>
inline __m256i simd_add(const __m256i x, const __m256i y)
{
    if (T_IS(int8_t))
        return _mm256_add_epi8(x, y);
    if (T_IS(int16_t))
        return _mm256_add_epi16(x, y);
    if (T_IS(int32_t))
        return _mm256_add_epi32(x, y);
}

// sub (saturated)
template<typename T,typename V>
inline V simd_subs(const V x, const V y);

template<typename T>
inline __m128i simd_subs(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_subs_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_subs_epi16(x, y);
}

template<typename T>
inline __m256i simd_subs(const __m256i x, const __m256i y)
{
    if (T_IS(int8_t))
        return _mm256_subs_epi8(x, y);
    if (T_IS(int16_t))
        return _mm256_subs_epi16(x, y);
}

// sub
template<typename T,typename V>
inline V simd_sub(const V x, const V y);

template<typename T>
inline __m128i simd_sub(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_sub_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_sub_epi16(x, y);
    if (T_IS(int32_t))
        return _mm_sub_epi32(x, y);
}

template<typename T>
inline __m256i simd_sub(const __m256i x, const __m256i y)
{
    if (T_IS(int8_t))
        return _mm256_sub_epi8(x, y);
    if (T_IS(int16_t))
        return _mm256_sub_epi16(x, y);
    if (T_IS(int32_t))
        return _mm256_sub_epi32(x, y);
}

// extract
template<typename T,typename V>
inline int simd_extract(const V x, const int m);

template<typename T>
inline int simd_extract(const __m128i x, const int m)
{
    if (T_IS(int8_t))
        return _mm_extract_epi8(x, m);
    if (T_IS(int16_t))
        return _mm_extract_epi16(x, m);
    if (T_IS(int32_t))
        return _mm_extract_epi32(x, m);
}

template<typename T>
inline int simd_extract(const __m256i x, const int m)
{
    if (T_IS(int8_t))
        return _mm256_extract_epi8(x, m);
    if (T_IS(int16_t))
        return _mm256_extract_epi16(x, m);
    if (T_IS(int32_t))
        return _mm256_extract_epi32(x, m);
}

// insert
template<typename T,typename V>
inline V simd_insert(const V x, const T y, const int m);

template<typename T>
inline __m128i simd_insert(const __m128i x, const T y, const int m)
{
    if (T_IS(int8_t))
        return _mm_insert_epi8(x, y, m);
    if (T_IS(int16_t))
        return _mm_insert_epi16(x, y, m);
    if (T_IS(int32_t))
        return _mm_insert_epi32(x, y, m);
}

template<typename T>
inline __m256i simd_insert(const __m256i x, const T y, const int m)
{
    if (T_IS(int8_t))
        return _mm256_insert_epi8(x, y, m);
    if (T_IS(int16_t))
        return _mm256_insert_epi16(x, y, m);
    if (T_IS(int32_t))
        return _mm256_insert_epi32(x, y, m);
}

#undef T_IS

#endif
