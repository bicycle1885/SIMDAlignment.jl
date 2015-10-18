#ifndef SIMD_H
#define SIMD_H

#include <array>
#include <type_traits>
#include <x86intrin.h>

#define T_IS(typ) (std::is_same<T,typ>::value)

template<typename T>
inline __m128i simd_set1(const T x)
{
    if (T_IS(int8_t))
        return _mm_set1_epi8(x);
    if (T_IS(int16_t))
        return _mm_set1_epi16(x);
    if (T_IS(int32_t))
        return _mm_set1_epi32(x);
}

template<typename T,size_t n>
inline __m128i simd_set(std::array<T,n> xs)
{
    if (T_IS(int8_t) && n == 16) {
        return _mm_set_epi8(
            xs[15], xs[14], xs[13], xs[12],
            xs[11], xs[10], xs[ 9], xs[ 8],
            xs[ 7], xs[ 6], xs[ 5], xs[ 4],
            xs[ 3], xs[ 2], xs[ 1], xs[ 0]
        );
    }
    if (T_IS(int16_t) && n == 8) {
        return _mm_set_epi16(
            xs[7], xs[6], xs[5], xs[4],
            xs[3], xs[2], xs[1], xs[0]
        );
    }
    if (T_IS(int32_t) && n == 4) {
        return _mm_set_epi32(
            xs[3], xs[2], xs[1], xs[0]
        );
    }
}

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
inline __m128i simd_adds(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_adds_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_adds_epi16(x, y);
}

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
inline __m128i simd_subs(const __m128i x, const __m128i y)
{
    if (T_IS(int8_t))
        return _mm_subs_epi8(x, y);
    if (T_IS(int16_t))
        return _mm_subs_epi16(x, y);
}

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
inline __m128i simd_insert(const __m128i x, const T y, const int m)
{
    if (T_IS(int8_t))
        return _mm_insert_epi8(x, y, m);
    if (T_IS(int16_t))
        return _mm_insert_epi16(x, y, m);
    if (T_IS(int32_t))
        return _mm_insert_epi32(x, y, m);
}

#undef T_IS

#endif
