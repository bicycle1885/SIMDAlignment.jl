#ifndef SIMD_H
#define SIMD_H

#include <array>
#include <type_traits>
#include "emmintrin.h"

template<typename T>
inline __m128i simd_set1(const T x)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_set1_epi16(x);
}

template<typename T,size_t n>
inline __m128i simd_set(std::array<T,n> xs)
{
    if (std::is_same<T,int16_t>::value && n == 8) {
        return _mm_set_epi16(
            xs[7], xs[6], xs[5], xs[4],
            xs[3], xs[2], xs[1], xs[0]
        );
    }
}

template<typename T>
inline __m128i simd_max(const __m128i x, const __m128i y)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_max_epi16(x, y);
}

template<typename T>
inline __m128i simd_adds(const __m128i x, const __m128i y)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_adds_epi16(x, y);
}

template<typename T>
inline __m128i simd_subs(const __m128i x, const __m128i y)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_subs_epi16(x, y);
}

template<typename T>
inline int simd_extract(const __m128i x, const int m)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_extract_epi16(x, m);
}

template<typename T>
inline __m128i simd_insert(const __m128i x, const T y, const int m)
{
    if (std::is_same<T,int16_t>::value)
        return _mm_insert_epi16(x, y, m);
}

#endif
