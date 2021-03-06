#ifndef SIMDALIGN_H
#define SIMDALIGN_H

#include <vector>
#include "stdlib.h"
#include "stdint.h"
#include "simd.h"

// sequence
struct seq_t
{
    const uint8_t* data;
    // sequence length
    const size_t len;
    // the first position accessed by seq[0]
    const size_t offset;
    // data is organized in reversed order
    const bool reversed;
    // whether the seqnece is packed, used for DNA/RNA sequence
    const bool packed;

    seq_t(const std::vector<uint8_t>& data) : data(data.data()), len(data.size()), offset(0), reversed(false), packed(false) {}

    // NOTE: 0-based index unlike Julia
    inline uint8_t operator[](const size_t i) const {
        size_t j = offset + (reversed ? -i : i);
        if (packed) {
            size_t q = j >> 2, r = j & 0b11;
            return (data[q] >> (r * 2)) & 0b11;
        }
        else {
            return data[j];
        }
    };
};


// sequence profile
template<typename T>
struct profile_s
{
    seq_t seq;
    T* data;
};

// substitution matrix (rectangular matrix: size x size)
template<typename T>
struct submat_t
{
    const T* data;
    const int size;

    submat_t(T* data, int size) : data(data), size(size) {};
};

// alignment result
struct alignment_t
{
    int64_t score;
    uint8_t* trace;
    size_t seqlen;
    size_t reflen;
    size_t endpos_seq;
    size_t endpos_ref;

    // score-only constructor
    alignment_t(const int64_t score) :
        score(score),
        trace(nullptr),
        seqlen(0),
        reflen(0),
        endpos_seq(0),
        endpos_ref(0) {}
};

// working space
struct buffer_t
{
    void* data;
    size_t len;
};


extern "C"
{
    buffer_t* make_buffer(void);
    int expand_buffer(buffer_t* buffer, size_t);
    void free_buffer(buffer_t*);

    // paralign.cpp
    int paralign_score_i8x16(buffer_t* buffer,
                             const submat_t<int8_t> submat,
                             const int8_t gap_open,
                             const int8_t gap_extend,
                             const seq_t seq,
                             const seq_t* refs,
                             const int n_refs,
                             alignment_t** alignments);
    int paralign_score_i16x8(buffer_t* buffer,
                             const submat_t<int16_t> submat,
                             const int16_t gap_open,
                             const int16_t gap_extend,
                             const seq_t seq,
                             const seq_t* refs,
                             const int n_refs,
                             alignment_t** alignments);
    int paralign_score_i32x4(buffer_t* buffer,
                             const submat_t<int32_t> submat,
                             const int32_t gap_open,
                             const int32_t gap_extend,
                             const seq_t seq,
                             const seq_t* refs,
                             const int n_refs,
                             alignment_t** alignments);
    int paralign_score_i8x32(buffer_t* buffer,
                             const submat_t<int8_t> submat,
                             const int8_t gap_open,
                             const int8_t gap_extend,
                             const seq_t seq,
                             const seq_t* refs,
                             const int n_refs,
                             alignment_t** alignments);
    int paralign_score_i16x16(buffer_t* buffer,
                              const submat_t<int16_t> submat,
                              const int16_t gap_open,
                              const int16_t gap_extend,
                              const seq_t seq,
                              const seq_t* refs,
                              const int n_refs,
                              alignment_t** alignments);
    int paralign_score_i32x8(buffer_t* buffer,
                             const submat_t<int32_t> submat,
                             const int32_t gap_open,
                             const int32_t gap_extend,
                             const seq_t seq,
                             const seq_t* refs,
                             const int n_refs,
                             alignment_t** alignments);
}

#endif
