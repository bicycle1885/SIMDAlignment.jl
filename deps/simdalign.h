#ifndef SIMDALIGN_H
#define SIMDALIGN_H

#include "stdlib.h"
#include "stdint.h"
#include "simd.h"

// sequence
struct seq_t
{
    uint8_t* data;
    size_t startpos;
    size_t len;
    // whether the seqnece is packed, used for DNA/RNA sequence
    bool packed;

    // NOTE: 0-based index unlike Julia
    uint8_t operator[](const int i) const {
        return data[i];
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
    T* data;
    int size;
};

// alignment result
template<typename T>
struct alignment_t
{
    T score;
    uint8_t* trace;
    size_t seqlen;
    size_t reflen;
    size_t endpos_seq;
    size_t endpos_ref;

    // score-only constructor
    alignment_t(T score) :
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
    int paralign_score_i8(buffer_t* buffer,
                          const submat_t<int8_t> submat,
                          const int8_t gap_open,
                          const int8_t gap_extend,
                          const seq_t seq,
                          const seq_t* refs,
                          const int n_refs,
                          alignment_t<int8_t>** alignments);
    int paralign_score_i16(buffer_t* buffer,
                           const submat_t<int16_t> submat,
                           const int16_t gap_open,
                           const int16_t gap_extend,
                           const seq_t seq,
                           const seq_t* refs,
                           const int n_refs,
                           alignment_t<int16_t>** alignments);
    int paralign_score_i32(buffer_t* buffer,
                           const submat_t<int32_t> submat,
                           const int32_t gap_open,
                           const int32_t gap_extend,
                           const seq_t seq,
                           const seq_t* refs,
                           const int n_refs,
                           alignment_t<int32_t>** alignments);
}

#endif
