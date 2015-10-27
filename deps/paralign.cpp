// parallel inter-sequence alignment

#include <limits>
#include <array>
#include "simdalign.h"

struct slot_t
{
    int id;
    size_t pos;

    slot_t() {}
    slot_t(int id, size_t pos) : id(id), pos(pos) {}

    inline bool operator==(const slot_t &other) const {
        return id == other.id;
    }
    inline bool operator!=(const slot_t &other) const {
        return id != other.id;
    }
};

const slot_t empty_slot = slot_t(-1, 0);

template<typename score_t>
static inline score_t affine_gap_score(int k, score_t gap_open, score_t gap_extend)
{
    return k > 0 ? -(gap_open + gap_extend * k) : 0;
}

template<size_t n>
static bool is_vacant(const std::array<slot_t,n> slots)
{
    for (const slot_t& slot : slots)
        if (slot != empty_slot)
            return false;
    return true;
}

template<typename vec_t,typename score_t,size_t n>
static void fill_profile(const seq_t* refs,
                         const std::array<slot_t,n>& slots,
                         const submat_t<score_t>& submat,
                         vec_t* profile)
{
    // prefetch characters in reference sequences
    std::array<uint8_t,n> refchars;
    for (int k = 0; k < n; k++) {
        slot_t slot = slots[k];
        if (slot == empty_slot)
            continue;
        refchars[k] = refs[slot.id][slot.pos];
    }
    for (uint8_t seqchar = 0; seqchar < submat.size; seqchar++) {
        std::array<score_t,n> svec;
        for (int k = 0; k < n; k++) {
            slot_t slot = slots[k];
            if (slot == empty_slot)
                continue;
            uint8_t refchar = refchars[k];
            svec[k] = submat.data[refchar * submat.size + seqchar];
        }
        profile[seqchar] = simd_set<score_t,n,vec_t>(svec);
    }
}

// update the next column
template<typename vec_t,typename score_t,size_t n>
static void loop(const uint8_t* useq,
                 const size_t seqlen,
                 const vec_t* prof,
                 const std::array<slot_t,n>& slots,
                 const score_t gap_open,
                 const score_t gap_extend,
                 vec_t* colE,
                 vec_t* colH)
{
    const vec_t Ginit = simd_set1<score_t,vec_t>(gap_open + gap_extend);
    const vec_t Gextd = simd_set1<score_t,vec_t>(gap_extend);
    vec_t H_diag = colH[0];
    std::array<score_t,n> vec;
    for (int k = 0; k < n; k++)
        vec[k] = affine_gap_score(slots[k].pos + 1, gap_open, gap_extend);
    vec_t F = simd_sub<score_t>(simd_set<score_t,n,vec_t>(vec), Ginit);
    colH[0] = simd_set<score_t,n,vec_t>(vec);
    for (size_t i = 1; i <= seqlen; i++) {
        vec_t E = colE[i];
        vec_t H = simd_max<score_t>(
            simd_add<score_t>(H_diag, prof[useq[i-1]]),
            simd_max<score_t>(E, F)
        );
        H_diag = colH[i];
        colH[i] = H;
        colE[i] = simd_max<score_t>(
            simd_sub<score_t>(H, Ginit),
            simd_sub<score_t>(E, Gextd)
        );
        F = simd_max<score_t>(
            simd_sub<score_t>(H, Ginit),
            simd_sub<score_t>(F, Gextd)
        );
    }
}


template<typename vec_t,typename score_t>
int paralign_score(buffer_t* buffer,
                   const submat_t<score_t> submat,
                   const score_t gap_open,
                   const score_t gap_extend,
                   const seq_t seq,
                   const seq_t* refs,
                   const int n_refs,
                   alignment_t** alignments)
{
    if (n_refs == 0)
        return 0;
    else if (n_refs < 0)
        return 1;

    // allocate working space
    if (expand_buffer(buffer, sizeof(vec_t) * (seq.len + 1) * 2 +
                              sizeof(vec_t) * submat.size +
                              sizeof(uint8_t) * seq.len)) {
        return 1;
    }
    // NOTE: colE[0] is not used
    vec_t* colE = (vec_t*)buffer->data;
    vec_t* colH = colE + seq.len + 1;
    vec_t* prof = colH + seq.len + 1;
    uint8_t* useq = reinterpret_cast<uint8_t*>(prof + submat.size);

    // unpack sequence
    for (size_t i = 0; i < seq.len; i++)
        useq[i] = seq[i];

    // initialize slots which hold the reference sequences
    const int n_max_par = sizeof(vec_t) / sizeof(score_t);
    std::array<slot_t,n_max_par> slots;
    slots.fill(empty_slot);
    int next_ref = 0;

    // outer loop along refs
    while (true) {
        // initialize the slots and the column vectors
        for (int k = 0; k < n_max_par; k++) {
            slot_t &slot = slots[k];

            if (slot != empty_slot) {
                slot.pos++;
                if (slot.pos < refs[slot.id].len)
                    continue;
                else
                    (*alignments[slot.id]).score = simd_extract<score_t>(colH[seq.len], k);
            }

            // find the next non-empty sequences if any
            bool found = false;
            while (next_ref < n_refs && !found) {
                // reset E and H
                colH[0] = simd_insert<score_t>(colH[0], 0, k);
                for (int i = 1; i <= seq.len; i++) {
                    score_t h = affine_gap_score(i, gap_open, gap_extend);
                    colH[i] = simd_insert(colH[i], h, k);
                    colE[i] = simd_insert(colE[i], static_cast<score_t>(h - (gap_open + gap_extend)), k);
                }
                seq_t ref = refs[next_ref];
                if (ref.len == 0) {
                    (*alignments[next_ref++]).score = simd_extract<score_t>(colH[seq.len], k);
                }
                else {
                    slot.id = next_ref++;
                    slot.pos = 0;
                    found = true;
                }
            }

            if (!found)
                slots[k] = empty_slot;
        }

        // check if there are remaining slots
        if (is_vacant(slots))
            break;

        // fill the temporary profile
        fill_profile(refs, slots, submat, prof);

        // inner loop along seq
        // TODO: detect saturation
        loop(useq, seq.len, prof, slots, gap_open, gap_extend, colE, colH);
    }

    return 0;
}


// 128 bits
int paralign_score_i8x16(buffer_t* buffer,
                         const submat_t<int8_t> submat,
                         const int8_t gap_open,
                         const int8_t gap_extend,
                         const seq_t seq,
                         const seq_t* refs,
                         const int n_refs,
                         alignment_t** alignments)
{
    return paralign_score<__m128i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

int paralign_score_i16x8(buffer_t* buffer,
                         const submat_t<int16_t> submat,
                         const int16_t gap_open,
                         const int16_t gap_extend,
                         const seq_t seq,
                         const seq_t* refs,
                         const int n_refs,
                         alignment_t** alignments)
{
    return paralign_score<__m128i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

int paralign_score_i32x4(buffer_t* buffer,
                         const submat_t<int32_t> submat,
                         const int32_t gap_open,
                         const int32_t gap_extend,
                         const seq_t seq,
                         const seq_t* refs,
                         const int n_refs,
                         alignment_t** alignments)
{
    return paralign_score<__m128i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}


// 256 bits
int paralign_score_i8x32(buffer_t* buffer,
                         const submat_t<int8_t> submat,
                         const int8_t gap_open,
                         const int8_t gap_extend,
                         const seq_t seq,
                         const seq_t* refs,
                         const int n_refs,
                         alignment_t** alignments)
{
    return paralign_score<__m256i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

int paralign_score_i16x16(buffer_t* buffer,
                          const submat_t<int16_t> submat,
                          const int16_t gap_open,
                          const int16_t gap_extend,
                          const seq_t seq,
                          const seq_t* refs,
                          const int n_refs,
                          alignment_t** alignments)
{
    return paralign_score<__m256i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

int paralign_score_i32x8(buffer_t* buffer,
                         const submat_t<int32_t> submat,
                         const int32_t gap_open,
                         const int32_t gap_extend,
                         const seq_t seq,
                         const seq_t* refs,
                         const int n_refs,
                         alignment_t** alignments)
{
    return paralign_score<__m256i>(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

