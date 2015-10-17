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

template<typename score_t,size_t n>
static void fill_profile(const seq_t* refs, const std::array<slot_t,n>& slots, const submat_t<score_t> submat, __m128i* profile)
{
    for (uint8_t seq_char = 0; seq_char < submat.size; seq_char++) {
        std::array<score_t,n> svec;
        for (int k = 0; k < n; k++) {
            slot_t slot = slots[k];
            if (slot == empty_slot)
                continue;
            uint8_t ref_char = refs[slot.id][slot.pos];
            svec[k] = submat.data[ref_char * submat.size + seq_char];
        }
        profile[seq_char] = simd_set(svec);
    }
}

template<typename score_t,size_t n>
static __m128i initH(const std::array<slot_t,n>& slots, const score_t gap_open, const score_t gap_extend)
{
    std::array<score_t,n> svec;
    for (int k = 0; k < n; k++)
        svec[k] = affine_gap_score(slots[k].pos, gap_open, gap_extend);
    return simd_set(svec);
}


template<typename score_t>
int paralign_score(buffer_t* buffer,
                   const submat_t<score_t> submat,
                   const score_t gap_open,
                   const score_t gap_extend,
                   const seq_t seq,
                   const seq_t* refs,
                   const int n_refs,
                   alignment_t<score_t>** alignments)
{
    // the number of the maximum parallel runs
    const int n_max_par = 16 / sizeof(score_t);
    if (n_refs == 0)
        return 0;
    else if (n_refs < 0)
        return 1;

    // allocate working space
    if (expand_buffer(buffer, sizeof(__m128i) * seq.len * 2 +
                              sizeof(__m128i) * submat.size))
        return 1;
    __m128i* colE = (__m128i*)buffer->data;
    __m128i* colH = colE + seq.len;
    __m128i* prof = colH + seq.len;

    // initialize slots which hold the reference sequences
    std::array<slot_t,n_max_par> slots;
    for (int k = 0; k < n_max_par; k++)
        slots[k] = k < n_refs ? slot_t(k, 0) : empty_slot;
    int next_ref = std::min(n_refs, n_max_par);

    // initialize colE and colH
    const score_t minscore = std::numeric_limits<score_t>::min();
    for (int i = 0; i < seq.len; i++) {
        colE[i] = simd_set1(minscore);
        colH[i] = simd_set1(affine_gap_score(i + 1, gap_open, gap_extend));
    }

    // set gap penalty vectors
    const __m128i Ginit = simd_set1<score_t>(gap_open + gap_extend);
    const __m128i Gextd = simd_set1(gap_extend);

    // outer loop along refs
    while (true) {
        // find finished reference sequences
        for (int k = 0; k < n_max_par; k++) {
            slot_t &slot = slots[k];
            if (slot == empty_slot || slot.pos < refs[slot.id].len)
                continue;
            // store the alignment result
            (*alignments[slot.id]).score = simd_extract<score_t>(colH[seq.len-1], k);
            if (next_ref >= n_refs) {
                // no more reference sequences
                slots[k] = empty_slot;
                continue;
            }
            // fill the next reference sequence into the slot
            // TODO: handle empty reference sequences
            slot.id = next_ref++;
            slot.pos = 0;
            // reset E and H
            for (int i = 0; i < seq.len; i++) {
                colE[i] = simd_insert(colE[i], minscore, k);
                colH[i] = simd_insert(colH[i], affine_gap_score(i + 1, gap_open, gap_extend), k);
            }
        }

        // check if there are remaining slots
        if (is_vacant(slots))
            break;

        // fill the temporary profile
        fill_profile(refs, slots, submat, prof);

        // initialize vectors
        __m128i E = colE[0];
        __m128i F = simd_set1(minscore);
        __m128i H_diag = initH(slots, gap_open, gap_extend);

        // inner loop along seq
        // TODO: detect saturation
        for (size_t i = 0; i < seq.len; i++) {
            E = colE[i];
            __m128i H = simd_max<score_t>(
                simd_adds<score_t>(H_diag, prof[seq[i]]),
                simd_max<score_t>(E, F)
            );
            H_diag = colH[i];
            colH[i] = H;
            colE[i] = simd_max<score_t>(
                simd_subs<score_t>(H, Ginit),
                simd_subs<score_t>(E, Gextd)
            );
            F = simd_max<score_t>(
                simd_subs<score_t>(H, Ginit),
                simd_subs<score_t>(F, Gextd)
            );
        }

        // increment the positions of the reference sequences
        for (slot_t& slot : slots)
            if (slot != empty_slot)
                slot.pos++;
    }

    return 0;
}

int paralign_score_i8(buffer_t* buffer,
                      const submat_t<int8_t> submat,
                      const int8_t gap_open,
                      const int8_t gap_extend,
                      const seq_t seq,
                      const seq_t* refs,
                      const int n_refs,
                      alignment_t<int8_t>** alignments)
{
    return paralign_score(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}

int paralign_score_i16(buffer_t* buffer,
                       const submat_t<int16_t> submat,
                       const int16_t gap_open,
                       const int16_t gap_extend,
                       const seq_t seq,
                       const seq_t* refs,
                       const int n_refs,
                       alignment_t<int16_t>** alignments)
{
    return paralign_score(buffer, submat, gap_open, gap_extend, seq, refs, n_refs, alignments);
}
