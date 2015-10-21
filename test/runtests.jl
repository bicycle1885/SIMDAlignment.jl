using SIMDAlignment
using Bio.Seq
using Bio.Align
using Base.Test

function make_submat{score_t}(::Type{score_t})
    submat = Matrix{score_t}(4, 4)
    fill!(submat, -6)
    submat[diagind(submat)] = 0
    return submat
end

function test_same_seqs{score_t}(::Type{score_t})
    submat = make_submat(score_t)
    gap_open = 5
    gap_extend = 3

    seq = seq_t([0x00, 0x01, 0x02, 0x03])
    for n in 1:100
        refs = seq_t[]
        for _ in 1:n
            push!(refs, [0x00, 0x01, 0x02, 0x03])
        end
        alns = paralign_score(submat, gap_open, gap_extend, seq, refs)
        for aln in alns
            @test aln.score == 0
        end
    end
end

function test_empty_seq{score_t}(::Type{score_t})
    seq = dna""
    refs = [
        dna"A",
        dna"AC",
        dna""
    ]
    submat = make_submat(score_t)
    model = AffineGapScoreModel(submat, gap_open_penalty=5, gap_extend_penalty=3)

    refs′ = [convert(seq_t, ref) for ref in refs]
    alns = paralign_score(submat, model.gap_open_penalty, model.gap_extend_penalty, seq, refs′)
    for i in 1:length(refs)
        ref = refs[i]
        aln = alns[i]
        aln′ = pairalign(GlobalAlignment(), seq, ref, model)
        @test aln.score == aln′.score
    end
end

function test_various_seqs{score_t}(::Type{score_t})
    seq = dna"ACGT"
    refs = [
        dna"ACGT",
        dna"ACGG",
        dna"AACGT",
        dna"ACGTT",
        dna"ACATG",
        dna"AACGT",
        dna"AACGTG",
        dna"AACCTG",
        dna"AACCTGA",
        dna"ACG",
        dna"ACT",
        dna"AGT",
        dna"AGG",
        dna"CGT",
        dna"CCT",
        dna"CT",
        dna"C",
        dna"",
        dna"AACCTGA"[2:5],
        dna"AACCTGA"[2:6],
    ]

    submat = make_submat(score_t)
    model = AffineGapScoreModel(submat, gap_open_penalty=5, gap_extend_penalty=3)

    refs′ = [convert(seq_t, ref) for ref in refs]
    alns = paralign_score(submat, model.gap_open_penalty, model.gap_extend_penalty, seq, refs′)

    for i in 1:length(refs)
        ref = refs[i]
        aln = alns[i]
        aln′ = pairalign(GlobalAlignment(), seq, ref, model)
        @test aln.score == aln′.score
    end
end

function test_reversed_seqs{score_t}(::Type{score_t})
    seq = dna"ACGTAT"
    refs = [
        dna"TATGCA",
        dna"ACGTAT",
        dna"ATTGA",
        dna"",
    ]

    submat = make_submat(score_t)
    model = AffineGapScoreModel(submat, gap_open_penalty=5, gap_extend_penalty=3)
    refs′ = [seq_t(ref, true) for ref in refs]
    alns = paralign_score(submat, model.gap_open_penalty, model.gap_extend_penalty, seq, refs′)

    for i in 1:length(refs)
        ref = reverse(refs[i])
        aln = alns[i]
        aln′ = pairalign(GlobalAlignment(), seq, ref, model)
        @test aln.score == aln′.score
    end
end

# run tests
for score_t in (Int8, Int16, Int32)
    test_same_seqs(score_t)
    test_empty_seq(score_t)
    test_various_seqs(score_t)
end
