using SIMDAlignment
using Bio.Seq
using Bio.Align
using Base.Test

for score_t in (Int8, Int16, Int32)
    submat = Matrix{score_t}(4, 4)
    fill!(submat, -6)
    submat[diagind(submat)] = 0
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
    ]

    affinegap = AffineGapScoreModel(submat, gap_open_penalty=gap_open, gap_extend_penalty=gap_extend)

    refs′ = [convert(seq_t, ref) for ref in refs]
    alns = paralign_score(submat, gap_open, gap_extend, seq, refs′)

    for i in 1:length(refs)
        ref = refs[i]
        aln = alns[i]
        aln′ = pairalign(GlobalAlignment(), seq, ref, affinegap)
        @test aln.score == aln′.score
    end
end
