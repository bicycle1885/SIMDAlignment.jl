using SIMDAlignment
using Base.Test

let
    submat = Int16[
         0 -1 -1 -1;
        -1  0 -1 -1;
        -1 -1  0 -1;
        -1 -1 -1  0;
    ]
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
