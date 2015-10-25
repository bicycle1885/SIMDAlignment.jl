module SIMDAlignment

export
    # wrapper types
    seq_t,
    submat_t,
    alignment_t,
    # functions
    paralign_score

using Bio.Seq
using Bio.Align

# 1-byte alphabets
typealias Byte Union{Int8,UInt8,DNANucleotide,RNANucleotide,AminoAcid}

# sequence
immutable seq_t
    data::Ptr{UInt8}
    len::Csize_t
    offset::Csize_t
    reversed::Bool
    packed::Bool
end

function Base.convert{T<:Byte}(::Type{seq_t}, seq::Vector{T})
    return seq_t(pointer(seq), length(seq), 0, false, false)
end

function Base.call(::Type{seq_t}, seq::Vector, offset::Int=0, reversed::Bool=false)
    return seq_t(pointer(seq), length(seq), offset, reversed, false)
end

function Base.convert(::Type{seq_t}, seq::NucleotideSequence)
    byteseq = reinterpret(UInt8, seq.data)
    return seq_t(pointer(byteseq), length(seq), seq.part.start - 1, false, true)
end

function Base.call{T}(::Type{seq_t}, seq::NucleotideSequence{T}, reversed::Bool=false)
    byteseq = reinterpret(UInt8, seq.data)
    len = length(seq)
    if len == 0
        offset = 0
    else
        offset = (reversed ? seq.part.stop : seq.part.start) - 1
    end
    return seq_t(pointer(byteseq), length(seq), offset, reversed, true)
end

# substitution matrix
immutable submat_t{T}
    data::Ptr{T}
    size::Cint
end

function Base.convert{T}(::Type{submat_t}, submat::Matrix{T})
    m, n = size(submat)
    @assert m == n
    return submat_t{T}(pointer(submat), convert(Cint, n))
end

function Base.convert{T}(::Type{submat_t}, submat::SubstitutionMatrix{T})
    return submat_t(submat.data)
end

# alignment result
type alignment_t{T}
    score::T
    trace::Ptr{UInt8}
    seqlen::Csize_t
    reflen::Csize_t
    endpos_seq::Csize_t
    endpos_ref::Csize_t
    function alignment_t()
        return new(0, C_NULL, 0, 0, 0, 0)
    end
end

function Base.show(io::IO, aln::alignment_t)
    print(io, aln.score)
end


const libsimdalign = Pkg.dir("SIMDAlignment", "deps", "libsimdalign.so")

function make_buffer()
    buffer = ccall((:make_buffer, libsimdalign), Ptr{Void}, ())
    @assert buffer != C_NULL "failed to allocate a buffer"
    return buffer
end

function free_buffer(buffer)
    ccall((:free_buffer, libsimdalign), Void, (Ptr{Void},), buffer)
end

@generated function paralign_score{score_t}(submat::Matrix{score_t}, gap_open::score_t, gap_extend::score_t, seq::seq_t, refs::Vector{seq_t})
    func = score_t === Int8  ? :(:paralign_score_i8x32)  :
           score_t === Int16 ? :(:paralign_score_i16x16) :
           score_t === Int32 ? :(:paralign_score_i32x8) :
           error("not supported type: $score_t")
    quote
        alns = Vector{alignment_t{score_t}}()
        for _ in 1:length(refs)
            push!(alns, alignment_t{score_t}())
        end
        buffer = make_buffer()
        ret = ccall(
            ($(func), libsimdalign),
            Cint,
            (Ptr{Void}, submat_t{score_t}, score_t, score_t, seq_t, Ptr{seq_t}, Cint, Ptr{Void}),
            buffer, submat_t(submat), gap_open, gap_extend, seq, pointer(refs), length(refs), alns
        )
        free_buffer(buffer)
        @assert ret == 0 "failed to align"
        return alns
    end
end

function paralign_score{score_t}(submat::Union{Matrix{score_t},SubstitutionMatrix{score_t}}, gap_open, gap_extend, seq, refs)
    paralign_score(
        convert(Matrix{score_t}, submat),
        score_t(gap_open),
        score_t(gap_extend),
        seq_t(seq),
        [seq_t(ref) for ref in refs]
    )
end

end # module
