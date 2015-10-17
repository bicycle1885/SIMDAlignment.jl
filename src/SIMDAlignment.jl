module SIMDAlignment

export
    # wrapper types
    seq_t,
    submat_t,
    alignment_t,
    # functions
    paralign_score

using Bio.Seq

const libsimdalign = Pkg.dir("SIMDAlignment", "deps", "libsimdalign.so")


# 1-byte alphabets
typealias Byte Union{Int8,UInt8,DNANucleotide,RNANucleotide,AminoAcid}


# sequence
immutable seq_t
    data::Ptr{UInt8}
    startpos::Csize_t
    len::Csize_t
    packed::Bool
end

function Base.convert{T<:Byte}(::Type{seq_t}, seq::Vector{T})
    return seq_t(pointer(seq), 1, length(seq), false)
end

function Base.convert{T}(::Type{seq_t}, seq::NucleotideSequence{T})
    # TODO: optimize
    byteseq = convert(Vector{T}, seq)
    return seq_t(byteseq)
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

typealias score_t Int16


function make_buffer()
    buffer = ccall((:make_buffer, libsimdalign), Ptr{Void}, ())
    @assert buffer != C_NULL "failed to allocate a buffer"
    return buffer
end

function free_buffer(buffer)
    ccall((:free_buffer, libsimdalign), Void, (Ptr{Void},), buffer)
end

function paralign_score(submat, gap_open, gap_extend, seq, refs)
    alns = Vector{alignment_t{score_t}}()
    for _ in 1:length(refs)
        push!(alns, alignment_t{score_t}())
    end
    buffer = make_buffer()
    ret = ccall(
        (:paralign_score, libsimdalign),
        Cint,
        (Ptr{Void}, submat_t{score_t}, score_t, score_t, seq_t, Ptr{seq_t}, Cint, Ptr{Void}),
        buffer, submat_t(submat), gap_open, gap_extend, seq_t(seq), pointer(refs), length(refs), alns
    )
    free_buffer(buffer)
    @assert ret == 0 "failed to align"
    return alns
end

end # module
