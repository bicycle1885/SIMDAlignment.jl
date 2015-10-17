module SIMDAlignment

export
    Seq,
    SubMat,
    Alignment,
    paralign_score

const libsimdalign = Pkg.dir("SIMDAlignment", "deps", "libsimdalign.so")


# sequence
immutable Seq
    data::Ptr{UInt8}
    startpos::Csize_t
    len::Csize_t
    packed::Bool
end

function Base.convert(::Type{Seq}, seq::Vector{UInt8})
    return Seq(pointer(seq), 1, length(seq), false)
end


# substitution matrix
immutable SubMat{T}
    data::Ptr{T}
    size::Cint
end

function Base.convert{T}(::Type{SubMat}, submat::Matrix{T})
    m, n = size(submat)
    @assert m == n
    return SubMat{T}(pointer(submat), convert(Cint, n))
end


# alignment result
type Alignment{T}
    score::T
    trace::Ptr{UInt8}
    seqlen::Csize_t
    reflen::Csize_t
    endpos_seq::Csize_t
    endpos_ref::Csize_t
    function Alignment()
        return new(0, C_NULL, 0, 0, 0, 0)
    end
end

typealias Score Int16


function make_buffer()
    buffer = ccall((:make_buffer, libsimdalign), Ptr{Void}, ())
    @assert buffer != C_NULL "failed to allocate a buffer"
    return buffer
end

function free_buffer(buffer)
    ccall((:free_buffer, libsimdalign), Void, (Ptr{Void},), buffer)
end

function paralign_score(submat, gap_open, gap_extend, seq, refs)
    alns = Vector{Alignment{Score}}()
    for _ in 1:length(refs)
        push!(alns, Alignment{Score}())
    end
    buffer = make_buffer()
    ret = ccall(
        (:paralign_score, libsimdalign),
        Cint,
        (Ptr{Void}, SubMat{Score}, Score, Score, Seq, Ptr{Seq}, Cint, Ptr{Void}),
        buffer, SubMat(submat), gap_open, gap_extend, Seq(seq), pointer(refs), length(refs), alns
    )
    free_buffer(buffer)
    @assert ret == 0 "failed to align"
    return alns
end

end # module
