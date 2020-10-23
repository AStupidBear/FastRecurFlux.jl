module FastRecurFlux

using LinearAlgebra
using Flux
using LoopVectorization
using PaddedMatrices
using Requires

using Flux: LSTMCell, GRUCell, gate

SubMatrix{T} = SubArray{T, 2, <: AbstractArray{T}, Tuple{Vararg{AbstractUnitRange, N} where N}, true}
MatTypesC{T} = Union{Matrix{T}, SubMatrix{T}}
MatTypesR{T} = Union{Adjoint{T, <: MatTypesC{T}}, Transpose{T, <: MatTypesC{T}}}
MatTypes{T} = Union{MatTypesC{T}, MatTypesR{T}}
SubVector{T} = SubArray{T, 1, <: AbstractArray{T}, Tuple{Vararg{AbstractUnitRange, N} where N}, true}
VecTypes{T} = Union{Vector{T}, SubVector{T}}

function mul(A::AbstractArray{T}, B::AbstractArray{T}) where T
    M, N, K = size(A, 1), size(A, 2), size(B, 2)
    C = Matrix{T}(undef, M, K)
    if M * N * K < 256^3
        PaddedMatrices.jmul!(C, A, B)
    else
        LinearAlgebra.mul!(C, A, B)
    end
    return C
end

@static if Sys.ARCH == :x86_64
    Base.:*(A::TA, B::TB) where {T <: AbstractFloat, TA <: Matrix{T}, TB <: Matrix{T}} = mul(A, B)
    Base.:*(A::TA, B::TB) where {T <: AbstractFloat, TA <: MatTypes{T}, TB <: MatTypes{T}} = mul(A, B)
    Base.:*(A::TA, B::TB) where {T <: AbstractFloat, TA <: MatTypes{T}, TB <: VecTypes{T}} = vec(mul(A, reshape(B, :, 1)))
end

function (m::LSTMCell)((h, c), x)
    b, o = m.b, size(h, 1)
    g = m.Wi * x .+ m.Wh * h .+ b
    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* (c_ = tanh.(c))
    return (h′, c), h′
end

function (m::GRUCell)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = σ.((r_ = gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1);))
    z = σ.((z_ = gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2);))
    h̃ = tanh.((h̃_ = gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3);))
    h′ = (1 .- z) .* h̃ .+ z .* h
    return h′, h′
end

function Flux.NNlib.σ(x::LoopVectorization.SLEEFPirates.FloatType)
    t = exp(-abs(x))
    LoopVectorization.vifelse(x ≥ 0, inv(one(t) + t), t / (one(t) + t))
end

function __init__()
    for f in (:σ, :tanh)
        @eval Base.broadcasted(::typeof($f), x::AbstractArray{T, N}) where {T, N} = vmap($f, x)
    end
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("tracker.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("reversediff.jl")
end

end