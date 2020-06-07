module FastRecurFlux

using Flux, LoopVectorization, Requires

using Flux: LSTMCell, GRUCell, gate

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

for f in (:σ, :tanh)
    @eval Base.broadcasted(::typeof($f), x::AbstractArray{T, N}) where {T <: Union{Float64, Float32}, N} = vmap($f, x)
end

function __init__()
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("tracker.jl")
end

end