module FastRecurFlux

using Flux

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
    gx, gh = m.Wi*x, m.Wh*h
    r = σ.((r_ = gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1);))
    z = σ.((z_ = gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2));)
    h̃ = tanh.((h̃_ = gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3);))
    h′ = (1 .- z).*h̃ .+ z.*h
    return h′, h′
end

end