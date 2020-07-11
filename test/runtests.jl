using FastRecurFlux
using Flux
using Random
using Statistics
using Test

subarray(x::AbstractArray{T, N}) where {T, N} = view(x, ntuple(i -> :, N)...)

@testset "FastRecurFlux.jl" begin
    M, V = rand(10, 10), rand(10, 1)
    for lsub in (true, false), ltrans in (true, false), 
        rsub in (true, false), rtrans in (true, false),
        subfirst in (true, false), rmat in (true, false)
        M′, V′ = M, rmat ? V : vec(V)
        if subfirst
            M′ = lsub ? subarray(M) : M
            V′ = rsub ? subarray(V) : V
            M′ = ltrans ? M′' : M′
            V′ = rtrans ? V′' : V
        else
            M′ = ltrans ? M′' : M′
            V′ = rtrans ? V′' : V
            M′ = lsub ? subarray(M) : M
            V′ = rsub ? subarray(V) : V
        end
        if size(M′, 2) != size(V′, 1)
            M′ = reshape(M′, :, size(V′, 1))
        end
        @test M′ * V′ ≈ Array(M′) * Array(V′)
    end
    Random.seed!(1234)
    x = randn(Float32, 10, 1, 5)
    y = mean(x, dims = 1)
    for layer in (GRU, LSTM)
        model = Chain(layer(10, 100), layer(100, 1))
        function loss(x, y)
            xs = Flux.unstack(x, 3)
            ys = Flux.unstack(y, 3)
            l = 0f0
            for t in 1:length(xs)
                ŷ = model(xs[t])
                l += Flux.mse(ys[t], ŷ)
            end
            return l / length(xs)
        end
        ps = Flux.params(model)
        data = repeat([(x, y)], 100)
        opt = ADAM(1e-3, (0.9, 0.999))
        cb = () -> Flux.reset!(model)
        Flux.@epochs 10 Flux.train!(loss, ps, data, opt, cb = cb)
        @test loss(x, y) < 0.01
    end
end
