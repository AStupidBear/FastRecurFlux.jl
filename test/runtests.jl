using FastRecurFlux
using Flux
using Random
using Statistics
using Test

@testset "FastRecurFlux.jl" begin
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
        opt = ADAM(1e-3, (0.9, 0.999))
        cb = () -> Flux.reset!(model)
        data = repeat([(x, y)], 100)
        Flux.@epochs 10 Flux.train!(loss, params(model), data, opt, cb = cb)
        @test loss(x, y) < 0.01
    end
end
