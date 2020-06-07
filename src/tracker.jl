LoopVectorization.vmap(f, x::Tracker.TrackedArray) = Tracker.track(vmap, f, x)

Tracker.@grad function vmap(::typeof(σ), x)
    y = vmap(σ, Flux.data(x))
    y, Δ -> (nothing, Δ .* (y .* (1 .- y)))
end

Tracker.@grad function vmap(::typeof(tanh), x)
    y = vmap(tanh, Flux.data(x))
    y, Δ -> (nothing, Δ .* (1 .- y.^2))
end