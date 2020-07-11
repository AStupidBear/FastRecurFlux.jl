LoopVectorization.vmap(f, x::Tracker.TrackedArray) = Tracker.track(vmap, f, x)

drelu(x, Δ) = ifelse(x > 0, Δ, zero(x))

Tracker.@grad function Base.broadcasted(::typeof(relu), x::Tracker.TrackedArray)
    relu.(x), Δ -> (nothing, drelu.(x, Δ))
end

Tracker.@grad function vmap(::typeof(σ), x)
    y = vmap(σ, Tracker.data(x))
    y, Δ -> (nothing, Δ .* (y .* (1 .- y)))
end

Tracker.@grad function vmap(::typeof(tanh), x)
    y = vmap(tanh, Tracker.data(x))
    y, Δ -> (nothing, Δ .* (1 .- y.^2))
end