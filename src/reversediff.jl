LoopVectorization.vmap(f, x::ReverseDiff.TrackedArray) = ReverseDiff.track(vmap, f, x)

drelu(x, Δ) = ifelse(x > 0, Δ, zero(x))

ReverseDiff.@grad function Base.broadcasted(f::typeof(relu), x::ReverseDiff.TrackedArray)
    relu.(x), Δ -> (nothing, drelu.(x, Δ))
end

ReverseDiff.@grad function vmap(f::typeof(σ), x)
    y = vmap(σ, ReverseDiff.value(x))
    y, Δ -> (nothing, Δ .* (y .* (1 .- y)))
end

ReverseDiff.@grad function vmap(f::typeof(tanh), x)
    y = vmap(tanh, ReverseDiff.value(x))
    y, Δ -> (nothing, Δ .* (1 .- y.^2))
end
