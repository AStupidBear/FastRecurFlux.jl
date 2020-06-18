LoopVectorization.vmap(f, x::ReverseDiff.TrackedArray) = ReverseDiff.track(vmap, f, x)

ReverseDiff.@grad function vmap(f::typeof(σ), x)
    y = vmap(σ, ReverseDiff.value(x))
    y, Δ -> (nothing, Δ .* (y .* (1 .- y)))
end

ReverseDiff.@grad function vmap(f::typeof(tanh), x)
    y = vmap(tanh, ReverseDiff.value(x))
    y, Δ -> (nothing, Δ .* (1 .- y.^2))
end
