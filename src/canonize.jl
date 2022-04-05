function fuse_batchnorm(d::Dense, bn::BatchNorm)
    d.σ != identity &&
        throw(ArgumentError("Can't fuse Dense layer with activation $(d.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = scale .* d.weight
    b = scale .* (d.bias - bn.μ) + bn.β
    return Dense(W, b, bn.λ)
end

function fuse_batchnorm(c::Conv, bn::BatchNorm)
    c.σ != identity && throw(ArgumentError("Can't fuse Conv layer with activation $(c.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = c.weight .* reshape(scale, 1, 1, 1, :)
    b = scale .* (c.bias - bn.μ) + bn.β
    return Conv(W, b, bn.λ)
end

"""
    try_fusing(model, i)

Attempt to fuse pair of model layers at indices `i` and `i+1`.
Returns fused model and `true` if layers were fused, unmodified model and `false` otherwise.
"""
function try_fusing(model, i)
    l1 = model[i]
    l2 = model[i + 1]
    if l1 isa Union{Dense,Conv} && l2 isa BatchNorm && activation(l1) == identity
        if i == length(model) - 1
            model = Chain(model[1:(i - 1)]..., fuse_batchnorm(l1, l2))
        end
        model = Chain(model[1:(i - 1)]..., fuse_batchnorm(l1, l2), model[(i + 2):end]...)
        return model, true
    end
    return model, false
end

"""
    canonize(model)

Canonize model by flattening it and fusing BatchNorm layers into preceding Dense and Conv
layers with linear activation functions.
"""
function canonize(model::Chain)
    model = flatten_model(model)
    i = 1
    while i < length(model)
        model, fused = try_fusing(model, i)
        !fused && (i += 1)
    end
    return model
end
