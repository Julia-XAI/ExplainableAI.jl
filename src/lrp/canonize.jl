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

is_fuseable(l::Union{Dense,Conv}, bn::BatchNorm) = activation_fn(l) == identity
is_fuseable(l1, l2) = false

"""
    try_fusing(model, i)

Attempt to fuse pair of model layers at indices `i` and `i+1`.
Returns fused model and `true` if layers were fused, unmodified model and `false` otherwise.
"""
function try_fusing(model, i)
    l1, l2 = model[i:(i + 1)]

    if is_fuseable(l1, l2)
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
    return _canonize(model)
end

function _canonize(model::Chain)
    model = Chain(_canonize.(model.layers)) # recursively canonize Parallel layers

    i = 1
    while i < length(model)
        l1, l2 = model[i:(i + 1)]

        if is_fuseable(l1, l2)
            fused = fuse_batchnorm(l1, l2)
            model = Chain(model[1:(i - 1)]..., fused, model[(i + 2):end]...)
            # if fused, don't increment i,
            # instead try fusing the new layer with the next one
        else
            i += 1
        end
    end
    return model
end

function _canonize(p::Parallel)
    return Parallel(p.connection, _canonize.(p.layers))
end

_canonize(layer) = layer
