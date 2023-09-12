function fuse_batchnorm(d::Dense, bn::BatchNorm)
    d.σ != identity &&
        throw(ArgumentError("Can't fuse Dense layer with activation $(d.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = scale .* d.weight
    b = if d.bias != false
        scale .* (d.bias - bn.μ) + bn.β
    else
        -scale .* bn.μ + bn.β
    end
    return Dense(W, b, bn.λ)
end

function fuse_batchnorm(c::Conv, bn::BatchNorm)
    c.σ != identity && throw(ArgumentError("Can't fuse Conv layer with activation $(c.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = c.weight .* reshape(scale, 1, 1, 1, :)
    b = if c.bias != false
        scale .* (c.bias - bn.μ) + bn.β
    else
        -scale .* bn.μ + bn.β
    end
    return Conv(bn.λ, W, b, c.stride, c.pad, c.dilation, c.groups)
end

is_fuseable(l::Union{Dense,Conv}, bn::BatchNorm) = activation_fn(l) == identity
is_fuseable(l1, l2) = false

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
