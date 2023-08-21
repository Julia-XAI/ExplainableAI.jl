"""
    activation_fn(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation_fn(layer) = nothing
activation_fn(l::Dense)         = l.σ
activation_fn(l::Conv)          = l.σ
activation_fn(l::CrossCor)      = l.σ
activation_fn(l::ConvTranspose) = l.σ
activation_fn(l::BatchNorm)     = l.λ

has_weight(layer) = hasproperty(layer, :weight)
has_bias(layer) = hasproperty(layer, :bias)
has_weight_and_bias(layer) = has_weight(layer) && has_bias(layer)

"""
    copy_layer(layer, W, b, [σ=identity])

Copy layer using weights `W` and `b`. The activation function `σ` can also be set,
defaulting to `identity`.
"""
copy_layer(::Dense, W, b; σ=identity) = Dense(W, b, σ)
copy_layer(l::Conv, W, b; σ=identity) = Conv(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
function copy_layer(l::ConvTranspose, W, b; σ=identity)
    return ConvTranspose(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
end
function copy_layer(l::CrossCor, W, b; σ=identity)
    return CrossCor(σ, W, b, l.stride, l.pad, l.dilation)
end
