"""
Print summary of Flux chain.
"""
function model_summary(chain::Chain)
    idxs = collect(1:length(chain.layers))
    layers = chain.layers
    sizes = [size.(Flux.params(layer)) for layer in chain.layers]

    println(idxs, typeof(layers), typeof(sizes))
    header = ["Index", "Layer", "Parameter sizes"]
    data = hcat(idxs, layers, sizes)
    pretty_table(data, header)
    return nothing
end

# helper function to work around Flux.Zeros
function get_weights(layer)
    W = layer.weight
    if typeof(layer.bias) <: Flux.Zeros
        b = zeros(eltype(W), size(W, 1))
    else
        b = layer.bias
    end
    return W, b
end

"""
    stabilize_denom(d; eps = 1f-6)

Replace zero terms of a matrix `d` with `eps`.
"""
function stabilize_denom(d; eps = 1f-9)
    return d + eps * (d .≈ 0)
end

"""
    safedivide(a, b; eps = 1f-6)

Elementwise division of two matrices avoiding zero terms
in the denominator by replacing them with `eps`.
"""
function safedivide(a, b; eps = 1f-9)
    return a ./ stabilize_denom(b; eps=eps)
end
