module TullioLRPRulesExt

using ExplainableAI, Flux, Tullio
import ExplainableAI: lrp!, modify_input, modify_denominator
import ExplainableAI: ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule

# Fast implementation for Dense layer using Tullio.jl's einsum notation:
for R in (ZeroRule, EpsilonRule, GammaRule)
    @eval function lrp!(Rᵏ, rule::$R, layer::Dense, modified_layer, aᵏ, Rᵏ⁺¹)
        layer = isnothing(modified_layer) ? layer : modified_layer
        ãᵏ = modify_input(rule, aᵏ)
        z = modify_denominator(rule, layer(ãᵏ))
        @tullio Rᵏ[j, b] = layer.weight[i, j] * ãᵏ[j, b] / z[i, b] * Rᵏ⁺¹[i, b]
    end
end

function lrp!(Rᵏ, ::WSquareRule, _layer::Dense, modified_layer::Dense, aᵏ, Rᵏ⁺¹)
    den = sum(modified_layer.weight; dims=2)
    @tullio Rᵏ[j, b] = modified_layer.weight[i, j] / den[i] * Rᵏ⁺¹[i, b]
end
end # module
