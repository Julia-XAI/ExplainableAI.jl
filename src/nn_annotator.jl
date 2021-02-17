"""
Annotate layers in the chain by mapping them to one of the following symbols:
* `:FirstPixels`
* `:FirstReals`
* `:Lower`
* `:Middle`
* `:Upper`
* `:Top`
* `:TopSoftmax`
"""
function annotate_chain(
    chain::Chain,
    middle_start::Integer,
    upper_start::Integer;
    input_type::String="pixels"
)
    n_layers = length(chain)

    function annotate_layer(i)::Symbol
        # Input layer
        if i == 1
            if input_type == "pixels"
                return :FirstPixels
            elseif input_type == "reals"
                return :FirstReals
            else
                throw(ArgumentError("""Unknown input type "$(input_type)". Expected "pixels" or "reals"."""))
            end
        # Lower layers
        elseif i < middle_start
            return :Lower
        # Middle layers
        elseif i < upper_start
            return :Middle
        # Upper layers
        elseif i < n_layers
            return :Upper
        # Output layer
        elseif i == n_layers
            if chain[n_layers] == softmax
                return :TopSoftmax
            else
                return :Top
            end
        else
            throw(DimensionError("_annotate_layer called outside of chain length."))
        end
    end

    return annotations = ntuple(annotate_layer, Val(n_layers))
end
