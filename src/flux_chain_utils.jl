#============================#
# ChainTuple & ParallelTuple #
#============================#

# To support map and zip on Flux Chains containing both `Chain` and `Parallel` layers,
# we need a flexible, general purpose container, e.g. a Tuple.
# We opt to introduce `ChainTuple` and `ParallelTuple` instead of `Chain` and `Parallel`
# to avoid type piracy.

for S in (:ChainTuple, :ParallelTuple)

    name = string(S)

    @eval begin
        """
            $($name)(xs)

        Thin wrapper around `Tuple` for use with Flux.jl models.

        Combining [`ChainTuple`](@ref) and [`ParallelTuple`](@ref),
        data `xs` can be stored while preserving the structure of a Flux model
        without risking type piracy.
        """
        struct ($S){T<:Tuple}
            vals::T
        end
        ($S)(xs...) = ($S)(xs)

        @forward $S.vals Base.getindex,
        Base.length,
        Base.first,
        Base.last,
        Base.iterate,
        Base.lastindex,
        Base.keys,
        Base.firstindex,
        Base.:(==)

        # Containers are equivalent if fields are equivalent
        Base.:(==)(a::$S, b::$S) = a.vals == b.vals

        # Print vals
        Base.show(io::IO, m::MIME"text/plain", t::$S) = print_vals(io, t)

        function print_vals(io::IO, t::$S, indent::Int=0)
            println(io, " "^indent, "$($name)(")
            for x in t
                print_vals(io, x, indent + 2)
            end
            println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
        end
    end
end
print_vals(io::IO, x, indent::Int=0) = println(io, " "^indent, x, ",")

#=====================#
# chainmap & chainzip #
#=====================#

# The following implementation of map and zip on Chains and Parallel layers
# are strongly inspired by StructWalk.jl's postwalk function.

isleaf(c::Chain) = false
isleaf(p::Parallel) = false
isleaf(c::ChainTuple) = false
isleaf(p::ParallelTuple) = false
isleaf(x) = true

constructor(::Chain) = ChainTuple
constructor(::ChainTuple) = ChainTuple
constructor(::Parallel) = ParallelTuple
constructor(::ParallelTuple) = ParallelTuple

children(c::Chain) = c.layers
children(p::Parallel) = p.layers
children(c::ChainTuple) = c.vals
children(p::ParallelTuple) = p.vals

"""
    chainmap(f, x)

`map` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainzip`](@ref).
"""
function chainmap(f, x)
    if isleaf(x)
        return f(x)
    else
        T = constructor(x)
        vals = chainmap.(f, children(x))
        return T(vals...)
    end
end

"""
    chainzip(f, a, b)

`zip` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers. Assumes that `a` and `b` have the same structure.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainmap`](@ref).
"""
function chainzip(f, a, b)
    if isleaf(a) && isleaf(b)
        return f(a, b)
    else
        T = constructor(a)
        # Assume that a and b are zippable if constructors match:
        T != constructor(b) && error("Cannot chainzip $a and $b.")
        vals = chainzip.(f, children(a), children(b))
        return T(vals...)
    end
end

"""
    id_list(model)

Return list of `objectid`s of all layers in the model.
"""
function id_list(model::Chain)
    ids = chainmap(objectid, model)
    idlist = UInt64[]
    push_id!(idlist, ids)
    return idlist
end

function push_id!(idlist, x)
    if isleaf(x)
        push!(idlist, x)
    else
        for y in children(x)
            push_id!(idlist, y)
        end
    end
end


#===============#
# Flatten model #
#===============#
"""
     flatten_model(c)

 Flatten a Flux chain containing Flux chains.
 """
 flatten_model(c::Chain) = Chain(mapreduce(_flatten_model, vcat, c.layers)...)
 _flatten_model(c::Chain) =  mapreduce(_flatten_model, vcat, c.layers)

 flatten_model(x) = x
 _flatten_model(x) = x

 flatten_model(p::Parallel) = Parallel(p.connection, flatten_model.(p.layers))
 _flatten_model(p::Parallel) = Parallel(p.connection, flatten_model.(p.layers))

#=========================#
# Strip output activation #
#=========================#
"""
    first_element(model)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
first_element(c::Union{Chain,ChainTuple}) = first_element(c[1])
first_element(layer) = layer

"""
    last_element(model)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
last_element(c::Union{Chain,ChainTuple}) = last_element(c[end])
last_element(layer) = layer

"""
  check_output_softmax(model)

Check whether model has softmax activation on output.
Return the model if it doesn't, throw error otherwise.
"""
function check_output_softmax(model::Chain)
    if has_output_softmax(model)
        throw(ArgumentError("""Model contains softmax activation function on output.
        Call `strip_softmax` on your model."""))
    end
    return model
end

has_output_softmax(model::Chain) = has_output_softmax(last_element(model))
has_output_softmax(x) = is_softmax(x) || is_softmax(activation_fn(x))
is_softmax(x) = x isa SoftmaxActivation

"""
    strip_softmax(model)
    strip_softmax(layer)

Remove softmax activation on layer or model if it exists.
"""
strip_softmax(l) = copy_layer(l, l.weight, l.bias; σ=identity)
strip_softmax(::SoftmaxActivation) = identity

function strip_softmax(model::Chain)
    output_layer = last_element(model)
    !has_output_softmax(output_layer) && return model

    function _strip_softmax(layer)
        layer != output_layer && return layer
        return strip_softmax(layer)
    end
    _strip_softmax(c::Chain) = Chain(_strip_softmax.(c.layers)...)
    _strip_softmax(p::Parallel) = p # p.connection can't be softmax
    return _strip_softmax(model)
end
