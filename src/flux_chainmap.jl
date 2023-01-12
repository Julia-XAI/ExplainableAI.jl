# To support `map` on Flux Chains containing both `Chain` and `Parallel` layers,
# we need a flexible, general purpose container, e.g. Tuple.
# Being able to distinguish between `ChainTuple` and `ParallelTuple` isn't necessary,
# but allows for stricter type checking.
for t in (:ChainTuple, :ParallelTuple)
    @eval begin
        """
            $(string($t))(xs)

        Thin wrapper around `Tuple` for use with Flux.jl models.

        When combined, [`ChainTuple`](@ref) and [`ParallelTuple`](@ref) can be used
        to store data while preserving the structure of a Flux model
        using `Chain` and `Parallel` without risk of type piracy.

        See also [`chainmap`](@ref).
        """
        struct ($t){T<:Tuple}
            vals::T
        end
        ($t)(xs...) = ($t)(xs)

        # Foward Base functions to wrapped Tuple `vals`:
        @forward ($t).vals Base.getindex,
        Base.length, Base.first, Base.last, Base.iterate, Base.lastindex, Base.keys,
        Base.firstindex

        Base.show(io::IO, m::MIME"text/plain", t::($t)) = _show_tuple(io, t, 0)
        function _show_tuple(io::IO, t::($t), indent::Int)
            println(io, " "^indent, string($t), "(")
            for x in t
                _show_tuple(io, x, indent + 2)
            end
            println(io, " "^indent, ")")
        end
    end
end
_show_tuple(io::IO, layer, indent::Int) = println(io, " "^indent, layer)

"""
    chainmap(f, model)

`map` for Flux `Chains`. Applies the function `f` to all layers in a Flux model,
returning a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.
"""
chainmap(f, c::Chain) = ChainTuple(chainmap.(f, c.layers)...)
chainmap(f, p::Parallel) = ParallelTuple(chainmap.(f, p.layers)...)
chainmap(f, layer) = f(layer)

# chainmap can be re-applied on results:
chainmap(f, c::ChainTuple) = ChainTuple(chainmap.(f, c.vals)...)
chainmap(f, p::ParallelTuple) = ChainTuple(chainmap.(f, p.vals)...)

"""
    heat_tail(xs)

Split input into head and tail.

## Examples
```julia-repl
julia> head_tail(1, 2, 3, 4)
(1, (2, 3, 4))

julia> head_tail((1, 2, 3, 4))
(1, (2, 3, 4))

julia> head_tail([1, 2, 3, 4])
(1, (2, 3, 4))

julia> head_tail(1, (2, 3), 4)
(1, ((2, 3), 4))

julia> head_tail(1)
(1, ())

julia> head_tail()
()
```
"""
head_tail(h, t...) = h, t
head_tail(h, t) = h, t
head_tail() = ()
head_tail(xs::Tuple) = head_tail(xs...)
head_tail(xs::AbstractVector) = head_tail(xs...)
