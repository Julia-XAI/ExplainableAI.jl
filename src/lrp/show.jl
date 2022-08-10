# Composites
_range_string(r::LayerRule) = "layer $(r.n)"
_range_string(::GlobalRule) = "all layers"
_range_string(r::RangeRule) = "layers $(r.range)"
_range_string(::FirstLayerRule) = "first layer"
_range_string(::LastLayerRule) = "last layer"
_range_string(r::GlobalTypeRule) = "all layers"
_range_string(r::RangeTypeRule) = "layers $(r.range)"
_range_string(::FirstLayerTypeRule) = "first layer"
_range_string(::LastLayerTypeRule) = "last layer"
_range_string(r::FirstNTypeRule) = "layers $(1:r.n)"
_range_string(r::LastNTypeRule) = "last $(r.n) layers"

function Base.show(io::IO, m::MIME"text/plain", c::Composite)
    println(io, "Composite(")
    for p in c.primitives
        _show_primitive(io, p, 2)
    end
    return println(io, ")")
end

function _show_primitive(io::IO, r::AbstractRulePrimitive, indent::Int=0)
    print(io, " "^indent, nameof(typeof(r)), ": ")
    printstyled(io, _range_string(r); color=:blue)
    print(io, " => ")
    printstyled(io, r.rule; color=:yellow)
    println(io, ",")
    return nothing
end

function _show_primitive(io::IO, r::AbstractCompositePrimitive, indent::Int=0)
    print(io, " "^indent, nameof(typeof(r)), "(")
    printstyled(io, "  # on ", _range_string(r); color=:light_black)
    println(io)
    _print_rules(io, r.map, indent)
    return nothing
end

function _print_rules(io::IO, rs, indent::Int=0)
    for r in rs
        printstyled(io, " "^(indent + 2), r.first; color=:blue)
        print(io, " => ")
        printstyled(io, r.second; color=:yellow)
        println(io, ",")
    end
    println(io, " "^(indent), "),")
    return nothing
end
