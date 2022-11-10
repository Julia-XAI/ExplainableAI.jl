const COLOR_COMMENT = :light_black
const COLOR_RULE    = :yellow
const COLOR_TYPE    = :blue
const COLOR_RANGE   = :green

typename(x) = string(nameof(typeof(x)))

################
# LRP analyzer #
################

_print_layer(io::IO, l) = string(sprint(show, l; context=io))
function Base.show(io::IO, m::MIME"text/plain", analyzer::LRP)
    layer_names = [_print_layer(io, l) for l in analyzer.model]
    rs = rules(analyzer)
    npad = maximum(length.(layer_names)) + 1 # padding to align rules with rpad

    println(io, "LRP", "(")
    for (r, l) in zip(rs, layer_names)
        print(io, "  ", rpad(l, npad), " => ")
        printstyled(io, r; color=COLOR_RULE)
        println(io, ",")
    end
    println(io, ")")
end

#############
# Composite #
#############

_range_string(r::LayerRule)         = "layer $(r.n)"
_range_string(::GlobalRule)         = "all layers"
_range_string(r::RangeRule)         = "layers $(r.range)"
_range_string(::FirstLayerRule)     = "first layer"
_range_string(::LastLayerRule)      = "last layer"
_range_string(r::GlobalTypeRule)    = "all layers"
_range_string(r::RangeTypeRule)     = "layers $(r.range)"
_range_string(::FirstLayerTypeRule) = "first layer"
_range_string(::LastLayerTypeRule)  = "last layer"
_range_string(r::FirstNTypeRule)    = "layers $(1:r.n)"
_range_string(r::LastNTypeRule)     = "last $(r.n) layers"

function Base.show(io::IO, m::MIME"text/plain", c::Composite)
    println(io, "Composite", "(")
    for p in c.primitives
        _show_primitive(io, p, 2)
    end
    println(io, ")")
end

function _show_primitive(io::IO, r::AbstractRulePrimitive, indent::Int=0)
    print(io, " "^indent, typename(r), ": ")
    printstyled(io, _range_string(r); color=COLOR_RANGE)
    print(io, " => ")
    printstyled(io, r.rule; color=COLOR_RULE)
    println(io, ",")
end

function _show_primitive(io::IO, r::AbstractTypeRulePrimitive, indent::Int=0)
    print(io, " "^indent, rpad(typename(r) * "(", 20))
    printstyled(io, "# on ", _range_string(r); color=COLOR_COMMENT)
    println(io)
    _print_rules(io, r.map, indent)
    println(io, " "^(indent), "),")
end

function _print_rules(io::IO, rs, indent::Int=0)
    for r in rs
        print(io, " "^(indent + 2))
        _print_type_rule(io, r)
        println(io, ",")
    end
end

function _print_type_rule(io::IO, r::TypeRulePair)
    printstyled(io, r.first; color=COLOR_TYPE)
    print(io, " => ")
    printstyled(io, r.second; color=COLOR_RULE)
end
