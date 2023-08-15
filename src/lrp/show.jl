const COLOR_COMMENT = :light_black
const COLOR_RULE    = :yellow
const COLOR_TYPE    = :light_blue
const COLOR_RANGE   = :green

typename(x) = string(nameof(typeof(x)))

#==============#
# LRP analyzer #
#==============#

layer_name(io::IO, l) = string(sprint(show, l; context=io))
function get_print_rule_padding(names::Union{ChainTuple, ParallelTuple})
    children = filter(isleaf, names.vals)
    isempty(children) && return 0
    return maximum(length.(children))
end
function Base.show(io::IO, m::MIME"text/plain", lrp::LRP)
    layer_names = chainmap(Base.Fix1(layer_name, io), lrp.model)
    npad = get_print_rule_padding(layer_names)

    println(io, "LRP", "(")
    for (rule, name) in zip(lrp.rules, layer_names)
        print_rule(io, rule, name, 1, npad)
    end
    println(io, ")")
end

for T in (:ChainTuple, :ParallelTuple)
    name = string(T)
    @eval begin
        function print_rule(io::IO, rules::$T, names::$T, indent::Int=0, npad::Int=0)
            npad = get_print_rule_padding(names)
            println(io, "  "^indent, $name, "(")
            for (r, l) in zip(rules.vals, names.vals)
                print_rule(io, r, l, indent + 1, npad)
            end
            println(io, "  "^indent, "),")
        end
    end # eval
end

function print_rule(io::IO, rule, name, indent::Int=0, npad::Int=0)
    print(io, "  "^indent, rpad(name, npad), " => ")
    printstyled(io, rule; color=COLOR_RULE)
    println(io, ",")
end

#===========#
# Composite #
#===========#

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
