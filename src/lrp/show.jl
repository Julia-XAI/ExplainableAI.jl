const COLOR_COMMENT    = :light_black
const COLOR_ARROW      = :light_black
const COLOR_RULE       = :yellow
const COLOR_TYPE       = :light_blue
const COLOR_RANGE      = :green
const COLOR_CHECK_PASS = :green
const COLOR_CHECK_FAIL = :red

typename(x) = string(nameof(typeof(x)))

#==============#
# LRP analyzer #
#==============#

layer_name(io::IO, l) = string(sprint(show, l; context=io))

function get_name_padding(names::Union{ChainTuple,ParallelTuple})
    children = filter(isleaf, names.vals)
    isempty(children) && return 0
    return maximum(length.(children))
end
function Base.show(io::IO, m::MIME"text/plain", lrp::LRP)
    layer_names = chainmap(Base.Fix1(layer_name, io), lrp.model)
    npad = get_name_padding(layer_names)

    println(io, "LRP", "(")
    for (name, rule) in zip(layer_names, lrp.rules)
        print_rule(io, name, rule, 1, npad)
    end
    print(io, ")")
end

for T in (:ChainTuple, :ParallelTuple)
    tuple_name = string(T)
    @eval begin
        function print_rule(io::IO, names::$T, rules::$T, indent::Int=0, npad::Int=0)
            npad = get_name_padding(names)
            println(io, "  "^indent, $tuple_name, "(")
            for (name, rule) in zip(children(names), children(rules))
                print_rule(io, name, rule, indent + 1, npad)
            end
            println(io, "  "^indent, "),")
        end
    end # eval
end

function print_rule(io::IO, name, rule, indent::Int=0, npad::Int=0)
    print(io, "  "^indent, rpad(name, npad))
    printstyled(io, " => "; color=COLOR_ARROW)
    printstyled(io, rule; color=COLOR_RULE)
    println(io, ",")
end

#=============================#
# Print result of model check #
#=============================#

function print_lrp_model_check(io::IO, model)
    layer_names = chainmap(Base.Fix1(layer_name, io), model)
    npad = get_name_padding(layer_names)
    print_lrp_model_check(io, model, layer_names, 1, npad)
end

for T in (:ChainTuple, :ParallelTuple)
    tuple_name = string(T)
    @eval begin
        function print_lrp_model_check(io::IO, model, names::$T, indent::Int=0, npad::Int=0)
            npad = get_name_padding(names)
            println(io, "  "^indent, $tuple_name, "(")
            for (layer, name) in zip(children(model), children(names))
                print_lrp_model_check(io, layer, name, indent + 1, npad)
            end
            println(io, "  "^indent, "),")
        end
    end # eval
end

function print_lrp_model_check(io::IO, layer, name, indent::Int=0, npad::Int=0)
    print(io, "  "^indent, rpad(name, npad))
    printstyled(io, " => "; color=COLOR_ARROW)
    print_layer_check(io, layer)
    println(io, ",")
end

function print_layer_check(io, l)
    layer_failed = !lrp_check_layer_type(l)
    activ_failed = !lrp_check_activation(l)
    activ = activation_fn(l)

    if layer_failed && activ_failed
        return printstyled(
            io,
            "unsupported or unknown activation function $activ and layer type";
            color=COLOR_CHECK_FAIL,
        )
    elseif activ_failed
        return printstyled(
            io, "unsupported or unknown activation function $activ"; color=COLOR_CHECK_FAIL
        )
    elseif layer_failed
        return printstyled(io, "unknown layer type"; color=COLOR_CHECK_FAIL)
    end
    return printstyled(io, "supported"; color=COLOR_CHECK_PASS)
end

#===========#
# Composite #
#===========#

_range_string(r::LayerMap)         = "layer $(r.index)"
_range_string(::GlobalMap)         = "all layers"
_range_string(r::RangeMap)         = "layers $(r.range)"
_range_string(::FirstLayerMap)     = "first layer"
_range_string(::LastLayerMap)      = "last layer"
_range_string(r::GlobalTypeMap)    = "all layers"
_range_string(r::RangeTypeMap)     = "layers $(r.range)"
_range_string(::FirstLayerTypeMap) = "first layer"
_range_string(::LastLayerTypeMap)  = "last layer"
_range_string(r::FirstNTypeMap)    = "layers $(1:r.n)"

function Base.show(io::IO, m::MIME"text/plain", c::Composite, indent::Int=0)
    println(io, "Composite", "(")
    for p in c.primitives
        _show_primitive(io, p, indent + 1)
    end
    print(io, ")")
end

function _show_primitive(io::IO, r::AbstractCompositeMap, indent::Int=0)
    print(io, "  "^indent, typename(r), "( ")
    printstyled(io, "# ", _range_string(r); color=COLOR_COMMENT)
    println(io)
    printstyled(io, "  "^(indent + 1), r.rule; color=COLOR_RULE)
    println(io)
    println(io, "  "^indent, "),")
end

function _show_primitive(io::IO, r::AbstractCompositeTypeMap, indent::Int=0)
    npad = get_type_padding(io, r.map)
    print(io, "  "^indent, typename(r), "(  ")
    printstyled(io, "# ", _range_string(r); color=COLOR_COMMENT)
    println(io)
    for (type, rule) in r.map
        _print_type_rule(io, type, rule, indent + 1, npad)
    end
    println(io, " "^(indent), "),")
end

function _print_type_rule(io::IO, type::Type, rule, indent::Int=0, npad=0)
    printstyled(io, "  "^indent, rpad(type, npad); color=COLOR_TYPE)
    print(io, " => ")
    printstyled(io, rule; color=COLOR_RULE)
    println(io, ",")
end
function _print_type_rule(io::IO, types::Union, rule, indent::Int=0, npad=0)
    for t in types_in_union(types)
        _print_type_rule(io, t, rule, indent, npad)
    end
end

function get_type_padding(io::IO, map::AbstractVector{<:TypeMapPair})
    isempty(map) && return 0
    types = first.(map)
    return maximum(max_type_name_length.((io,), types))
end

max_type_name_length(io::IO, t::Type) = length(string(sprint(show, t; context=io)))
function max_type_name_length(io::IO, ts::Union)
    maximum(max_type_name_length.((io,), types_in_union(ts)))
end

types_in_union(x) = _types_in_union(x, Any[])
_types_in_union(x::Union, ts) = (_types_in_union(x.a, ts); _types_in_union(x.b, ts); ts)
_types_in_union(x, ts) = (push!(ts, x); ts)
