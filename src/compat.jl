# https://github.com/JuliaLang/julia/pull/39794
if VERSION < v"1.7.0-DEV.793"
    export Returns

    struct Returns{V} <: Function
        value::V
        Returns{V}(value) where {V} = new{V}(value)
        Returns(value) = new{Core.Typeof(value)}(value)
    end

    (obj::Returns)(args...; kw...) = obj.value
    function Base.show(io::IO, obj::Returns)
        show(io, typeof(obj))
        print(io, "(")
        show(io, obj.value)
        return print(io, ")")
    end
end
