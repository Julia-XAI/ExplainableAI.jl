macro warnpcfail(ex::Expr)
    modl = __module__
    file = __source__.file === nothing ? "?" : String(__source__.file)
    line = __source__.line
    quote
        $(esc(ex)) || @warn """precompile directive $($(Expr(:quote, ex)))
        failed. Please report an issue in $($modl) (after checking for duplicates) or remove this directive.""" _file =
            $file _line = $line
    end
end

function _precompile_()
    eltypes = (Float32,)
    ruletypes = (ZeroRule, EpsilonRule, GammaRule, ZBoxRule)
    layertypes = (
        Dense,
        Conv,
        MaxPool,
        AdaptiveMaxPool,
        GlobalMaxPool,
        MeanPool,
        AdaptiveMeanPool,
        GlobalMeanPool,
        DepthwiseConv,
        ConvTranspose,
        CrossCor,
        Dropout,
        AlphaDropout,
        typeof(Flux.flatten),
    )

    for R in ruletypes
        for T in eltypes
            AT = Array{T}
            @warnpcfail precompile(modify_denominator, (R, AT))
            @warnpcfail precompile(modify_params, (R, AT, AT))

            for L in layertypes
                @warnpcfail precompile(_modify_layer, (R, L))
                @warnpcfail precompile(lrp, (R, L, AT, AT))
            end
        end
    end
end
