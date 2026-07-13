# Shared code-quality (Aqua.jl) and type-stability (JET.jl) checks.
# Kept consistent across the Julia-XAI packages — see REFACTOR.md.
using ExplainableAI
using Test
using Aqua
using JET

@testset "Aqua.jl" begin
    @info "Running Aqua.jl code-quality tests. These might print warnings from dependencies."
    Aqua.test_all(ExplainableAI; ambiguities = false)
end
# JET's v0.11 series supports Julia v1.12 and above only.
if VERSION >= v"1.12"
    @testset "JET.jl" begin
        @info "Running JET.jl type-stability tests."
        JET.test_package(ExplainableAI; target_defined_modules = true)
    end
end
