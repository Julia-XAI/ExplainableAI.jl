# Shared code-quality (Aqua.jl), type-stability (JET.jl) and import-hygiene
# (ExplicitImports.jl) checks.
# Kept consistent across the Julia-XAI packages — see REFACTOR.md.
using ExplainableAI
using Test
using Aqua
using JET
using ExplicitImports

@testset "Aqua.jl" begin
    @info "Running Aqua.jl code-quality tests. These might print warnings from dependencies."
    Aqua.test_all(ExplainableAI; ambiguities = false)
    Aqua.test_ambiguities(ExplainableAI)
end

@testset "ExplicitImports.jl" begin
    @info "Running ExplicitImports.jl import-hygiene tests."
    @test check_no_stale_explicit_imports(ExplainableAI) === nothing
    @test check_all_explicit_imports_via_owners(ExplainableAI) === nothing
    @test check_all_qualified_accesses_via_owners(ExplainableAI) === nothing
    @test check_no_self_qualified_accesses(ExplainableAI) === nothing
end

# JET's v0.11 series supports Julia v1.12 and above only.
if VERSION >= v"1.12"
    @testset "JET.jl" begin
        @info "Running JET.jl type-stability tests."
        JET.test_package(ExplainableAI; target_defined_modules = true)
    end
end
