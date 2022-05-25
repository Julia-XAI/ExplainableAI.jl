# ExplainableAI.jl
## Version `v0.4.0`
Changes:
- ![BREAKING][badge-breaking] Update heatmapping normalizer, using ColorScheme's `get`. Breaking due to renaming `normalize` to ColorScheme's `rangescale`. ([#57][pr-57])
- ![BREAKING][badge-breaking] Rename `InputAugmentation` to `NoiseAugmentation`. ([#65][pr-65])
- ![BREAKING][badge-breaking] `GammaRule` and `EpsilonRule` now use default arguments instead of keyword arguments, removing the need for users to type unicode symbols. ([#70][pr-70]) 
- ![BREAKING][badge-breaking]![Bugfix][badge-bugfix] `ZBoxRule` now requires parameters `low` and `high` instead of computing them from the input. ([#69][pr-69]) 
- ![Feature][badge-feature] Add `IntegratedGradients` analyzer. ([#65][pr-65])
- ![Feature][badge-feature] Add `InterpolationAugmentation` wrapper. ([#65][pr-65])
- ![Feature][badge-feature] Allow any type of `Sampleable` in `NoiseAugmentation`. ([#65][pr-65])

Performance improvements:
- ![Enhancement][badge-enhancement] Remove use of `mapreduce`. ([#58][pr-58])
- ![Enhancement][badge-enhancement] Load LoopVectorization.jl in tests and benchmarks to speed up Tullio on CPU. ([#66][pr-66])
- ![Enhancement][badge-enhancement] Type stability fixes for `GammaRule`. ([#70][pr-70])

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Security][badge-security]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[pr-70]: https://github.com/adrhill/ExplainableAI.jl/pull/70
[pr-69]: https://github.com/adrhill/ExplainableAI.jl/pull/69
[pr-67]: https://github.com/adrhill/ExplainableAI.jl/pull/67
[pr-66]: https://github.com/adrhill/ExplainableAI.jl/pull/66
[pr-65]: https://github.com/adrhill/ExplainableAI.jl/pull/65
[pr-58]: https://github.com/adrhill/ExplainableAI.jl/pull/58
[pr-57]: https://github.com/adrhill/ExplainableAI.jl/pull/57

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
