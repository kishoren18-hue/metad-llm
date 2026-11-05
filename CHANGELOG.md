# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-04

### Added

- `compare()` and `metad-llm compare`: paired bootstrap comparison of two models on the same items, with CIs, p-values and a verdict per metric
- `confidence_from_logprobs()`: option-normalized confidence from token log-probabilities
- Contributor docs, issue templates, linting and coverage in CI, release workflow

## [0.1.0] - 2026-11-04

### Added

- Maximum-likelihood meta-d' (Maniscalco & Lau 2012), type-1 SDT, ECE, Brier, type-2 AUROC
- `assess()` and `assess_binary()` reports with bootstrap CIs; CLI with plots
- Simulated observers and parameter-recovery tests
