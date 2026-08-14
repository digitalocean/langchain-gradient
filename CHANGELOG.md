# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-14

### Changed
- **BREAKING**: Migrated the inference client from the deprecated `gradient` SDK to `pydo` (`pydo.inference.Client`). The Gradient SDK retires on 2026-08-15. (#32)
- **BREAKING**: Exception types now come from azure-core/pydo (`HttpResponseError`, `ClientAuthenticationError`, `ServiceResponseTimeoutError`, etc.) instead of Gradient SDK exceptions. Update any `except gradient.*` handlers.
- **BREAKING**: Response objects are now pydo `DotDict`s rather than Gradient/Stainless typed models. Attribute access still works; `response_metadata` shape is unchanged.
- **BREAKING**: `timeout` is now a required positive `int` (default `120`). Floats are truncated; `None`, `0`, and negative values raise `ValidationError`. Sub-second floats (e.g. `0.5`) no longer work. The value is applied as azure-core `connection_timeout` / `read_timeout` (and the retry budget), so requests actually abort after N seconds.
- User-agent is set via the public `user_agent` client kwarg.
- Streaming responses now close the SSE stream in a `finally` block.

### Removed
- Dependency on `gradient` (replaced by `pydo ^0.39.0`).

## [0.2.1] - 2026-07-23

### Fixed
- Streaming no longer crashes on chunks with empty `choices` (e.g. the usage-only final chunk emitted when `stream_options={"include_usage": True}`). Thanks @MrDiggles2! (#27)
- `stream_options` is no longer sent on non-streaming requests. The inference API rejects it unless `stream=True`, so `invoke()` on models configured with `stream_options` failed with a 400 validation error. `usage_metadata` is still populated from the response. (#29)
- `user_agent_version` now reports the actual installed package version instead of a hardcoded stale one.

### Changed
- Updated integration tests for the current inference API: replaced retired models (`mistral-nemo-instruct-2407`, `llama3-8b-instruct`) with currently served ones (`mistral-3-14B`, `openai-gpt-oss-20b`); empty prompts are now accepted by the API; raised the o3-mini test token budget so reasoning doesn't exhaust it. (#28)

## [0.2.0] - 2026-06-24

### Changed
- **BREAKING**: Widened `langchain-core` constraint to `>=0.3.81,<2.0.0` to support langchain-core 1.x. This resolves dependency conflicts with the wider LangChain ecosystem (langgraph, langchain-mcp-adapters, langgraph-sdk), which now require `langchain-core>=1.4.0`.
- **BREAKING**: Dropped Python 3.9 support (minimum is now Python 3.10), as langchain-core 1.x requires Python `>=3.10`. Python 3.9 reached end-of-life in October 2025.
- `ChatGradient.is_lc_serializable()` now returns `False`. langchain-core 1.x only deserializes classes in its internal serialization allowlist; as a third-party partner package, `ChatGradient` is not registered there, so `load()` cannot round-trip it. Declaring the model non-serializable reflects this rather than advertising a path that fails at deserialization time.

### Removed
- Removed `langchain-tests` from runtime dependencies (it is a testing-only helper and now lives solely in the `test` dependency group).

### Fixed
- Removed an extraneous `f` prefix on a non-interpolated string (`user_agent_package`).

## [0.1.23] - 2025-12-27

### Security
- **CRITICAL**: Updated langchain-core to ^0.3.81 to fix CVE-2025-68664 (serialization injection vulnerability)

### Changed
- Removed obsolete `c63a5cfe-b235-4fbe-8bbb-82a9e02a482a-python` dependency (legacy GradientAI package no longer needed)

### Added
- Tool calling support via `bind_tools()` method
- Support for `ToolMessage` handling in chat completions
- Tool calling support in streaming mode
- `_convert_message()` method to handle ToolMessage and AIMessage with tool_calls
- `_parse_tool_calls()` with proper logging for parse errors

## [0.1.22] - 2024-09-23

### Changed
- Upgraded gradient package dependency to version ^3.0.0
- Refactored ChatGradient to use the new Gradient package API

## [0.1.21] - 2024-09-19

### Added
- User agent properties (`user_agent_package` and `user_agent_version`) to ChatGradient
- API key exclusion from serialization for security

### Changed
- Enhanced ChatGradient with user agent information when calling GradientAI SDK

## [0.1.20] - 2024-08-14

### Changed
- Renamed package from `langchain-gradientai` to `langchain-gradient`
- Updated all related files and documentation to reflect new package name
- Removed deprecated code

### Added
- PyPI downloads badge in README

## [0.1.19] - 2024-07-24

### Added
- Usage metadata support in ChatGradientAI
- Integration tests for various model parameters
- Integration tests for streaming functionality

### Changed
- Updated ChatGradientAI to use streaming for completions
- Improved documentation and examples in README and notebooks

## [0.1.18] and earlier

### Added
- Initial release of langchain-gradientai package
- ChatGradientAI class for LangChain integration
- Basic chat completion support
- Streaming support
- Support for various model parameters (temperature, max_tokens, etc.)

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

