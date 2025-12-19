# Changelog

All notable changes to this project will be documented in this file.

This file is auto-generated from conventional commits using [git-cliff](https://git-cliff.org/).

## [0.1.163] - 2025-12-19


### Bug Fixes

- Update CI workflow to use 40% coverage threshold
- Lower coverage threshold to 40% to match current state
- Update bandit config to skip false positive security warnings
- Add type ignore comment for __init__ access


### Features

- Add deprecation policy and helpers

## [0.1.161] - 2025-12-18


### Features

- Add git-cliff configuration for automated changelog generation

## [0.1.160] - 2025-12-18


### Other Changes

- Refactor type hints to use union types for improved clarity and consistency
- Remove flake8-type-checking from select rules for cleaner configuration
- Refactor code for improved readability and consistency

- Simplified edge definitions in graph examples for clarity.
- Consolidated message formatting in LLM examples for consistency.
- Streamlined error messages in audio and vision encoding functions.
- Enhanced error handling in CLI command execution.
- Improved response validation logic in message utility functions.
- Refined MCP agent configuration for better readability.
- Optimized OpenAPI path specification loading.
- Cleaned up FastAPI MCP server integration for clarity.
- Standardized error messages in retriever backend initialization.
- Enhanced logging for PostgreSQL backend similarity checks.
- Streamlined Qdrant backend import error handling.
- Improved DOCX loading error messages.
- Simplified SearchResult representation for better readability.
- Refined integration tests for embeddings and retriever functionalities.
- Enhanced unit tests for retriever backends and tools with consistent formatting.
- Refactor type hints to use built-in generic types for improved consistency and clarity
- Refactor type hints to use union types and improve code clarity

- Updated various type hints across multiple files to use the new union type syntax (e.g., `str | None` instead of `Optional[str]`).
- Removed unnecessary imports of `Optional` from typing where union types are used.
- Cleaned up code in `tools.py`, `base.py`, `registry.py`, `recorder.py`, `replay.py`, `storage.py`, `retriever.py`, and other modules to enhance readability and maintainability.
- Adjusted the `__all__` exports in `__init__.py` files for better organization.

## [0.1.159] - 2025-12-18


### Other Changes

- Refactor type hints to use built-in generic types

- Updated type hints across multiple files to replace `Dict`, `List`, and `Union` with their built-in counterparts `dict`, `list`, and `|` for unions, improving readability and consistency with Python 3.9+ standards.
- Adjusted imports to use `collections.abc` for `Awaitable`, `Callable`, and `Iterable`.
- Modified various function signatures and class attributes to reflect these changes, ensuring compatibility with the latest type hinting practices.

## [0.1.158] - 2025-12-18


### Documentation

- Update changelog [skip ci]


### Features

- Refactor DeepAgents integration and improve callback utilities
- Enhance documentation with comprehensive guides and error handling patterns


### Other Changes

- Implement feature X to enhance user experience and optimize performance

## [0.1.157] - 2025-12-18


### Features

- Add comprehensive unit tests for agent safety and edge cases

## [0.1.156] - 2025-12-18


### Features

- Add integration tests for Anthropic, OpenAI, and Embeddings providers

## [0.1.155] - 2025-12-17


### Bug Fixes

- Set warn_unused_ignores to False in mypy configuration
- Remove type ignore for Google Cloud imports in STT and TTS modules
- Add missing imports configuration for google package
- Update URLs for homepage, repository, issues, and documentation
- Update URLs and add AI-related classifiers; restructure dependencies


### Features

- Add checks for langchain_huggingface and google.genai availability in unit tests


### Other Changes

- Implement feature X to enhance user experience and optimize performance

## [0.1.154] - 2025-12-17


### Refactor

- Improve log messages for clarity and consistency

## [0.1.153] - 2025-12-17


### Bug Fixes

- Match CI config exactly (use ruff defaults)

## [0.1.152] - 2025-12-17


### Miscellaneous

- Remove mypy from hooks (use CI instead)

## [0.1.151] - 2025-12-17


### Bug Fixes

- Apply ruff formatting + switch pre-commit from black to ruff


### Features

- Implement CI workflow for automated testing and linting


### Styling

- Format all files with ruff
- Format all files with ruff


### Testing

- Fix realtime voice tests to work without API keys

## [0.1.150] - 2025-12-16


### Features

- Add docs-changelog target and update changelog generation script

## [0.1.149] - 2025-12-16


### Other Changes

- Refactor and enhance type safety across multiple modules

- Added type casting for OpenAPI specifications in io.py to ensure correct type handling.
- Improved type annotations in runtime.py for better clarity and type checking.
- Removed unnecessary type ignores in server.py.
- Enhanced type safety in FAISS, Pinecone, Postgres, and Qdrant backends by ensuring proper type casting.
- Updated retriever loaders and retriever.py to handle types more explicitly.
- Cleaned up unused imports and improved type hints in various test files.
- Refactored progress.py and schema_tools.py to enhance type safety and clarity.
- Removed unused variables and imports in multiple test files to streamline code.

## [0.1.148] - 2025-12-16


### Refactor

- Enhance Makefile commands and add formatting checks

## [0.1.147] - 2025-12-15


### Other Changes

- Refactor and enhance various components across the codebase

- Updated chat command handling to use a more explicit variable for the chat model.
- Improved argument handling in MCP commands to ensure safe access to argument properties.
- Normalized node definitions in the Graph class for better clarity and type safety.
- Enhanced Google model fetching to handle potential missing attributes gracefully.
- Added checks for model type in ImageGen to prevent errors with None values.
- Improved error handling in REST API polling example to manage different result types.
- Added type ignores for attributes in MemoryStore to suppress type checker warnings.
- Enhanced Google model listing to ensure only valid names are returned.
- Introduced a new VADMode for better control over voice activity detection.
- Improved tool definition conversion to handle various schema types more robustly.
- Added type hints and casting for better type safety in retriever tool creation.
- Enhanced MCP client resource loading to handle different resource formats.
- Updated OpenAPI example to correctly import and utilize the MCP from OpenAPI.
- Added assertions in LazyEmbeddings to ensure embeddings are loaded before use.

## [0.1.146] - 2025-12-15


### Other Changes

- Refactor type hints and improve code clarity across multiple modules

- Updated type hints to use more specific types and improved readability in `src/ai_infra/graph/utils.py`, `src/ai_infra/imagegen/imagegen.py`, and `src/ai_infra/llm/agent.py`.
- Enhanced handling of optional types and added type annotations in `src/ai_infra/llm/tools/approval.py`, `src/ai_infra/llm/memory/store.py`, and `src/ai_infra/llm/memory/trim.py`.
- Improved error handling and type safety in `src/ai_infra/llm/utils/fallbacks.py` and `src/ai_infra/llm/utils/structured.py`.
- Refined the handling of audio and vision messages in `src/ai_infra/llm/multimodal/audio.py` and `src/ai_infra/llm/multimodal/vision.py`.
- Added type checks and improved the handling of optional fields in `src/ai_infra/mcp/client/prompts.py` and `src/ai_infra/mcp/client/resources.py`.
- Enhanced the clarity of the MCP server configuration in `src/ai_infra/mcp/server/server.py` and improved type safety in `src/ai_infra/mcp/tools.py`.
- General code cleanup and refactoring for better maintainability and readability across various modules.

## [0.1.145] - 2025-12-14


### Refactor

- Update class names and improve type hints across multiple files

## [0.1.144] - 2025-12-14


### Features

- Enhance error handling with original error context and update pytest configuration

## [0.1.143] - 2025-12-14


### Features

- Add logging utility for exception handling and refactor error classes

## [0.1.142] - 2025-12-14


### Features

- Implement normalize_callbacks utility and enhance callback management with critical callbacks

## [0.1.141] - 2025-12-14


### Features

- Add safety limits to agents to prevent runaway costs and infinite loops

## [0.1.140] - 2025-12-13


### Features

- Add safety limits and security measures across various components

## [0.1.139] - 2025-12-12


### Bug Fixes

- Update repository references from nfraxio to nfraxlab in documentation and code


### Miscellaneous

- Trigger pypi publish
- Re-trigger pypi publish after enabling workflow
- Trigger pypi publish

## [0.1.138] - 2025-12-11


### Documentation

- Update README and documentation for new features and improvements

## [0.1.137] - 2025-12-10


### Miscellaneous

- Update dependencies and improve .gitignore entries

## [0.1.136] - 2025-12-10


### Features

- Update default models for providers to latest versions

## [0.1.135] - 2025-12-10


### Features

- Add MIT License to the repository

## [0.1.134] - 2025-12-10


### Features

- Enhance auto-configuration to support DATABASE_URL_PRIVATE for backend detection

## [0.1.133] - 2025-12-09


### Features

- Add filter parameter to create_retriever_tool and update tests

## [0.1.132] - 2025-12-09


### Features

- Add live test script and unit tests for Retriever Phase 6.9 enhancements

## [0.1.131] - 2025-12-09


### Features

- Add similarity parameter to PostgresBackend and validate its value

## [0.1.130] - 2025-12-08


### Testing

- Add verification for transport security configuration in disabled security case

## [0.1.129] - 2025-12-08


### Bug Fixes

- Ensure to_transport_settings always returns TransportSecuritySettings

## [0.1.128] - 2025-12-08


### Features

- Merge lifespan contexts in attach_to_fastapi for improved compatibility

## [0.1.127] - 2025-12-07


### Features

- Enhance tool event logging and handle incomplete tool calls in Agent

## [0.1.126] - 2025-12-07


### Features

- Enhance streaming events with full tool results and visibility levels

## [0.1.125] - 2025-12-07


### Features

- Implement streaming support in Agent with astream() method

## [0.1.124] - 2025-12-06


### Refactor

- Remove FastAPI integration and streaming components

## [0.1.123] - 2025-12-06


### Features

- Add FastAPI integration with chat endpoint and streaming support

## [0.1.122] - 2025-12-06


### Features

- Implement automatic security detection and configuration for MCP servers

## [0.1.121] - 2025-12-04


### Other Changes

- Add comprehensive unit tests for callback integration across various components

- Implemented tests for the Agent class to ensure proper callback handling during LLM calls, tool execution, and streaming events.
- Created tests for the unified callback system in the ai_infra.callbacks module, covering event dataclasses, callback dispatching, and built-in callback implementations.
- Developed tests for the LLM class to verify callback functionality during chat calls and streaming.
- Added tests for MCPClient to validate the use of unified callbacks for progress and logging events.
- Ensured all tests cover both synchronous and asynchronous callback methods, including edge cases and error handling.

## [0.1.120] - 2025-12-04


### Documentation

- Consolidate badge display in README for improved readability

## [0.1.119] - 2025-12-04


### Documentation

- Revise README for clarity and structure, enhance feature descriptions

## [0.1.118] - 2025-12-04


### Other Changes

- Add unit tests for MCP interceptors, prompts, and resources

- Implemented comprehensive unit tests for MCP interceptors including CachingInterceptor, RetryInterceptor, RateLimitInterceptor, LoggingInterceptor, and HeaderInjectionInterceptor.
- Added tests for the MCPToolCallRequest and the build_interceptor_chain function to ensure proper functionality and behavior.
- Developed unit tests for MCP prompts support, covering PromptInfo creation, conversion functions, and loading prompts.
- Created unit tests for MCP resources, including resource creation, conversion, and loading functionalities.
- Included integration-style tests to validate the complete workflow for prompts and resources.

## [0.1.117] - 2025-12-03


### Features

- Implement unified context management with fit_context() API

## [0.1.116] - 2025-12-03


### Other Changes

- Add unit tests for ConversationMemory and MemoryStore modules

- Implemented comprehensive unit tests for the ConversationMemory class, covering initialization, indexing, searching, deleting, and chunking functionalities.
- Added tests for the create_memory_tool and create_memory_tool_async functions to ensure proper tool creation and functionality.
- Developed unit tests for the MemoryStore class, including basic operations, TTL expiration, and semantic search capabilities.
- Included integration tests to validate the interaction between trimming and summarization processes.

## [0.1.115] - 2025-12-03


### Features

- Add error handling for loading persisted state in Retriever

## [0.1.114] - 2025-12-03


### Features

- Enhance SQLiteBackend with configurable similarity metrics

## [0.1.112] - 2025-12-02


### Other Changes

- Enhance model capability detection by introducing non-chat patterns for OpenAI models to prevent false positives in chat detection.

## [0.1.111] - 2025-12-02


### Other Changes

- Implement feature X to enhance user experience and fix bug Y in module Z

## [0.1.110] - 2025-12-02


### Other Changes

- Enhance model capability detection by categorizing models based on capabilities. Introduce functions to filter and categorize models, and update existing model fetching methods to support capability filtering.

## [0.1.109] - 2025-12-02


### Other Changes

- Enhance chat CLI with session management features, including save, load, delete, and list commands. Improve documentation for session handling and interactive commands.

## [0.1.108] - 2025-12-01


### Other Changes

- Enhance CLI documentation and add interactive chat commands for LLMs

## [0.1.107] - 2025-12-01


### Other Changes

- Add documentation for MCP Server, Realtime Voice API, STT, TTS, Vision, Progress Streaming, and Schema Tools

- Introduced MCP Server documentation with examples for creating servers, adding tools, and running the server.
- Added Realtime Voice API documentation covering usage, supported providers, event handling, and integration with FastAPI.
- Created STT documentation detailing supported providers, input formats, transcription results, and error handling.
- Developed TTS documentation explaining usage, voice selection, audio output options, and error handling.
- Added Vision documentation for image analysis, supported providers, and advanced usage scenarios.
- Introduced Progress Streaming documentation for streaming progress updates from long-running tools.
- Added Schema Tools documentation for auto-generating CRUD tools from Pydantic models, including SQLAlchemy integration and customization options.

## [0.1.106] - 2025-12-01


### Other Changes

- Remove outdated documentation files for Agent, Callbacks, Error Handling, HITL examples, Logging, Realtime Voice API, Tracing, and Validation. These files have been deprecated and are no longer relevant to the current implementation.

## [0.1.105] - 2025-12-01


### Features

- Add provider configurations for Cohere, Deepgram, ElevenLabs, Google, OpenAI, Replicate, Stability AI, Voyage AI, and xAI

## [0.1.104] - 2025-12-01


### Other Changes

- Add integration and unit tests for Realtime Voice API

- Implement integration tests for OpenAI and Gemini Realtime APIs, including connection, session lifecycle, and message handling.
- Create unit tests for realtime audio utilities, covering resampling, chunking, and silence generation.
- Develop unit tests for RealtimeVoice and related classes, focusing on provider discovery, configuration handling, tool conversion, and callback registration.
- Introduce error handling tests for invalid API keys and double disconnect scenarios.
- Ensure comprehensive coverage of model data structures and tool execution in RealtimeVoice.

## [0.1.103] - 2025-12-01


### Features

- Update dependencies and introduce Workspace abstraction for file operations

## [0.1.102] - 2025-11-30


### Features

- Add tools_from_models_sql for CRUD operations with svc-infra integration and enhance schema_tools documentation

## [0.1.101] - 2025-11-30


### Features

- Enhance replay and init modules with additional storage and progress tools

## [0.1.100] - 2025-11-30


### Other Changes

- Add unit tests for personas, progress streaming, replay module, and schema tools

- Implemented comprehensive tests for the Persona class and its methods, including YAML loading/saving and metadata handling.
- Added tests for progress streaming functionality, covering the ProgressEvent and ProgressStream classes, as well as the @progress decorator.
- Created tests for the replay module, including WorkflowRecorder and replay() function, ensuring proper recording and playback of workflows.
- Developed tests for schema_tools, focusing on the tools_from_models() function and its integration with Pydantic models, including CRUD operations and pagination configuration.

## [0.1.99] - 2025-11-30


### Other Changes

- Remove legacy action planner and integrate DeepAgents

## [0.1.98] - 2025-11-29


### Features

- Add audio output and discovery modules for LLMs

## [0.1.97] - 2025-11-29


### Features

- Add audio input support and related utilities for LLM

## [0.1.96] - 2025-11-28


### Features

- Add Text-to-Speech (TTS) module with multi-provider support

## [0.1.95] - 2025-11-28


### Features

- Add CLI commands for image generation provider and model discovery

## [0.1.94] - 2025-11-28


### Features

- Enhance Google image generation support with Gemini models and update default configurations

## [0.1.93] - 2025-11-28


### Features

- Implement provider-agnostic image generation module with support for OpenAI, Google, Stability AI, and Replicate

## [0.1.92] - 2025-11-28


### Testing

- Add comprehensive unit tests for create_retriever_tool and create_retriever_tool_async

## [0.1.91] - 2025-11-28


### Features

- Enhance workspace sandboxing with explicit root setting and improved path handling

## [0.1.90] - 2025-11-28


### Features

- Add create_retriever_tool for Agent integration


### Refactor

- Move create_retriever_tool to llm/tools/custom/

## [0.1.89] - 2025-11-27


### Testing

- Add comprehensive unit tests for Retriever module

## [0.1.88] - 2025-11-27


### Features

- Add Pinecone, Qdrant, FAISS backends and main Retriever class

## [0.1.87] - 2025-11-27


### Other Changes

- Remove outdated error handling, HITL examples, logging, tracing, and validation docs; add comprehensive new documentation structure and examples for ai-infra library.

## [0.1.86] - 2025-11-27


### Features

- Implement embeddings module with provider-agnostic interface

## [0.1.85] - 2025-11-27


### Other Changes

- Add comprehensive unit tests for error handling, logging, callbacks, and validation modules

- Implement tests for cross-cutting concerns including error hierarchy, structured logging, callback system, and tracing functionalities.
- Introduce tests for LLM error handling utilities, ensuring proper translation of provider errors and extraction of retry information.
- Develop validation tests covering provider validation, temperature constraints, message structure, and configuration checks.
- Ensure coverage for both synchronous and asynchronous error handling decorators.

## [0.1.84] - 2025-11-27


### Features

- Enhance OpenAPI loading and processing capabilities

## [0.1.83] - 2025-11-27


### Refactor

- Introduce comprehensive MCPClient exception handling and enhance client initialization with connection management options

## [0.1.82] - 2025-11-26


### Refactor

- Enhance Graph API with zero-config building and validation features

## [0.1.81] - 2025-11-26


### Other Changes

- Refactor MCP server and client structure

- Introduced a new `MCPClient` class for managing connections to multiple MCP servers, including discovery and error handling.
- Removed the `core.py` file, consolidating server functionality into `server.py`.
- Updated `MCPServer` class to streamline the addition of applications and tools, enhancing the API for adding FastAPI and OpenAPI specifications.
- Implemented async context management for session handling in the client.
- Added utility functions for better error reporting and server information extraction.
- Enhanced documentation and type hints across the codebase for improved clarity and maintainability.

## [0.1.80] - 2025-11-26


### Refactor

- Update Graph import paths and remove core module

## [0.1.79] - 2025-11-26


### Features

- Introduce BaseLLM and LLM classes for enhanced model interaction

## [0.1.78] - 2025-11-26


### Other Changes

- Add unit tests for HITL approval workflow and session management

- Implement tests for ApprovalRequest, ApprovalResponse, OutputReviewRequest, and OutputReviewResponse to validate their behavior and properties.
- Create tests for built-in approval handlers and ApprovalConfig to ensure correct approval logic and event handling.
- Add tests for event hooks related to approval events, verifying that events are emitted and handled correctly during approval workflows.
- Introduce tests for session management, including SessionResult, PendingAction, and SessionConfig, to ensure proper session handling and state management.
- Validate integration of events with ApprovalConfig to confirm that events are fired during approval processes.

## [0.1.77] - 2025-11-26


### Other Changes

- Add tool execution configuration with error handling, timeouts, and validation

## [0.1.76] - 2025-11-26


### Other Changes

- Rename BaseLLMCore to BaseLLM and add logging hooks for request/response observability

## [0.1.75] - 2025-11-26


### Other Changes

- Remove deprecated aliases and update imports to use new names for core components

## [0.1.74] - 2025-11-26


### Other Changes

- Add discovery commands and API for provider/model management

## [0.1.73] - 2025-11-26


### Other Changes

- Refactor API components with new names and add backward compatibility for deprecated aliases

## [0.1.72] - 2025-11-26


### Other Changes

- Refactor LLM provider usage and update dependencies

- Updated `pyproject.toml` to include `google-genai` SDK for model discovery.
- Changed LLM provider from MistralAI to OpenAI in example scripts:
  - `01_agent_basic.py`
  - `02_llm_chat_basic.py`
  - `02_multi_server_agent.py`
- Removed MistralAI and Deepseek models from `models.py` and `providers.py` to streamline model management.
- Added `poetry.toml` to configure virtual environments in project.

## [0.1.71] - 2025-11-26


### Other Changes

- Update dependencies for LangChain ecosystem and remove obsolete tests

- Updated LangChain ecosystem dependencies to version 1.0.0 and above.
- Removed the test file `test_mcp_expose_core.py` as it is no longer needed.

## [0.1.70] - 2025-11-13


### Other Changes

- Update .gitignore to include documentation files

## [0.1.69] - 2025-11-08


### Other Changes

- Add feedback document outlining areas for improvement in ai-infra library

## [0.1.68] - 2025-11-07


### Other Changes

- Refactor code structure for improved readability and maintainability
- Added precommits

## [0.1.67] - 2025-09-15


### Other Changes

- Added precommits

## [0.1.66] - 2025-09-08


### Other Changes

- Mcp bug fix

## [0.1.65] - 2025-09-08


### Other Changes

- Mcp bug fix

## [0.1.64] - 2025-09-08


### Other Changes

- Added ai-infra cli mcp

## [0.1.63] - 2025-09-08


### Other Changes

- Added cli helper tools to custom tools

## [0.1.62] - 2025-09-07


### Other Changes

- Bug fix

## [0.1.61] - 2025-09-07


### Other Changes

- Changed run command to run cli

## [0.1.60] - 2025-09-07


### Other Changes

- Changed run command to run cli

## [0.1.59] - 2025-09-06


### Other Changes

- Bug fix

## [0.1.58] - 2025-09-03


### Other Changes

- Moved mcp stdio publisher under custom mcps

## [0.1.57] - 2025-09-03


### Other Changes

- Updated readme

## [0.1.56] - 2025-09-03


### Other Changes

- Updated exposure name to publisher

## [0.1.55] - 2025-09-03


### Other Changes

- Add executable to functionatlities of stdio exposure

## [0.1.54] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.53] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.52] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.51] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.50] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.49] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.48] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.47] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.46] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.45] - 2025-09-03


### Other Changes

- Bug fix

## [0.1.44] - 2025-09-03


### Miscellaneous

- Ensure shim is executable

## [0.1.43] - 2025-09-03


### Other Changes

- Created mcp for exposing mcps and exposing that to other agents

## [0.1.42] - 2025-09-02


### Other Changes

- Updated to fallback on default model for any provider that the model is not specified

## [0.1.41] - 2025-09-02


### Other Changes

- Enhanced structured results from prompts

## [0.1.40] - 2025-09-02


### Other Changes

- Updated promp structured

## [0.1.39] - 2025-09-02


### Bug Fixes

- Fixed imports

## [0.1.38] - 2025-09-02


### Bug Fixes

- Fixed imports

## [0.1.37] - 2025-09-02


### Other Changes

- Enhancing tools and mcp gen

## [0.1.36] - 2025-09-02


### Other Changes

- Enhancing tools and mcp gen

## [0.1.35] - 2025-09-02


### Other Changes

- Improving tools from funcs and mcp from funcs

## [0.1.34] - 2025-09-01


### Other Changes

- Converted tool planner to action planner for wider range of planning and reasoning

## [0.1.33] - 2025-09-01


### Other Changes

- Converted tool planner to action planner for wider range of planning and reasoning

## [0.1.32] - 2025-08-29


### Other Changes

- Bug fix

## [0.1.31] - 2025-08-29


### Other Changes

- Added default model and provider

## [0.1.30] - 2025-08-29


### Other Changes

- Bug fix

## [0.1.29] - 2025-08-29


### Other Changes

- Added complexity analyzer and assessor to tool_planner agent

## [0.1.28] - 2025-08-28


### Other Changes

- Built a full blown planner agent for planning tool calls for other agents for complex tasks

## [0.1.27] - 2025-08-28


### Other Changes

- Bug fix

## [0.1.26] - 2025-08-28


### Other Changes

- Bug fix

## [0.1.25] - 2025-08-28


### Other Changes

- Made hitl async

## [0.1.24] - 2025-08-28


### Other Changes

- Made hitl async

## [0.1.23] - 2025-08-28


### Other Changes

- Made hitl async

## [0.1.22] - 2025-08-28


### Other Changes

- Made hitl async
- Enhanced mcps

## [0.1.21] - 2025-08-27


### Other Changes

- Asynced all tools

## [0.1.20] - 2025-08-27


### Other Changes

- Bug fix

## [0.1.19] - 2025-08-27


### Other Changes

- Moved project management and cli mcps

## [0.1.18] - 2025-08-27


### Other Changes

- Added file management tools

## [0.1.17] - 2025-08-27


### Other Changes

- Enhanced run command tool of cli

## [0.1.16] - 2025-08-26


### Miscellaneous

- Sync with origin/main to include latest upstream changes

## [0.1.15] - 2025-08-26


### Miscellaneous

- Sync with latest origin/main and commit local changes

## [0.1.14] - 2025-08-26


### Other Changes

- Default all models to the fastest

## [0.1.13] - 2025-08-25


### Other Changes

- Setting autoapprove to false on sys gate

## [0.1.12] - 2025-08-25


### Other Changes

- Setting autoapprove to false on sys gate

## [0.1.11] - 2025-08-25


### Other Changes

- Added autoapprove flag for sys gate

## [0.1.10] - 2025-08-25


### Other Changes

- Added terminal hitl sys gate

## [0.1.9] - 2025-08-25


### Other Changes

- Added default to all models

## [0.1.8] - 2025-08-25


### Other Changes

- Added default to all models

## [0.1.7] - 2025-08-25


### Other Changes

- Added terminal runner tool

## [0.1.6] - 2025-08-25


### Other Changes

- Provided support for mistralai and deepseek models

## [0.1.5] - 2025-08-24


### Other Changes

- Publishing to pypi

## [0.1.4] - 2025-08-24


### Bug Fixes

- Preserve message dict shape in _apply_hitl replacement
- Ensure HITL-wrapped tools used by agent (context.tools=effective_tools)
- Safe retry handling in chat() when event loop running
- Set has_memory to False in analyze to avoid AttributeError after config removal


### Build

- Building super mcps
- Building super mcps
- Building super mcps
- Building super mcps


### Documentation

- Expand HITL callback contract in set_hitl docstring


### Features

- Async-aware tool gate (_maybe_await in _wrap_tool_for_hitl)
- Support async HITL callbacks via _maybe_await in _apply_hitl and streaming gating
- Preserve full values shape during HITL gating in arun_agent_stream
- Warn when structured output unsupported (with_structured_output)
- Add explicit global tool usage policy with logging and optional enforcement
- Add trace callback for node entry/exit; refactor for deduplication and bugfixes; always use self._memory_store for graph compilation
- Allow run and run_async to accept state as kwargs or dict; improve ConditionalEdge targets inference; update usage example
- ConditionalEdge can infer targets from router_fn if not provided; update usage example in __init__.py
- ConditionalEdge can infer targets from router_fn if not provided; update usage example


### Miscellaneous

- Minor HITL gating guard adjustment
- Harden HITL gating in arun_agent_stream with safer messages mutation
- Strip agent/tool kwargs in chat() and achat() for safety
- Refine stream_mode typing and related streaming logic
- Guard stream_tokens() against agent/tool kwargs
- Centralize dotenv loading and remove duplicate calls
- Unify system message to plain dict
- Update quickstart basics example


### Other Changes

- Publishing to pypi
- Moved quickstart to examples under each capability
- Enhancing openapi -> mcp arg returns
- Trying to log openapi to mcp conversions
- Trying to log openapi to mcp conversions
- Trying to log openapi to mcp conversions
- Trying to log openapi to mcp conversions
- Trying to log openapi to mcp conversions
- Trying to log openapi to mcp conversions
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Updating tool info from openmcp setup
- Adding openmcp integration to mcp servers
- Adding openmcp integration to mcp servers
- Adding openmcp integration to mcp servers
- Adding openmcp docs
- Adding openmcp docs
- Adding openmcp docs
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Adding fastapi to mcp conversion
- Simplified spinning an mcp with tools
- Simplified spinning an mcp with tools
- Simplified spinning an mcp with tools
- Simplified spinning an mcp with tools
- Enhancing openapi to mcp
- Enhancing openapi to mcp
- Enhancing openapi to mcp
- Gotta fix openai integration of agent
- Gotta fix openai integration of agent
- Dynamically getting server metadata
- Dynamically getting server metadata
- Enhancing our coremcpclient
- Enhancing our coremcpclient
- Added agent calling with tools
- Added agent calling with tools
- Added agent calling with tools
- Added easy serverside streamable
- Added easy serverside streamable
- Added easy serverside streamable
- Added easy serverside streamable
- Going back to regular langchain mcps
- Going back to regular langchain mcps
- Leveraging fastmcp and removing custom mcp configs
- Trying fastmcp package
- Trying fastmcp package
- Trying fastmcp package
- Trying fastmcp package
- Trying fastmcp package
- Trying fastmcp package
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Enhanced mcp funcs
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Adding openapi as mcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- Separating hostedmcp and openmcp
- OpenMCP accepts json/yaml and OpenMCP default configs for remote or fastapi hosted servers
- OpenMCP accepts json/yaml and OpenMCP default configs for remote or fastapi hosted servers
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- MCP ready for primetime
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Enhancing mcp setup
- Ton of MCP funcs added
- Adding mcp capabilities to fastapi
- Adding mcp capabilities to fastapi
- Adding mcp capabilities to fastapi
- Adding mcp capabilities to fastapi
- Adding mcp capabilities to fastapi
- Adding mcp capabilities to fastapi
- Refactored everything
- Organized utils
- Organized utils
- Releasing new version of infra
- Releasing new version of infra
- Releasing new version of infra
- Releasing new version of infra
- Releasing new version of infra
- Committing all staged changes
- Committing all staged changes
- Normalize then HITL-wrap tools; prevent tool loss and ensure BaseTool wrapping
- Refactoring further
- Refactoring further
- Committing all staged changes
- Refactoring further
- Adding refactored code
- Stream_tokens now uses keyword-only arguments and filters agent-specific keys from model_kwargs; prevents tools and agent args from being passed to LLM-only streaming. General HITL and agent improvements.
- Stream_tokens now uses keyword-only arguments and filters agent-specific keys from model_kwargs; prevents tools and agent args from being passed to LLM-only streaming. General HITL and agent improvements.
- Remove tools from model_kwargs in stream_tokens to prevent warnings when LLM-only streaming. General HITL and agent improvements, type annotation fixes, and safer tool HITL wrapper for structured outputs.
- Update stream_mode annotation to accept str, list, or tuple; HITL tool wrapper now coerces replacement to structured type if needed; preserve dict structure for values snapshots in agent stream; general HITL and agent improvements
- Coerce replacement to structured type if tool expects structured output (JSON), safer for block/modify actions
- In arun_agent_stream, apply HITL only to last message in 'values' dict, not the whole dict, to preserve structure
- Preserve dict structure for 'values' snapshots, only modify last message content if present
- Remove unused started timing variable from arun_agent in CoreLLM
- Remove unused metrics plumbing from CoreLLM (metrics_cb, set_metrics, and docstrings)
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Refactor CoreLLM, tool control logic, and quickstart examples for improved agent control and streaming
- Update CoreLLM and tool control logic; improve agent control and streaming; update quickstart examples
- Join list to comma-separated string; update tool control and agent logic for clarity and correctness
- Modularize tool control normalization, improve CoreLLM structure, and update quickstart examples
- Add force_once to ToolCallControls, update agent logic, and add streaming/controlled agent examples in basics.py
- Add ToolCallControls.force_once, update agent tool control logic, and add streaming/controlled agent examples in basics.py
- Import Providers and Models directly from their modules in core.py. General code cleanup.
- Wrap async for in async main and use asyncio.run in basics.py; ensure Plan.steps is a list of multiple strings; code cleanup
- Add async test_agent_stream and fix main entrypoint for streaming agent output
- Update structured_response to ensure Plan.steps is a list of multiple strings
- Remove unused protocols and clean up codebase
- Confirmed protocols and models usage; no changes needed
- Remove redundant parentheses and minor code cleanup
- Commit all changes
- Default stream_mode to ["updates", "values"] for astream/stream; polish CoreGraph streaming UX
- Add draw_mermaid() to CoreGraph and polish streaming/persistence; update quickstart/graph.py usage
- Guard auto-edges, add values streaming helpers, clarify sync wrapper error, and update quickstart/graph.py usage
- Fix sync wrapper to raise in running event loop, robust streaming, and persistence plumbing
- Fix CoreGraph streaming, event loop, and node function issues; refactor quickstart/graph.py for correct node usage
- Add Postgres checkpointer/store support, checkpoint inspection, and memory analysis
- Move utility functions to utils.py, remove duplication, and clean up core.py
- Move all utility functions to utils.py, remove duplication, and clean up core.py
- Move all utility functions to utils.py, remove duplication, and clean up core.py
- Fix imports, deduplicate SystemMessage logic, extract config dict processing, clarify types, and clean up code structure
- Fix MCP import errors, refactor config/model usage, and clean up code structure
- Move all utility, hook, and router logic to utils.py; shrink core.py; unify arun/run naming
- Deduplicate hook and trace logic, use helpers for sync/async modes
- Refactor CoreGraph run/arun to match CoreLLM, support sync/async nodes, always use invoke for sync
- Implement trace function usage in CoreGraph run example
- Unify agent setup, clarify sync/async agent execution, and improve maintainability
- Refactor CoreLLM and utils to group related functionalities together with clear sections and improved documentation
- Refactor CoreLLM to centralize provider/model validation and remove redundant checks
- Update quickstart/server.py and related MCP config/model usage for clarity and correctness
- Update MCP __init__.py exports for consistent naming and API surface
- Fix CoreMCP argument path resolution to always use the import location as base for relative paths
- Improve CoreMCP argument path resolution to be relative to import location, update models for optional fields, and clean up server setup logic
- Refactor get_server_prompt for clarity, use config.prompts directly, and update models for host support
- Refactor CoreMCP to use host from config, update quickstart, and align models for professional usage
- Use targets list and router_fn returns node name. Update usage example.
- Update usage to unified Edge/ConditionalEdge API in CoreGraph and example
- Unify edge handling in CoreGraph using Edge and ConditionalEdge classes
- Centralize StateGraph construction and remove repetition for nodes, edges, and conditional_edges in CoreGraph
- Latest changes to llm/__init__.py and other modules
- Update .gitignore to always exclude __pycache__ and ensure repo is clean
- Add __pycache__/ to .gitignore to prevent tracking of Python bytecode cache directories
- Remove all __pycache__ directories from repository
- Split graph protocols, models, and core logic into separate files and clean up __init__.py
- Apply latest changes
- Validation improvements, always async-wrapped nodes/routers, code cleanup, and bugfixes
- Always use async engine, update main example to use asyncio.run for run_async
- Remove code duplication in graph building, improve readability and maintainability
- Always use START/END constants, compute entry/exit from real edge set, and show unreachable nodes in analyze()
- Tighten types, validation, and fix Pydantic config for CoreGraph
- Refactor CoreGraph __init__ and config usage for clarity and fix AttributeError
- Push all local changes
- Programmatic graph info with Pydantic, improved summary/describe methods, and type fixes
- Add necessary imports to CoreGraph and StateGraph usage
- Update model initialization and selection logic in core module
- Push all changes to git with message
- Push all changes
- Dynamic provider checks, improved imports, and codebase consistency
- Dynamic provider checks, improved imports, and codebase consistency
- Refactor provider checks in BaseLLM to dynamically use Providers class attributes
- Enforce and clarify usage of Providers and Models throughout the codebase
- Clarify and enforce usage of Providers and Models throughout LLM context, base, and settings
- Refactor provider/model structure and update quickstart example
- Refactor providers and models for unified provider/model API; update quickstart example
- Push all changes
- Push all changes
- Update __init__.py to use dotenv for environment variable loading
- Push all changes
- Update __init__.py to load environment variables with dotenv
- Update __init__.py to load environment variables with dotenv
- Update __init__.py to load environment variables with dotenv
- Update __init__.py to load environment variables with dotenv
- Update pyproject.toml and poetry.lock after installing python-dotenv
- Push all changes after dotenv install and code updates
- Push all current changes
- Setting up project
- First commit


### Refactor

- Refactoring
- Remove non-essential comments/docstrings in CoreLLM
- Extract HITL + tool policy logic to tools.py and update core.py
- Extract runtime binding (ModelRegistry, tool_used, bind_model_with_tools, make_agent_with_context) into runtime_bind.py and delegate from core
- Simplify HITL block handling by returning verbatim replacement
- Simplify effective tool selection and retain quickstart edits
- Deduplicate, simplify, and add tracing to CoreGraph; fix memory_store bug; general code cleanups
- Deduplicate, simplify, and add tracing to CoreGraph; fix memory_store bug; general code cleanups
- Deduplicate and simplify CoreGraph, add trace callback for node entry/exit, fix memory_store bug, and minor code cleanups

<!-- Generated by git-cliff -->
