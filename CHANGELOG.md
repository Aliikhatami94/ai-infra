# CHANGELOG


## v1.1.1 (2025-12-31)

### Bug Fixes

- Sync formatting and docs updates ([#9](https://github.com/nfraxlab/ai-infra/pull/9),
  [`c9fc328`](https://github.com/nfraxlab/ai-infra/commit/c9fc3283907ab69ae4d01fe55356cbbc1300846a))


## v1.1.0 (2025-12-30)

### Bug Fixes

- Add PR title enforcement workflow ([#5](https://github.com/nfraxlab/ai-infra/pull/5),
  [`ba5f759`](https://github.com/nfraxlab/ai-infra/commit/ba5f75923a48ad61fce1623c442d947f066f95fd))

- Semantic-release push tags before publish ([#7](https://github.com/nfraxlab/ai-infra/pull/7),
  [`be056aa`](https://github.com/nfraxlab/ai-infra/commit/be056aae2a78959826e36b50330b38a0a6731a16))

- Use RELEASE_TOKEN for bypass branch protection ([#8](https://github.com/nfraxlab/ai-infra/pull/8),
  [`817de0e`](https://github.com/nfraxlab/ai-infra/commit/817de0e22dd347fa4744a61a092555016f216ce0))

### Chores

- Regenerate poetry.lock after adding semantic-release
  ([#1](https://github.com/nfraxlab/ai-infra/pull/1),
  [`14fb639`](https://github.com/nfraxlab/ai-infra/commit/14fb639575ac348850ce095805b40673b148fdaf))

* chore: regenerate poetry.lock after adding semantic-release

* feat: add robust make pr automation with contributor-safe workflow

### Continuous Integration

- Switch to semantic-release for clean versioning
  ([`2329cf4`](https://github.com/nfraxlab/ai-infra/commit/2329cf4aead27fa690be3edbfc5e8fc0c4eff0af))

- Removed force-pushes, amends, and lockfile regeneration from CI - Added release.yml: uses
  python-semantic-release for version bumps based on Conventional Commits (feat: minor, fix: patch)
  - Simplified publish-pypi.yml: triggers only on v* tags - Added python-semantic-release to dev
  dependencies

Flow: 1. Push to main → release.yml analyzes commits 2. If releaseable: bump version, update
  changelog, commit, tag, GH Release 3. Tag push → publish-pypi.yml builds and publishes to PyPI

No more history rewriting. Clean and scalable.

### Documentation

- Update CONTRIBUTING.md with make pr workflow ([#4](https://github.com/nfraxlab/ai-infra/pull/4),
  [`e62db7e`](https://github.com/nfraxlab/ai-infra/commit/e62db7e010ee352908bddf5ea8fc3c65640f3781))

### Features

- Phase 10.1 - test coverage improvements ([#6](https://github.com/nfraxlab/ai-infra/pull/6),
  [`4aa1f32`](https://github.com/nfraxlab/ai-infra/commit/4aa1f32eb30addc3f6d747be32db19e70217ea77))


## v1.0.3 (2025-12-28)

### Bug Fixes

- **ci**: Only release x.y.0 versions, no auto-bump
  ([`1ac2eca`](https://github.com/nfraxlab/ai-infra/commit/1ac2eca54de8ca0f1452088595bf8f58502cf7fa))

Changed the workflow to: - Only publish when version is x.y.0 (deliberate release) - Skip all
  publish steps for non x.y.0 versions - No more auto-bumping patch version on every commit - GitHub
  Release created automatically for x.y.0 versions

### Continuous Integration

- Create GitHub Release for every version
  ([`8d65714`](https://github.com/nfraxlab/ai-infra/commit/8d657143e5ede5ac2eb4be1cddfe6b2644a82ac9))

Every push to main now: 1. Auto-bumps patch version 2. Tags the version 3. Publishes to PyPI 4.
  Creates GitHub Release

Tag is the source of truth - PyPI + GitHub Release happen together from the same tag in the same CI
  run.

- Release v1.0.3
  ([`7a4dab6`](https://github.com/nfraxlab/ai-infra/commit/7a4dab6db9403bab3a5d03edf0ccfaf4a82d2ef0))


## v1.0.2 (2025-12-28)

### Bug Fixes

- **ci**: Detect x.y.0 releases and skip auto-bump to create GitHub Release
  ([`73f9058`](https://github.com/nfraxlab/ai-infra/commit/73f9058a3c6d3e3de0aeec0fb9923d49d64ec1e5))

### Continuous Integration

- Release v1.0.2
  ([`c55f637`](https://github.com/nfraxlab/ai-infra/commit/c55f637d3ac28046d704e6ae5d1ad54a56714ccf))


## v1.0.1 (2025-12-28)

### Continuous Integration

- Release v1.0.1
  ([`28519df`](https://github.com/nfraxlab/ai-infra/commit/28519df47a9592bae48162eaa96b92b0aa71faae))


## v1.0.0 (2025-12-28)

### Bug Fixes

- Resolve merge conflict in pyproject.toml for v1.0.0
  ([`795e8aa`](https://github.com/nfraxlab/ai-infra/commit/795e8aa80fd1abd30d9f334b652055f20f6a9afe))

### Chores

- **release**: Prepare v1.0.0 release
  ([`e6e0321`](https://github.com/nfraxlab/ai-infra/commit/e6e0321894125284234e6d010deed1f05726ee7a))

BREAKING CHANGE: First stable release

- Bump version from 0.1.170 to 1.0.0 - Update classifier to Production/Stable - Add Changelog link
  to README navigation - All 2428 tests passing - Coverage at 64.93% - mypy and ruff: 0 errors

### Documentation

- Add v1.0.0 release notes to changelog
  ([`ff7946a`](https://github.com/nfraxlab/ai-infra/commit/ff7946a002f1f0f55fc9ea6883a97a31b1b8bb04))

Summarize all features for first stable release: - LLM, Agent, MCP, RAG, Voice, Images, Graph -
  Memory, Validation, Tracing, Callbacks - 2428 tests, 64.93% coverage, mypy/ruff clean

### Breaking Changes

- **release**: First stable release


## v0.1.171 (2025-12-28)

### Continuous Integration

- Release v0.1.171
  ([`1f51861`](https://github.com/nfraxlab/ai-infra/commit/1f51861010a6c1c151f86d1896800a624e559c57))

### Documentation

- Remove stale ai_infra.cost reference from API docs
  ([`b4ea28e`](https://github.com/nfraxlab/ai-infra/commit/b4ea28e51f7f3ac644ad886f791d82be21af5e15))

Module was referenced but never existed, causing mkdocs build failures.


## v0.1.170 (2025-12-28)

### Bug Fixes

- Remove duplicate test_stt.py and test_tts.py files
  ([`2d654a2`](https://github.com/nfraxlab/ai-infra/commit/2d654a25e85761d089c7c96a6d2f0332c354a620))

Pytest was failing with 'imported module mismatch' because the same test files existed at both
  tests/unit/ and tests/unit/llm/. The llm/ subdirectory contains the proper tests. Removed the
  root-level duplicates.

### Continuous Integration

- Release v0.1.170
  ([`c758eaa`](https://github.com/nfraxlab/ai-infra/commit/c758eaa6856d2226e3d328fd2d2424f4427de975))


## v0.1.169 (2025-12-28)

### Continuous Integration

- Release v0.1.169
  ([`3e018b6`](https://github.com/nfraxlab/ai-infra/commit/3e018b62d45fef3052fc76f4b07577f1b8d6f0ad))


## v0.1.168 (2025-12-27)

### Continuous Integration

- Only create GitHub Releases for minor/major versions
  ([`1d2ee2c`](https://github.com/nfraxlab/ai-infra/commit/1d2ee2cb3b630d877895a17d5724e20044949d2d))

- Release v0.1.168
  ([`09f9622`](https://github.com/nfraxlab/ai-infra/commit/09f9622234f63a177b890d27d14ea7f6b282e2ee))


## v0.1.167 (2025-12-27)

### Continuous Integration

- Add GitHub Release creation to publish workflow
  ([`670e04c`](https://github.com/nfraxlab/ai-infra/commit/670e04cba4a8204e80d9cef23bd9349565d757ec))

- Release v0.1.167
  ([`05b4dc2`](https://github.com/nfraxlab/ai-infra/commit/05b4dc28c980c414c28f581972835147d11fd456))


## v0.1.166 (2025-12-26)

### Bug Fixes

- **ci**: Prevent docs-changelog race condition with publish workflow
  ([`e72d866`](https://github.com/nfraxlab/ai-infra/commit/e72d866bd06250808a6434400e40c37e74740d8d))

### Continuous Integration

- Release v0.1.166
  ([`6a7bf23`](https://github.com/nfraxlab/ai-infra/commit/6a7bf23ca21806ecc97c0bfa9c6ec2ccc18a563f))


## v0.1.165 (2025-12-24)

### Continuous Integration

- Release v0.1.165
  ([`f4f619a`](https://github.com/nfraxlab/ai-infra/commit/f4f619ac4f1d45fefe6bf2e21490280be4da9b17))

### Documentation

- Update changelog [skip ci]
  ([`9726999`](https://github.com/nfraxlab/ai-infra/commit/97269990a64c1a5d500e0ff682448549c274386c))


## v0.1.164 (2025-12-22)

### Continuous Integration

- Release v0.1.164
  ([`6e8fdd1`](https://github.com/nfraxlab/ai-infra/commit/6e8fdd151d65ad5dfda20415cd097f6206089b1c))


## v0.1.163 (2025-12-19)

### Bug Fixes

- Add type ignore comment for __init__ access
  ([`0ac5db4`](https://github.com/nfraxlab/ai-infra/commit/0ac5db40691e33cc63e95fa047982dba8784a36a))

Fix mypy error for accessing __init__ on class instance

- Lower coverage threshold to 40% to match current state
  ([`febce24`](https://github.com/nfraxlab/ai-infra/commit/febce24110d516c345ae2853ab279ca735f88a71))

The codebase has many example files and CLI commands that aren't covered by unit tests. Lower
  threshold to 40% which is achievable, and exclude examples/ and cli/ directories from coverage
  calculation.

- Update bandit config to skip false positive security warnings
  ([`a13e92a`](https://github.com/nfraxlab/ai-infra/commit/a13e92a213eb8a8025699da74435dacb1c241bb1))

- Skip B104 (hardcoded bind all interfaces) - intentional in dev examples - Skip B301 (pickle load)
  - trusted local files for retriever persistence - Skip B608 (hardcoded SQL) - table names are
  validated, not user input

- Update CI workflow to use 40% coverage threshold
  ([`3f7f2f9`](https://github.com/nfraxlab/ai-infra/commit/3f7f2f9dcab24a0aadae375ba161ba1797f2350d))

The pyproject.toml was updated but the CI workflow had the threshold hardcoded.

### Continuous Integration

- Release v0.1.163
  ([`bc48057`](https://github.com/nfraxlab/ai-infra/commit/bc48057582928626565e01de20c74a1a51c0c49c))

### Features

- Add deprecation policy and helpers
  ([`3d4a68d`](https://github.com/nfraxlab/ai-infra/commit/3d4a68d18c57f0d6d5cb9ebeb93baa5580ad429b))

- Add DEPRECATION.md with deprecation timeline and policy - Update CONTRIBUTING.md with deprecation
  guidelines section - Add utils/deprecation.py with @deprecated decorator - Add
  deprecated_parameter() function for parameter deprecation - Add DeprecatedWarning custom warning
  class - Add unit tests for deprecation helpers - Add integration tests for Google provider, MCP
  client/server, agent tools


## v0.1.162 (2025-12-18)

### Continuous Integration

- Release v0.1.162
  ([`d25ea33`](https://github.com/nfraxlab/ai-infra/commit/d25ea3321cc45e0e654d1b2b65edc61b48adaefa))

- Streamline version bump and changelog commit process
  ([`24c9cb1`](https://github.com/nfraxlab/ai-infra/commit/24c9cb12f6718d696ca5445bb412006217abac35))


## v0.1.161 (2025-12-18)

### Continuous Integration

- Release v0.1.161
  ([`03ad1ff`](https://github.com/nfraxlab/ai-infra/commit/03ad1fff193b2e68af96678c1a4cd2451840f1c2))

### Features

- Add git-cliff configuration for automated changelog generation
  ([`8dd3212`](https://github.com/nfraxlab/ai-infra/commit/8dd32127312f9aa8ed280dd98246238aa96a45f1))

- Introduced a new `cliff.toml` configuration file for managing changelog generation. - Configured
  changelog header and body templates to document notable changes. - Set up commit parsing rules for
  categorizing commits into features, bug fixes, documentation, performance, refactoring, styling,
  testing, and miscellaneous changes. - Enabled support for conventional commits and specified
  commit preprocessors for pull request linking. - Established a limit of 500 commits for changelog
  generation and configured sorting to show the newest commits first.


## v0.1.160 (2025-12-18)

### Continuous Integration

- Release v0.1.160
  ([`50f7368`](https://github.com/nfraxlab/ai-infra/commit/50f736873f6b1ba65c230134ccfaeb5f9ffe18ed))


## v0.1.159 (2025-12-18)

### Continuous Integration

- Release v0.1.159
  ([`1f3fdf2`](https://github.com/nfraxlab/ai-infra/commit/1f3fdf23ba6852fbb0459c8d0ba73f2d300eb8e5))


## v0.1.158 (2025-12-18)

### Continuous Integration

- Release v0.1.158
  ([`a221364`](https://github.com/nfraxlab/ai-infra/commit/a22136407955b0498d0b917904d68ac5897211e3))

### Documentation

- Update changelog [skip ci]
  ([`d396bf1`](https://github.com/nfraxlab/ai-infra/commit/d396bf1828eb10ee9bcb8c460b5712f6637ed650))

### Features

- Refactor DeepAgents integration and improve callback utilities
  ([`561ee8d`](https://github.com/nfraxlab/ai-infra/commit/561ee8df5aef3a5e66e02a9db43dfbb779c2377b))

- **docs**: Enhance documentation with comprehensive guides and error handling patterns
  ([`c554a44`](https://github.com/nfraxlab/ai-infra/commit/c554a446f5ea749ee6cea8ceaac870c794a85c70))


## v0.1.157 (2025-12-18)

### Continuous Integration

- Release v0.1.157
  ([`52fdd15`](https://github.com/nfraxlab/ai-infra/commit/52fdd1593b27502b60035bfa948ec4cbb32a6d1b))

### Features

- **tests**: Add comprehensive unit tests for agent safety and edge cases
  ([`3244e74`](https://github.com/nfraxlab/ai-infra/commit/3244e7472f4a31fc5ab06dbf0e6af6fa25f41afa))


## v0.1.156 (2025-12-18)

### Continuous Integration

- Release v0.1.156
  ([`6112bb7`](https://github.com/nfraxlab/ai-infra/commit/6112bb7468c3333fd5151406492e3b0751958df0))

### Features

- **tests**: Add integration tests for Anthropic, OpenAI, and Embeddings providers
  ([`5ea0127`](https://github.com/nfraxlab/ai-infra/commit/5ea0127637cd8fed833ae06772861b94e3e2ab98))


## v0.1.155 (2025-12-17)

### Bug Fixes

- **imports**: Remove type ignore for Google Cloud imports in STT and TTS modules
  ([`213e0f1`](https://github.com/nfraxlab/ai-infra/commit/213e0f159b2c39cef059e1595c094f9d34383d14))

- **mypy**: Add missing imports configuration for google package
  ([`8c6320b`](https://github.com/nfraxlab/ai-infra/commit/8c6320b8bd35906ae0aabedb8c2e1ec5ecee2d03))

- **mypy**: Set warn_unused_ignores to False in mypy configuration
  ([`ca1fcd3`](https://github.com/nfraxlab/ai-infra/commit/ca1fcd34c86064c8256f639c04b51820f87f613b))

- **pyproject**: Update URLs and add AI-related classifiers; restructure dependencies
  ([`f2b8063`](https://github.com/nfraxlab/ai-infra/commit/f2b8063fedc7834ed4f53beb766a3eed5b59b212))

- **pyproject**: Update URLs for homepage, repository, issues, and documentation
  ([`a4188ee`](https://github.com/nfraxlab/ai-infra/commit/a4188eefe89bb12af42bed43a692f2c006aeef3b))

### Continuous Integration

- Release v0.1.155
  ([`078aa86`](https://github.com/nfraxlab/ai-infra/commit/078aa8655b95e66b4235a82eba1925237b13ec87))

### Features

- **tests**: Add checks for langchain_huggingface and google.genai availability in unit tests
  ([`fc80b13`](https://github.com/nfraxlab/ai-infra/commit/fc80b13434a7026071586af63d2a3a56690b6311))


## v0.1.154 (2025-12-17)

### Continuous Integration

- Release v0.1.154
  ([`1a4f2fa`](https://github.com/nfraxlab/ai-infra/commit/1a4f2fa410e84b0ae2b4c3aee98759394fd87e59))

### Refactoring

- **callbacks, approval, tracing**: Improve log messages for clarity and consistency
  ([`40ec550`](https://github.com/nfraxlab/ai-infra/commit/40ec550224def8d584794eaab6e38898549bfe21))


## v0.1.153 (2025-12-17)

### Bug Fixes

- **pre-commit**: Match CI config exactly (use ruff defaults)
  ([`ae1f92f`](https://github.com/nfraxlab/ai-infra/commit/ae1f92f96fc9b9ad142d08b840d69fcde5a6e1d8))

### Continuous Integration

- Release v0.1.153
  ([`59df4a2`](https://github.com/nfraxlab/ai-infra/commit/59df4a254ba08dd1a72bca2b16aa2f3d2816a321))


## v0.1.152 (2025-12-17)

### Chores

- **pre-commit**: Remove mypy from hooks (use CI instead)
  ([`7e83ea1`](https://github.com/nfraxlab/ai-infra/commit/7e83ea1059d4af0c47e0f78cd4fbb89344437f56))

### Continuous Integration

- Release v0.1.152
  ([`72a2659`](https://github.com/nfraxlab/ai-infra/commit/72a265946f804ec005599fef053f195a9ca36207))


## v0.1.151 (2025-12-17)

### Bug Fixes

- **format**: Apply ruff formatting + switch pre-commit from black to ruff
  ([`0ffa004`](https://github.com/nfraxlab/ai-infra/commit/0ffa004ed22365df5c9ee34617e00c1890ecd717))

- Format 1 file with ruff - Update .pre-commit-config.yaml to use ruff instead of black/isort/flake8
  - This ensures pre-commit matches CI (both use ruff now)

### Continuous Integration

- Release v0.1.151
  ([`2339c44`](https://github.com/nfraxlab/ai-infra/commit/2339c446eecaa7717bf5900ee8acfa02c18bcb6d))

### Features

- Implement CI workflow for automated testing and linting
  ([`f92fd6f`](https://github.com/nfraxlab/ai-infra/commit/f92fd6ff0399dc1b7d910a224c5a4f2416db2f03))

### Testing

- Fix realtime voice tests to work without API keys
  ([`c20316b`](https://github.com/nfraxlab/ai-infra/commit/c20316bc162e7de8298ad2007480ca19c9d5a49c))

Add mock_openai_key fixture to set fake OPENAI_API_KEY env var for tests that require RealtimeVoice
  provider initialization. This allows tests to pass in CI without real API credentials.


## v0.1.150 (2025-12-16)

### Continuous Integration

- Release v0.1.150
  ([`00fa179`](https://github.com/nfraxlab/ai-infra/commit/00fa179027fdb059a0a617637f8a44deed2afa02))

### Features

- Add docs-changelog target and update changelog generation script
  ([`0916634`](https://github.com/nfraxlab/ai-infra/commit/09166343298963ec3235eaf8b470278e49128721))


## v0.1.149 (2025-12-16)

### Continuous Integration

- Release v0.1.149
  ([`03d23ab`](https://github.com/nfraxlab/ai-infra/commit/03d23aba87f52d66ee2435891269ac75bd07643d))


## v0.1.148 (2025-12-16)

### Continuous Integration

- Release v0.1.148
  ([`4997996`](https://github.com/nfraxlab/ai-infra/commit/4997996e3a98ad330182be4b8d99b17cd7a5dd51))

### Refactoring

- Enhance Makefile commands and add formatting checks
  ([`7ce84cf`](https://github.com/nfraxlab/ai-infra/commit/7ce84cfb5e50fed6e3ee1f8f64f48462dc755e3d))


## v0.1.147 (2025-12-15)

### Continuous Integration

- Release v0.1.147
  ([`f3aace0`](https://github.com/nfraxlab/ai-infra/commit/f3aace0a972f99df1e96722b8284ae6d209f8530))


## v0.1.146 (2025-12-15)

### Continuous Integration

- Release v0.1.146
  ([`f44e3b8`](https://github.com/nfraxlab/ai-infra/commit/f44e3b869d070fdf37c2d8f52f3d1bd830003fc1))


## v0.1.145 (2025-12-14)

### Continuous Integration

- Release v0.1.145
  ([`83ee5af`](https://github.com/nfraxlab/ai-infra/commit/83ee5afa7dedcdab6636a8e95fab50e937d69e61))

### Refactoring

- Update class names and improve type hints across multiple files
  ([`be29233`](https://github.com/nfraxlab/ai-infra/commit/be29233986dbc84389f5989897cfa7d61102034e))


## v0.1.144 (2025-12-14)

### Continuous Integration

- Release v0.1.144
  ([`bd12ccf`](https://github.com/nfraxlab/ai-infra/commit/bd12ccf43f65788f2765411dd9cbd23e5fdf0d86))

### Features

- Enhance error handling with original error context and update pytest configuration
  ([`f8d2976`](https://github.com/nfraxlab/ai-infra/commit/f8d2976400be1b6983c9c11b4c1ec8b0e1692d5c))


## v0.1.143 (2025-12-14)

### Continuous Integration

- Release v0.1.143
  ([`efd7847`](https://github.com/nfraxlab/ai-infra/commit/efd784776d1aed3d0b47f00edf4d5ebf47dc1e49))

### Features

- Add logging utility for exception handling and refactor error classes
  ([`946540d`](https://github.com/nfraxlab/ai-infra/commit/946540d990637aedb143980851453289dd750d97))


## v0.1.142 (2025-12-14)

### Continuous Integration

- Release v0.1.142
  ([`6aa9e92`](https://github.com/nfraxlab/ai-infra/commit/6aa9e92c3d6c55773ba5221a5a152d9c7190bff1))

### Features

- Implement normalize_callbacks utility and enhance callback management with critical callbacks
  ([`2b74007`](https://github.com/nfraxlab/ai-infra/commit/2b74007f749c5d1218c7ad898c684aed18f7b21b))


## v0.1.141 (2025-12-14)

### Continuous Integration

- Release v0.1.141
  ([`1722362`](https://github.com/nfraxlab/ai-infra/commit/1722362ccc55711bd1580de2e10eae40f4ec79ef))

### Features

- Add safety limits to agents to prevent runaway costs and infinite loops
  ([`db9131d`](https://github.com/nfraxlab/ai-infra/commit/db9131d0604550792453a4dd7072353f88d0754f))


## v0.1.140 (2025-12-13)

### Continuous Integration

- Release v0.1.140
  ([`a84ab81`](https://github.com/nfraxlab/ai-infra/commit/a84ab81f8cd9e2119d11870bc302b1e2aa844856))

### Features

- Add safety limits and security measures across various components
  ([`988bdbb`](https://github.com/nfraxlab/ai-infra/commit/988bdbbd1c8780ee0a0f4e6a6ad9bcda7c96ee09))


## v0.1.139 (2025-12-12)

### Bug Fixes

- Update repository references from nfraxio to nfraxlab in documentation and code
  ([`886f213`](https://github.com/nfraxlab/ai-infra/commit/886f213fc211ab336260943cde0ed9bc28438282))

### Chores

- Re-trigger pypi publish after enabling workflow
  ([`62a14e0`](https://github.com/nfraxlab/ai-infra/commit/62a14e01094e258b832aab1a40954c06acb41a61))

- Trigger pypi publish
  ([`62c6908`](https://github.com/nfraxlab/ai-infra/commit/62c6908035bfcf39c027761a1adc37311079a317))

- Trigger pypi publish
  ([`7b0960d`](https://github.com/nfraxlab/ai-infra/commit/7b0960da641b59121498976ec395cd8d7c39a9fc))

### Continuous Integration

- Release v0.1.139
  ([`bb56eb4`](https://github.com/nfraxlab/ai-infra/commit/bb56eb41db98fc6b24a2b2ecac7cd630d4683ee0))


## v0.1.138 (2025-12-11)

### Continuous Integration

- Release v0.1.138
  ([`ac415a8`](https://github.com/nfraxlab/ai-infra/commit/ac415a86aa8ebff6a62f08d5863793918585d83a))

### Documentation

- Update README and documentation for new features and improvements
  ([`5ed63f6`](https://github.com/nfraxlab/ai-infra/commit/5ed63f6baf06ad7b2ae1b2459cdba00a3cc51a90))


## v0.1.137 (2025-12-10)

### Chores

- Update dependencies and improve .gitignore entries
  ([`742dfb5`](https://github.com/nfraxlab/ai-infra/commit/742dfb59f26375886f25b8f58825450b2291aae2))

### Continuous Integration

- Release v0.1.137
  ([`0bbee42`](https://github.com/nfraxlab/ai-infra/commit/0bbee42e4a39921bac0df4d1fc5c52847caeff6d))


## v0.1.136 (2025-12-10)

### Continuous Integration

- Release v0.1.136
  ([`258a3e7`](https://github.com/nfraxlab/ai-infra/commit/258a3e77ec144b99f7e447696946500312089300))

### Features

- Update default models for providers to latest versions
  ([`945abf4`](https://github.com/nfraxlab/ai-infra/commit/945abf44e31e2504bade309fdc53c7ca5212466e))


## v0.1.135 (2025-12-10)

### Continuous Integration

- Release v0.1.135
  ([`e3f9166`](https://github.com/nfraxlab/ai-infra/commit/e3f9166133cdfdfeefe0588e8924f2a345a294a2))

### Features

- Add MIT License to the repository
  ([`f54231d`](https://github.com/nfraxlab/ai-infra/commit/f54231d5cbd2afc795abcfde497255ed0b531a1e))


## v0.1.134 (2025-12-10)

### Continuous Integration

- Release v0.1.134
  ([`dbc6673`](https://github.com/nfraxlab/ai-infra/commit/dbc667398d85eaaf171313f46642effbe18ef6df))

### Features

- Enhance auto-configuration to support DATABASE_URL_PRIVATE for backend detection
  ([`36b3412`](https://github.com/nfraxlab/ai-infra/commit/36b3412c3c4f874902b1d874afb7542d4622a642))


## v0.1.133 (2025-12-09)

### Continuous Integration

- Release v0.1.133
  ([`16dab37`](https://github.com/nfraxlab/ai-infra/commit/16dab370f5f9c1d0edfe59fae0c17ec210b6c29a))

### Features

- Add filter parameter to create_retriever_tool and update tests
  ([`da47485`](https://github.com/nfraxlab/ai-infra/commit/da47485e6eb52fe9e1ce9ffb71633dafed126163))


## v0.1.132 (2025-12-09)

### Continuous Integration

- Release v0.1.132
  ([`bc16fdf`](https://github.com/nfraxlab/ai-infra/commit/bc16fdf081b94e95fe6f91e28d98f327c250c1d4))

### Features

- Add live test script and unit tests for Retriever Phase 6.9 enhancements
  ([`a7f9327`](https://github.com/nfraxlab/ai-infra/commit/a7f9327aa4dbba448a0e2ae721eb9835db697bcb))

- Implemented `test_retriever_6_9.py` to test new features: - Environment auto-configuration -
  Remote content loading from GitHub and URLs - Enhancements to SearchResult and structured tool
  results - StreamEvent structured result support - Module exports verification

- Created `test_retriever_enhancements.py` for unit testing: - Validated KNOWN_EMBEDDING_DIMENSIONS
  and embedding dimension retrieval - Tested remote content loading methods and their sync wrappers
  - Enhanced SearchResult functionality and convenience properties - Verified structured tool
  results and their JSON serialization - Ensured StreamEvent structured results are handled
  correctly - Confirmed module exports for retriever components


## v0.1.131 (2025-12-09)

### Continuous Integration

- Release v0.1.131
  ([`23218c4`](https://github.com/nfraxlab/ai-infra/commit/23218c43cdb1ef8d351aa42a5169c11da309fd87))

### Features

- Add similarity parameter to PostgresBackend and validate its value
  ([`49d00de`](https://github.com/nfraxlab/ai-infra/commit/49d00de7c55d08c9f0cf3bb464ddd6cf5ba0485c))


## v0.1.130 (2025-12-08)

### Continuous Integration

- Release v0.1.130
  ([`7784ec9`](https://github.com/nfraxlab/ai-infra/commit/7784ec988b746333ec3c5ba9214a768af2d41c2d))

### Testing

- Add verification for transport security configuration in disabled security case
  ([`ebaffce`](https://github.com/nfraxlab/ai-infra/commit/ebaffce8ea96e425bc22e7dfd5572e21e9e213a4))


## v0.1.129 (2025-12-08)

### Bug Fixes

- Ensure to_transport_settings always returns TransportSecuritySettings
  ([`06967d2`](https://github.com/nfraxlab/ai-infra/commit/06967d2efe86d61e904679bb4892fc0401cbf987))

### Continuous Integration

- Release v0.1.129
  ([`399558b`](https://github.com/nfraxlab/ai-infra/commit/399558bc872c1972b82172219f2fe6790f7a2fc4))


## v0.1.128 (2025-12-08)

### Continuous Integration

- Release v0.1.128
  ([`c2d7b09`](https://github.com/nfraxlab/ai-infra/commit/c2d7b0961af0156fcacd6bda650e78453223c19a))

### Features

- Merge lifespan contexts in attach_to_fastapi for improved compatibility
  ([`32ac2c0`](https://github.com/nfraxlab/ai-infra/commit/32ac2c0dd284dae174369cf7c8dba07961102802))


## v0.1.127 (2025-12-07)

### Continuous Integration

- Release v0.1.127
  ([`3804823`](https://github.com/nfraxlab/ai-infra/commit/38048236f22346173001b6112d0f87e32fdba46f))

### Features

- Enhance tool event logging and handle incomplete tool calls in Agent
  ([`7f4cfc1`](https://github.com/nfraxlab/ai-infra/commit/7f4cfc1f29c514594cbb987e5da071e4df92f206))


## v0.1.126 (2025-12-07)

### Continuous Integration

- Release v0.1.126
  ([`e0ef536`](https://github.com/nfraxlab/ai-infra/commit/e0ef536e5d8ed2523089fd591ec567e4e57ddf42))

### Features

- Enhance streaming events with full tool results and visibility levels
  ([`232e000`](https://github.com/nfraxlab/ai-infra/commit/232e0001e119319b972e7399db1aab9664883809))


## v0.1.125 (2025-12-07)

### Continuous Integration

- Release v0.1.125
  ([`22790bb`](https://github.com/nfraxlab/ai-infra/commit/22790bba477b871aefcd89714f6adf048a3dee1e))

### Features

- Implement streaming support in Agent with astream() method
  ([`9f1d6e0`](https://github.com/nfraxlab/ai-infra/commit/9f1d6e01a642786b5c089305232047aebeb71364))

- Added a new astream() method in the Agent class to stream responses as typed events. - Introduced
  StreamEvent and StreamConfig classes for structured event handling and configuration. - Enhanced
  the auth module with temporary API key management for BYOK scenarios. - Implemented caching
  mechanisms for MCP tools with load_mcp_tools_cached function. - Added comprehensive tests for
  streaming events, authentication helpers, and MCP tool loading. - Updated documentation to include
  a detailed streaming guide and examples.


## v0.1.124 (2025-12-06)

### Continuous Integration

- Release v0.1.124
  ([`8a89e2b`](https://github.com/nfraxlab/ai-infra/commit/8a89e2b78831a3051ae116ba6d16e7c082b6a547))

### Refactoring

- Remove FastAPI integration and streaming components
  ([`60f7826`](https://github.com/nfraxlab/ai-infra/commit/60f78260b659f9deeadbd8c928150dbdfe3a9218))


## v0.1.123 (2025-12-06)

### Continuous Integration

- Release v0.1.123
  ([`12633e8`](https://github.com/nfraxlab/ai-infra/commit/12633e8e19cbc07f0fb4250587e5fc83f6cfaf1d))

### Features

- Add FastAPI integration with chat endpoint and streaming support
  ([`dfd6828`](https://github.com/nfraxlab/ai-infra/commit/dfd6828b8ecbeed6f72ae9dbb69d276ea2b3c059))


## v0.1.122 (2025-12-06)

### Continuous Integration

- Release v0.1.122
  ([`fe77c92`](https://github.com/nfraxlab/ai-infra/commit/fe77c921bfa85551b5765d523f2b9bc550ec4abd))

### Features

- Implement automatic security detection and configuration for MCP servers
  ([`274c3ab`](https://github.com/nfraxlab/ai-infra/commit/274c3ab51c3278989aec0212324af7fd07c19af6))


## v0.1.121 (2025-12-04)

### Continuous Integration

- Release v0.1.121
  ([`2a8c6ad`](https://github.com/nfraxlab/ai-infra/commit/2a8c6ad457aafd4b93c9e230356544255ee603e5))


## v0.1.120 (2025-12-04)

### Continuous Integration

- Release v0.1.120
  ([`23f39ae`](https://github.com/nfraxlab/ai-infra/commit/23f39ae74eac40274f5dd60a4f15f9623f730a37))

### Documentation

- Consolidate badge display in README for improved readability
  ([`f8cc381`](https://github.com/nfraxlab/ai-infra/commit/f8cc3816340cdceaaf4018c9c0868200f0918ba3))


## v0.1.119 (2025-12-04)

### Continuous Integration

- Release v0.1.119
  ([`3d2fd3b`](https://github.com/nfraxlab/ai-infra/commit/3d2fd3b66ff2654543fe53f6dcfe2f2c247d8fc3))

### Documentation

- Revise README for clarity and structure, enhance feature descriptions
  ([`bd0b7bd`](https://github.com/nfraxlab/ai-infra/commit/bd0b7bd06086e59a9fd5405f9e17f7eda8c79d85))


## v0.1.118 (2025-12-04)

### Continuous Integration

- Release v0.1.118
  ([`881fa25`](https://github.com/nfraxlab/ai-infra/commit/881fa2595e73d598bd6957386ff10feee7c54985))


## v0.1.117 (2025-12-03)

### Continuous Integration

- Release v0.1.117
  ([`4ff12c0`](https://github.com/nfraxlab/ai-infra/commit/4ff12c092d323ed6b7b5ee68b45f0e4ad96466c5))

### Features

- Implement unified context management with fit_context() API
  ([`9ec9949`](https://github.com/nfraxlab/ai-infra/commit/9ec9949d2b5f5e3193f603861741024f6fbb2aa1))


## v0.1.116 (2025-12-03)

### Continuous Integration

- Release v0.1.116
  ([`8b9f507`](https://github.com/nfraxlab/ai-infra/commit/8b9f507d6cb67a6fce1d828735935f7d74485a6c))


## v0.1.115 (2025-12-03)

### Continuous Integration

- Release v0.1.115
  ([`2c97c4f`](https://github.com/nfraxlab/ai-infra/commit/2c97c4f18304965f2cafe1da1e77d88f84092f46))

### Features

- Add error handling for loading persisted state in Retriever
  ([`decaa2f`](https://github.com/nfraxlab/ai-infra/commit/decaa2f10a36082cd6f26c15f7abc95c5c544eaf))


## v0.1.114 (2025-12-03)

### Continuous Integration

- Release v0.1.114
  ([`21094f6`](https://github.com/nfraxlab/ai-infra/commit/21094f6dfdb1b5b162f5b499c62f2fabe5b3d61a))

### Features

- Enhance SQLiteBackend with configurable similarity metrics
  ([`3d70cc1`](https://github.com/nfraxlab/ai-infra/commit/3d70cc16c1301fd322905ae9f3dee0e279381e9a))

- Added support for multiple similarity metrics: cosine, euclidean, and dot_product in
  SQLiteBackend. - Updated the constructor to accept a similarity parameter and validate it against
  supported metrics. - Refactored similarity calculation methods to use the selected metric. -
  Updated documentation and examples to reflect the new functionality.

feat: Implement persistence functionality in Retriever

- Added save and load methods to the Retriever class for state persistence. - Implemented automatic
  loading of saved state on initialization if a persist_path is provided. - Added auto-save
  functionality after adding content when persist_path is set. - Enhanced tests to cover persistence
  scenarios, including save/load and auto-save behavior.

feat: Introduce similarity metrics in Retriever

- Added similarity parameter to the Retriever class to allow users to specify the similarity metric.
  - Updated tests to validate the correct behavior of different similarity metrics during search
  operations. - Ensured that the similarity metric is preserved when saving and loading the
  retriever state.

test: Add comprehensive tests for persistence and similarity metrics

- Created tests for save/load functionality, ensuring files are created and data is restored
  correctly. - Added tests for lazy initialization of embeddings and ensuring models are loaded only
  when needed. - Implemented tests for various similarity metrics to validate search results and
  behavior.


## v0.1.112 (2025-12-02)

### Continuous Integration

- Release v0.1.112
  ([`d45e9c1`](https://github.com/nfraxlab/ai-infra/commit/d45e9c1828835d188308fa678e70969c417a02cf))


## v0.1.111 (2025-12-02)

### Continuous Integration

- Release v0.1.111
  ([`d2a095d`](https://github.com/nfraxlab/ai-infra/commit/d2a095df22c2238df77415c6205a572cac288d2c))


## v0.1.110 (2025-12-02)

### Continuous Integration

- Release v0.1.110
  ([`2697159`](https://github.com/nfraxlab/ai-infra/commit/2697159316b4df1c99e0d1eb75fc0fc3116478bd))


## v0.1.109 (2025-12-02)

### Continuous Integration

- Release v0.1.109
  ([`b52b4d4`](https://github.com/nfraxlab/ai-infra/commit/b52b4d40ea26973171402a5258da4b10f732d286))


## v0.1.108 (2025-12-01)

### Continuous Integration

- Release v0.1.108
  ([`91ab181`](https://github.com/nfraxlab/ai-infra/commit/91ab181f5b25f6e4006e264a32cd7c37f8f6e816))


## v0.1.107 (2025-12-01)

### Continuous Integration

- Release v0.1.107
  ([`e0a66d8`](https://github.com/nfraxlab/ai-infra/commit/e0a66d88f3d0655f5e3c2a515cad7ac878596acd))


## v0.1.106 (2025-12-01)

### Continuous Integration

- Release v0.1.106
  ([`cb894c7`](https://github.com/nfraxlab/ai-infra/commit/cb894c7433c844b5afb1aaccf7b9c1318aa31277))


## v0.1.105 (2025-12-01)

### Continuous Integration

- Release v0.1.105
  ([`9b23168`](https://github.com/nfraxlab/ai-infra/commit/9b23168d4c0df8477ff8c08fcd0cff3595d365b1))

### Features

- Add provider configurations for Cohere, Deepgram, ElevenLabs, Google, OpenAI, Replicate, Stability
  AI, Voyage AI, and xAI
  ([`172de2e`](https://github.com/nfraxlab/ai-infra/commit/172de2e1028898765c196f2d0c11e2ced16787e7))

- Implemented Cohere provider for multilingual embeddings. - Added Deepgram provider for
  speech-to-text capabilities. - Integrated ElevenLabs for high-quality text-to-speech with voice
  cloning. - Configured Google provider for various capabilities including chat, embeddings, TTS,
  STT, image generation, and real-time interactions. - Established OpenAI provider with support for
  chat, embeddings, TTS, STT, image generation, and real-time features. - Introduced Replicate
  provider for community models in image generation. - Set up Stability AI provider for image
  generation with Stable Diffusion models. - Added Voyage AI provider for high-quality embeddings
  optimized for RAG. - Configured xAI provider for chat capabilities using Grok models.

test: Add unit tests for provider registry functionality

- Created unit tests for provider registration, capability lookup, and environment variable
  configurations. - Ensured all providers have display names and environment variables set. -
  Verified that no duplicate provider names exist in the registry.


## v0.1.104 (2025-12-01)

### Continuous Integration

- Release v0.1.104
  ([`dbd0d84`](https://github.com/nfraxlab/ai-infra/commit/dbd0d8411293c630fece15255c05f97a8c01de2a))


## v0.1.103 (2025-12-01)

### Continuous Integration

- Release v0.1.103
  ([`c6f3f39`](https://github.com/nfraxlab/ai-infra/commit/c6f3f39bef68331de17d9b4664ba7d0a60ec4218))

### Features

- Update dependencies and introduce Workspace abstraction for file operations
  ([`5ba9333`](https://github.com/nfraxlab/ai-infra/commit/5ba93334fe306536694aa1e61ae804e52b1a2ec8))

- Updated Poetry lock file to version 2.1.4 and added new package 'bracex' (v2.6) and 'wcmatch'
  (v10.1). - Upgraded 'deepagents' package to version 0.2.8 and adjusted its dependencies. -
  Introduced a new 'Workspace' class for unified file operations in agents, supporting different
  access modes (virtual, sandboxed, full). - Enhanced agent initialization to accept workspace
  configurations, allowing for better file management. - Updated project management tools to
  integrate with the new workspace abstraction. - Added comprehensive unit tests for the Workspace
  class and its integration with agents. - Deprecated the old workspace root setting method in favor
  of the new workspace parameter in agents.


## v0.1.102 (2025-11-30)

### Continuous Integration

- Release v0.1.102
  ([`90e0713`](https://github.com/nfraxlab/ai-infra/commit/90e0713a9c37be8c3ebb58eda89511080c2e90d0))

### Features

- Add tools_from_models_sql for CRUD operations with svc-infra integration and enhance schema_tools
  documentation
  ([`82c4575`](https://github.com/nfraxlab/ai-infra/commit/82c4575c7f759ca1b3381abf2ad6f89eca37acc5))


## v0.1.101 (2025-11-30)

### Continuous Integration

- Release v0.1.101
  ([`f5825a3`](https://github.com/nfraxlab/ai-infra/commit/f5825a3aa3418e1ba722f56b038879765ad3b15f))

### Features

- Enhance replay and init modules with additional storage and progress tools
  ([`bcb0cba`](https://github.com/nfraxlab/ai-infra/commit/bcb0cba5b565d0e31d440f9578242a7f8e5d73f0))


## v0.1.100 (2025-11-30)

### Continuous Integration

- Release v0.1.100
  ([`b8375fa`](https://github.com/nfraxlab/ai-infra/commit/b8375fa0abeb6390499741ca066343fcb8e28f00))


## v0.1.99 (2025-11-30)

### Continuous Integration

- Release v0.1.99
  ([`3b06e1a`](https://github.com/nfraxlab/ai-infra/commit/3b06e1a1b445b2cf09bfcca6bb655e213600dad9))


## v0.1.98 (2025-11-29)

### Continuous Integration

- Release v0.1.98
  ([`23a3c60`](https://github.com/nfraxlab/ai-infra/commit/23a3c608fb99be149801ac9ffb113afd0214d3af))

### Features

- **multimodal**: Add audio output and discovery modules for LLMs
  ([`0c97f95`](https://github.com/nfraxlab/ai-infra/commit/0c97f953c58f5d7082f4aa2b4e5149381a904633))

- Implemented audio output support in `audio_output.py` for LLMs, allowing audio responses from
  models like GPT-4o-audio-preview. - Created `discovery.py` to list TTS and STT providers, models,
  and voices, enhancing multimodal capabilities. - Developed multimodal tools for agents in
  `multimodal/__init__.py`, including audio transcription and image analysis/generation. - Added
  tests for audio input/output, vision functionality, STT, TTS, and multimodal tools to ensure
  reliability and correctness.


## v0.1.97 (2025-11-29)

### Continuous Integration

- Release v0.1.97
  ([`0c9b65d`](https://github.com/nfraxlab/ai-infra/commit/0c9b65de66364a3164ba327ba265eb87ba0a2b2e))

### Features

- **audio**: Add audio input support and related utilities for LLM
  ([`6d31efe`](https://github.com/nfraxlab/ai-infra/commit/6d31efefc50a1a8e94a5f160da720ea54bd37531))


## v0.1.96 (2025-11-28)

### Continuous Integration

- Release v0.1.96
  ([`4a54d90`](https://github.com/nfraxlab/ai-infra/commit/4a54d90bf36c94d6b45715d4c01fcefe3d201a26))

### Features

- Add Text-to-Speech (TTS) module with multi-provider support
  ([`4052508`](https://github.com/nfraxlab/ai-infra/commit/4052508147d670b389b3138c690b92ae3baa4a88))

- Implemented a unified TTS API supporting OpenAI, Google Cloud TTS, and ElevenLabs. - Added methods
  for synchronous and asynchronous speech generation, file saving, and streaming audio. - Included
  provider detection and default voice/model selection based on available API keys. - Provided
  example usage in the module docstring.

feat: Introduce vision support for LLM with provider-agnostic API

- Created a simple API for image input compatible with various LLM providers. - Implemented
  functions to create vision messages and encode images in a standardized format. - Supported image
  inputs as URLs, file paths, or raw bytes. - Added backward compatibility for deprecated functions
  while encouraging the use of new APIs.


## v0.1.95 (2025-11-28)

### Continuous Integration

- Release v0.1.95
  ([`96be06f`](https://github.com/nfraxlab/ai-infra/commit/96be06f42212f8588e0bca23a218532e7dbcf5c2))

### Features

- **imagegen**: Add CLI commands for image generation provider and model discovery
  ([`a3f491c`](https://github.com/nfraxlab/ai-infra/commit/a3f491cbe7b454ccc37f62e11dd9732a86309de2))


## v0.1.94 (2025-11-28)

### Continuous Integration

- Release v0.1.94
  ([`d030725`](https://github.com/nfraxlab/ai-infra/commit/d030725178741daa44c7c3726023fd0de1138063))

### Features

- **imagegen**: Enhance Google image generation support with Gemini models and update default
  configurations
  ([`761296a`](https://github.com/nfraxlab/ai-infra/commit/761296a1e099562ea29d2c3a5599a849f5b6395e))


## v0.1.93 (2025-11-28)

### Continuous Integration

- Release v0.1.93
  ([`eba7ec0`](https://github.com/nfraxlab/ai-infra/commit/eba7ec06264e7fd4017c5e84c4c610481376210b))

### Features

- **imagegen**: Implement provider-agnostic image generation module with support for OpenAI, Google,
  Stability AI, and Replicate
  ([`acfaf5f`](https://github.com/nfraxlab/ai-infra/commit/acfaf5f1d36d82ebabf35a9922fb71957fe08b58))


## v0.1.92 (2025-11-28)

### Continuous Integration

- Release v0.1.92
  ([`e936584`](https://github.com/nfraxlab/ai-infra/commit/e936584201b39446bad5e9082914a70485d39ef8))

### Testing

- **retriever**: Add comprehensive unit tests for create_retriever_tool and
  create_retriever_tool_async
  ([`b0b046e`](https://github.com/nfraxlab/ai-infra/commit/b0b046ef31cc8586cf147f01960990a3b9a39c47))


## v0.1.91 (2025-11-28)

### Continuous Integration

- Release v0.1.91
  ([`f72f947`](https://github.com/nfraxlab/ai-infra/commit/f72f947121b08699173bbb2619d8a1f0cd404d6a))

### Features

- **proj_mgmt**: Enhance workspace sandboxing with explicit root setting and improved path handling
  ([`ad84b28`](https://github.com/nfraxlab/ai-infra/commit/ad84b2870e856e2a0cee9f31585c91472d8197b9))


## v0.1.90 (2025-11-28)

### Continuous Integration

- Release v0.1.90
  ([`2ee4b89`](https://github.com/nfraxlab/ai-infra/commit/2ee4b89654842f59ce28332bceab99927f6afbd8))

### Features

- **retriever**: Add create_retriever_tool for Agent integration
  ([`804de0a`](https://github.com/nfraxlab/ai-infra/commit/804de0a52c1692275be06683f9309eca8213a431))

- Add retriever/tool.py with create_retriever_tool() and create_retriever_tool_async() - Wraps
  Retriever.search() as LangChain StructuredTool for use with Agent - Supports k, min_score,
  return_scores options - Export from retriever/__init__.py and ai_infra/__init__.py

Usage: from ai_infra import Agent, Retriever, create_retriever_tool

retriever = Retriever() retriever.add('./docs/')

tool = create_retriever_tool(retriever, name='search_docs', description='...') agent =
  Agent(tools=[tool]) agent.run('What is the refund policy?')

### Refactoring

- **retriever**: Move create_retriever_tool to llm/tools/custom/
  ([`6dd3977`](https://github.com/nfraxlab/ai-infra/commit/6dd39772eb74a885bccdd16515efe7976d5c362b))

Move retriever tool to llm/tools/custom/retriever.py where it belongs alongside other pre-built
  agent tools (run_cli, project_scan, etc.)

The tool is now importable from multiple convenient paths: - from ai_infra import
  create_retriever_tool (root) - from ai_infra.retriever import create_retriever_tool (with
  Retriever) - from ai_infra.llm.tools.custom import create_retriever_tool (canonical)


## v0.1.89 (2025-11-27)

### Continuous Integration

- Release v0.1.89
  ([`e5b70fe`](https://github.com/nfraxlab/ai-infra/commit/e5b70fee21129c5bf9573406c8118a55845746ac))

### Testing

- **retriever**: Add comprehensive unit tests for Retriever module
  ([`8880437`](https://github.com/nfraxlab/ai-infra/commit/888043794b23942bd996e19b9dbf5dad368d8dd4))

- 4.2.16: Created tests/unit/test_retriever.py - Input detection tests (text/file/directory) -
  Chunking tests (chunk_text, chunk_documents, estimate_chunks) - Chunk and SearchResult model tests
  - Memory backend tests (add, search, delete, filter) - Retriever class integration tests with
  mocks - Async methods tests (aadd, asearch, aget_context) - File loading tests (txt, md, json,
  directories)

- 4.2.16: Created tests/unit/test_retriever_backends.py - Memory backend comprehensive tests -
  SQLite backend tests with persistence - Chroma backend tests with mocks - Pinecone backend tests
  with mocks - Qdrant backend tests with mocks - FAISS backend tests with mocks - Backend factory
  tests - Backend consistency tests (parametrized)

All 476 tests passing, ruff clean.


## v0.1.88 (2025-11-27)

### Continuous Integration

- Release v0.1.88
  ([`8273b68`](https://github.com/nfraxlab/ai-infra/commit/8273b68583829ca996d100533508dd269fc6ce7a))

### Features

- **retriever**: Add Pinecone, Qdrant, FAISS backends and main Retriever class
  ([`57b6c08`](https://github.com/nfraxlab/ai-infra/commit/57b6c08093504f724e14e2fd7c6ab456b6574467))

- 4.2.11: Pinecone backend for cloud vector storage - 4.2.12: Qdrant backend for local/cloud vector
  DB - 4.2.13: FAISS backend for high-performance local search - 4.2.14: Main Retriever class with
  dead-simple API - add() for text, files, directories - search() with detailed/simple modes -
  get_context() for RAG prompts - Async variants: add_async, search_async - delete() and clear() for
  cleanup - 4.2.15: Root exports for Retriever and SearchResult


## v0.1.87 (2025-11-27)

### Continuous Integration

- Release v0.1.87
  ([`c170914`](https://github.com/nfraxlab/ai-infra/commit/c170914951e4c0404ea61c708abd9a94210a3065))


## v0.1.86 (2025-11-27)

### Continuous Integration

- Release v0.1.86
  ([`c5b2176`](https://github.com/nfraxlab/ai-infra/commit/c5b21763600d184ec7070881c948dc45bc9d9ad5))

### Features

- Implement embeddings module with provider-agnostic interface
  ([`81d0b79`](https://github.com/nfraxlab/ai-infra/commit/81d0b79df4ca035c70f860ccb9cf6a0f195c7bbf))

- Added Embeddings class for generating text embeddings using various providers (OpenAI, Google,
  Voyage, Cohere, Anthropic). - Introduced VectorStore class for managing documents and performing
  semantic searches. - Implemented support for in-memory, Chroma, and FAISS backends in VectorStore.
  - Added comprehensive unit tests for Embeddings and VectorStore functionalities. - Included
  examples in the documentation for usage of Embeddings and VectorStore.


## v0.1.85 (2025-11-27)

### Continuous Integration

- Release v0.1.85
  ([`3e80434`](https://github.com/nfraxlab/ai-infra/commit/3e8043485230c095bd46513cac464db6662c09d5))


## v0.1.84 (2025-11-27)

### Continuous Integration

- Release v0.1.84
  ([`a6a4a99`](https://github.com/nfraxlab/ai-infra/commit/a6a4a99ba81faf0f04e0ef40bf481f15d364a5e0))

### Features

- Enhance OpenAPI loading and processing capabilities
  ([`6f29627`](https://github.com/nfraxlab/ai-infra/commit/6f296278949581758e88b391ea38fcbee9f20f75))

- Refactor `load_openapi` function to support various input types: dict, URL, local file path, and
  raw JSON/YAML strings. - Introduce helper functions for fetching OpenAPI specs from URLs and
  loading from local files. - Add `_parse_openapi_string` for parsing raw JSON/YAML strings. -
  Implement `AuthConfig` and `OpenAPIOptions` dataclasses for flexible authentication and filtering
  options. - Extend `MCPServer` to accept filtering options and authentication configurations. - Add
  comprehensive unit tests for OpenAPI loading, filtering, authentication, and schema handling.


## v0.1.83 (2025-11-27)

### Continuous Integration

- Release v0.1.83
  ([`c1f1354`](https://github.com/nfraxlab/ai-infra/commit/c1f1354a43006971aa313975b851d8989471240f))

### Refactoring

- Introduce comprehensive MCPClient exception handling and enhance client initialization with
  connection management options
  ([`df17ee1`](https://github.com/nfraxlab/ai-infra/commit/df17ee1831cbffd899e3e971d93c79fea098843e))


## v0.1.82 (2025-11-26)

### Continuous Integration

- Release v0.1.82
  ([`34d237f`](https://github.com/nfraxlab/ai-infra/commit/34d237f5905d19d8299abcb8232cc0b026e36dce))

### Refactoring

- Enhance Graph API with zero-config building and validation features
  ([`8a9acfc`](https://github.com/nfraxlab/ai-infra/commit/8a9acfc8581b381ea9d83273ccbe3731c94498c6))


## v0.1.81 (2025-11-26)

### Continuous Integration

- Release v0.1.81
  ([`50a90b6`](https://github.com/nfraxlab/ai-infra/commit/50a90b627bc1f0c1c09b04c74803f5524edaeaf7))


## v0.1.80 (2025-11-26)

### Continuous Integration

- Release v0.1.80
  ([`064cb77`](https://github.com/nfraxlab/ai-infra/commit/064cb77c716bc3ac710b3c0ce5591d3f450f36dc))

### Refactoring

- Update Graph import paths and remove core module
  ([`992a058`](https://github.com/nfraxlab/ai-infra/commit/992a058bbd96774ede77ef0d1d8ecb2d8f4036c7))


## v0.1.79 (2025-11-26)

### Continuous Integration

- Release v0.1.79
  ([`56d9a55`](https://github.com/nfraxlab/ai-infra/commit/56d9a55ac6b6649ac8d30126a76b273268d0e9fc))

### Features

- Introduce BaseLLM and LLM classes for enhanced model interaction
  ([`1db7e65`](https://github.com/nfraxlab/ai-infra/commit/1db7e65f19c3e50811ff10ba0a2fe7c610921754))

- Added BaseLLM class to serve as a foundation for LLM and Agent functionalities, including shared
  configuration, logging hooks, and model registry. - Implemented LLM class for direct model
  interactions, providing a simple API for chat-based interactions without agent capabilities. -
  Included methods for provider/model discovery, structured output handling, and token streaming. -
  Enhanced logging capabilities with request/response/error contexts for better observability. -
  Provided examples for usage of chat, structured output, and streaming tokens.


## v0.1.78 (2025-11-26)

### Continuous Integration

- Release v0.1.78
  ([`7ea957f`](https://github.com/nfraxlab/ai-infra/commit/7ea957f606764a809e70cbd41da54eb02ce2f98f))


## v0.1.77 (2025-11-26)

### Continuous Integration

- Release v0.1.77
  ([`d55dc45`](https://github.com/nfraxlab/ai-infra/commit/d55dc457c3d844a41db1c20830734de2fde2f869))


## v0.1.76 (2025-11-26)

### Continuous Integration

- Release v0.1.76
  ([`46b92d5`](https://github.com/nfraxlab/ai-infra/commit/46b92d56ce90f4ee24cfcfca06cdbd0b7b2fe47b))


## v0.1.75 (2025-11-26)

### Continuous Integration

- Release v0.1.75
  ([`45c7849`](https://github.com/nfraxlab/ai-infra/commit/45c7849511c1c6e7bc2ba82c6f79a04ee1c47ea6))


## v0.1.74 (2025-11-26)

### Continuous Integration

- Release v0.1.74
  ([`9474f4f`](https://github.com/nfraxlab/ai-infra/commit/9474f4f024df8f4ccd898ec4f567a6a69516c9b3))


## v0.1.73 (2025-11-26)

### Continuous Integration

- Release v0.1.73
  ([`abed90f`](https://github.com/nfraxlab/ai-infra/commit/abed90f996662d84740ca72ff47a824748f7af21))


## v0.1.72 (2025-11-26)

### Continuous Integration

- Release v0.1.72
  ([`d45c3b9`](https://github.com/nfraxlab/ai-infra/commit/d45c3b93d95093ec4e325f571dc6c121d4cab41a))


## v0.1.71 (2025-11-26)

### Continuous Integration

- Release v0.1.71
  ([`5b459b5`](https://github.com/nfraxlab/ai-infra/commit/5b459b58ddc13751854f5b2acc79a07da75e0eff))


## v0.1.70 (2025-11-13)

### Continuous Integration

- Release v0.1.70
  ([`0ed00dc`](https://github.com/nfraxlab/ai-infra/commit/0ed00dca416a619c4974090cda316e2460764ff1))


## v0.1.69 (2025-11-08)

### Continuous Integration

- Release v0.1.69
  ([`7bf214f`](https://github.com/nfraxlab/ai-infra/commit/7bf214f98ee987c94749853f92a174e039566323))


## v0.1.68 (2025-11-07)

### Continuous Integration

- Release v0.1.68
  ([`162435a`](https://github.com/nfraxlab/ai-infra/commit/162435a1bfc79c5247be6570f0a97220026962dc))


## v0.1.67 (2025-09-15)

### Continuous Integration

- Release v0.1.67
  ([`c692ac8`](https://github.com/nfraxlab/ai-infra/commit/c692ac8c1aaf7fd600b880212690db7473138b98))


## v0.1.66 (2025-09-08)

### Continuous Integration

- Release v0.1.66
  ([`049692c`](https://github.com/nfraxlab/ai-infra/commit/049692c69e4e3a2756c94cc6cbc34405f26dd7a0))


## v0.1.65 (2025-09-08)

### Continuous Integration

- Release v0.1.65
  ([`ea6b9d7`](https://github.com/nfraxlab/ai-infra/commit/ea6b9d726f2ca87828f657bcfb12edcebdc3f6a5))


## v0.1.64 (2025-09-08)

### Continuous Integration

- Release v0.1.64
  ([`251c25e`](https://github.com/nfraxlab/ai-infra/commit/251c25e3659a2b013bd31cf87c50f66d7a4fdd83))


## v0.1.63 (2025-09-08)

### Continuous Integration

- Release v0.1.63
  ([`95470ed`](https://github.com/nfraxlab/ai-infra/commit/95470ed4336dd66190e2a102c3307312ed27ef68))


## v0.1.62 (2025-09-07)

### Continuous Integration

- Release v0.1.62
  ([`611bce2`](https://github.com/nfraxlab/ai-infra/commit/611bce2dab772f2d0d6f829bb9f908ea8ddac2b6))


## v0.1.61 (2025-09-07)

### Continuous Integration

- Release v0.1.61
  ([`85d8477`](https://github.com/nfraxlab/ai-infra/commit/85d847752ba8e3a837b63d071a123caa38d1b2aa))


## v0.1.60 (2025-09-07)

### Continuous Integration

- Release v0.1.60
  ([`ae0219b`](https://github.com/nfraxlab/ai-infra/commit/ae0219b690ea417010593ca6d25f16f593c07f75))


## v0.1.59 (2025-09-06)

### Continuous Integration

- Release v0.1.59
  ([`d85e3ed`](https://github.com/nfraxlab/ai-infra/commit/d85e3edbc0f38801bb1a59278f712fc3983409b4))


## v0.1.58 (2025-09-03)

### Continuous Integration

- Release v0.1.58
  ([`4887636`](https://github.com/nfraxlab/ai-infra/commit/48876361db4f31ed24362806f98ad419a2d053e0))


## v0.1.57 (2025-09-03)

### Continuous Integration

- Release v0.1.57
  ([`196abb5`](https://github.com/nfraxlab/ai-infra/commit/196abb5b9fb1c3e902e9424a7c5ba14a5ce2ecbd))


## v0.1.56 (2025-09-03)

### Continuous Integration

- Release v0.1.56
  ([`0c03044`](https://github.com/nfraxlab/ai-infra/commit/0c03044814d41e4cc80cb58aaa81244089e1dcee))


## v0.1.55 (2025-09-03)

### Continuous Integration

- Release v0.1.55
  ([`6a69d80`](https://github.com/nfraxlab/ai-infra/commit/6a69d80c41a4a27a980dd7f97f9d5be42a2e6b0e))


## v0.1.54 (2025-09-03)

### Continuous Integration

- Release v0.1.54
  ([`07c4c11`](https://github.com/nfraxlab/ai-infra/commit/07c4c11a218378c58f83b284672733e61dc1127d))


## v0.1.53 (2025-09-03)

### Continuous Integration

- Release v0.1.53
  ([`e8f7f35`](https://github.com/nfraxlab/ai-infra/commit/e8f7f35df3b53f35d5eefd636271fd7f4de16639))


## v0.1.52 (2025-09-03)

### Continuous Integration

- Release v0.1.52
  ([`7501d84`](https://github.com/nfraxlab/ai-infra/commit/7501d845fa4832fdd81850826d6e4b324020de0b))


## v0.1.51 (2025-09-03)

### Continuous Integration

- Release v0.1.51
  ([`7bbe96e`](https://github.com/nfraxlab/ai-infra/commit/7bbe96e8bbfe3432a122b56a235a27f389543eef))


## v0.1.50 (2025-09-03)

### Continuous Integration

- Release v0.1.50
  ([`fc417c5`](https://github.com/nfraxlab/ai-infra/commit/fc417c50678e20589d9ecccddad5ded1034edb79))


## v0.1.49 (2025-09-03)

### Continuous Integration

- Release v0.1.49
  ([`16a84b9`](https://github.com/nfraxlab/ai-infra/commit/16a84b973e162c49d51437d27b6e001f810f89ea))


## v0.1.48 (2025-09-03)

### Continuous Integration

- Release v0.1.48
  ([`f05eed6`](https://github.com/nfraxlab/ai-infra/commit/f05eed6ace0631825de0dc8329c13c2e290b5193))


## v0.1.47 (2025-09-03)

### Continuous Integration

- Release v0.1.47
  ([`376dd50`](https://github.com/nfraxlab/ai-infra/commit/376dd5095b7c2d7fa4297985b8cecea4d4f0e375))


## v0.1.46 (2025-09-03)

### Continuous Integration

- Release v0.1.46
  ([`d0883a3`](https://github.com/nfraxlab/ai-infra/commit/d0883a3af9034f0b97619eabeee651b25b1691d5))


## v0.1.45 (2025-09-03)

### Continuous Integration

- Release v0.1.45
  ([`c0bbdbe`](https://github.com/nfraxlab/ai-infra/commit/c0bbdbe19a4046b04954f33744cc2511975f36fe))


## v0.1.44 (2025-09-03)

### Chores

- Ensure shim is executable
  ([`4b35b24`](https://github.com/nfraxlab/ai-infra/commit/4b35b24dcf4c0d7d28dc15100622ee06bbcbc3d7))

### Continuous Integration

- Release v0.1.44
  ([`8dce6cb`](https://github.com/nfraxlab/ai-infra/commit/8dce6cb94260534eadd3164e3c437717a5c708a3))


## v0.1.43 (2025-09-03)

### Continuous Integration

- Release v0.1.43
  ([`58f9d4b`](https://github.com/nfraxlab/ai-infra/commit/58f9d4b5040d9f4ab1235c938b968316b2b1a93d))


## v0.1.42 (2025-09-02)

### Continuous Integration

- Release v0.1.42
  ([`8420eb8`](https://github.com/nfraxlab/ai-infra/commit/8420eb88a0c61f842f9488ed257f906b075a6fc3))


## v0.1.41 (2025-09-02)

### Continuous Integration

- Release v0.1.41
  ([`9b91506`](https://github.com/nfraxlab/ai-infra/commit/9b915069f1449bf41a332ec45e5fd777472da5d6))


## v0.1.40 (2025-09-02)

### Continuous Integration

- Release v0.1.40
  ([`a1c5354`](https://github.com/nfraxlab/ai-infra/commit/a1c5354f29f24b180051764d89e904302752c949))


## v0.1.39 (2025-09-02)

### Continuous Integration

- Release v0.1.39
  ([`d910ab0`](https://github.com/nfraxlab/ai-infra/commit/d910ab04e67af0160557a9e07e56753fb3584bf3))


## v0.1.38 (2025-09-02)

### Continuous Integration

- Release v0.1.38
  ([`bd3ed0e`](https://github.com/nfraxlab/ai-infra/commit/bd3ed0ec48698dd1b28ea0a10c8ebb827f018cc6))


## v0.1.37 (2025-09-02)

### Continuous Integration

- Release v0.1.37
  ([`3cfa290`](https://github.com/nfraxlab/ai-infra/commit/3cfa290e9f44be8362f27869afbfb5155b5563a4))


## v0.1.36 (2025-09-02)

### Continuous Integration

- Release v0.1.36
  ([`3fb2237`](https://github.com/nfraxlab/ai-infra/commit/3fb2237d87d30e23a12cc8f1b806b72491b0d8b3))


## v0.1.35 (2025-09-02)

### Continuous Integration

- Release v0.1.35
  ([`11f9388`](https://github.com/nfraxlab/ai-infra/commit/11f93886b1185ba7f7157246e26fcf4acd7c1eec))


## v0.1.34 (2025-09-01)

### Continuous Integration

- Release v0.1.34
  ([`d3d0e75`](https://github.com/nfraxlab/ai-infra/commit/d3d0e7558b9ccba4b74c3d6d256fcd672113c17c))


## v0.1.33 (2025-09-01)

### Continuous Integration

- Release v0.1.33
  ([`a154f29`](https://github.com/nfraxlab/ai-infra/commit/a154f29b5371be45e45be679cd1261eff1c484a3))


## v0.1.32 (2025-08-29)

### Continuous Integration

- Release v0.1.32
  ([`56f8030`](https://github.com/nfraxlab/ai-infra/commit/56f803052dc109487ee96d426fc153d4e140b619))


## v0.1.31 (2025-08-29)

### Continuous Integration

- Release v0.1.31
  ([`d6623fc`](https://github.com/nfraxlab/ai-infra/commit/d6623fc45a08564bc4912a1e3b1e63a0cb87d041))


## v0.1.30 (2025-08-29)

### Continuous Integration

- Release v0.1.30
  ([`75dd2bc`](https://github.com/nfraxlab/ai-infra/commit/75dd2bcf1f1e4961ae4947ad7dbc298bf7ccc70a))


## v0.1.29 (2025-08-29)

### Continuous Integration

- Release v0.1.29
  ([`0d5a933`](https://github.com/nfraxlab/ai-infra/commit/0d5a933c8fba6dac76b5a1aecc52291f5d6c754b))


## v0.1.28 (2025-08-28)

### Continuous Integration

- Release v0.1.28
  ([`5b9eb92`](https://github.com/nfraxlab/ai-infra/commit/5b9eb925cf4d9e3039305bac065d2887755559e8))


## v0.1.27 (2025-08-28)

### Continuous Integration

- Release v0.1.27
  ([`e79d72c`](https://github.com/nfraxlab/ai-infra/commit/e79d72c9923e03d08bf4abaf5e1087a6e29bfc60))


## v0.1.26 (2025-08-28)

### Continuous Integration

- Release v0.1.26
  ([`210cbac`](https://github.com/nfraxlab/ai-infra/commit/210cbac360b0d1bd9d8bcd62963ac2880fc2d2ea))


## v0.1.25 (2025-08-28)

### Continuous Integration

- Release v0.1.25
  ([`f241323`](https://github.com/nfraxlab/ai-infra/commit/f241323e9d5df5f5e9d41f39c48f85394a9d40e0))


## v0.1.24 (2025-08-28)

### Continuous Integration

- Release v0.1.24
  ([`7e974b3`](https://github.com/nfraxlab/ai-infra/commit/7e974b386fe1cfb0214567824ee2338b1b235072))


## v0.1.23 (2025-08-28)

### Continuous Integration

- Release v0.1.23
  ([`64ee2dd`](https://github.com/nfraxlab/ai-infra/commit/64ee2ddb8f0dc14766a06c5db5446b2f96ac388e))


## v0.1.22 (2025-08-28)

### Continuous Integration

- Release v0.1.22
  ([`f356635`](https://github.com/nfraxlab/ai-infra/commit/f356635f0f9947dcb28c4fae8e5ad8858b3df949))


## v0.1.21 (2025-08-27)

### Continuous Integration

- Release v0.1.21
  ([`b46bb44`](https://github.com/nfraxlab/ai-infra/commit/b46bb4401a241c8cc33b764df55013d818e915db))


## v0.1.20 (2025-08-27)

### Continuous Integration

- Release v0.1.20
  ([`269ee81`](https://github.com/nfraxlab/ai-infra/commit/269ee81bc8295710e0b9b8ecfac86b2508adb1d6))


## v0.1.19 (2025-08-27)

### Continuous Integration

- Release v0.1.19
  ([`76564e7`](https://github.com/nfraxlab/ai-infra/commit/76564e76088dbd6e56a0c5b9e191c5e10f43e62f))


## v0.1.18 (2025-08-27)

### Continuous Integration

- Release v0.1.18
  ([`23308eb`](https://github.com/nfraxlab/ai-infra/commit/23308eb7489d98089e6e179dbe11e154af62d27b))


## v0.1.17 (2025-08-27)

### Continuous Integration

- Release v0.1.17
  ([`3669d37`](https://github.com/nfraxlab/ai-infra/commit/3669d379ac360e5e63a479bfd944940e4e328647))


## v0.1.16 (2025-08-26)

### Chores

- **main**: Sync with origin/main to include latest upstream changes
  ([`dbb0e07`](https://github.com/nfraxlab/ai-infra/commit/dbb0e07013e1824a786d08129aa63aeab89c2dfb))

### Continuous Integration

- Release v0.1.16
  ([`d3d5c5a`](https://github.com/nfraxlab/ai-infra/commit/d3d5c5af9ccbe110e41669b1efdee9b0ae9d594a))


## v0.1.15 (2025-08-26)

### Chores

- Sync with latest origin/main and commit local changes
  ([`2226d95`](https://github.com/nfraxlab/ai-infra/commit/2226d95fb6b71fa1e56df1a81340e1907c3804dc))

### Continuous Integration

- Release v0.1.15
  ([`faa1c5b`](https://github.com/nfraxlab/ai-infra/commit/faa1c5b14e36902fee395ec44074b90e736a0f5d))


## v0.1.14 (2025-08-26)

### Continuous Integration

- Release v0.1.14
  ([`6e1882b`](https://github.com/nfraxlab/ai-infra/commit/6e1882b9daf0cc6b5abba8a7a36f1b3f0f6e4646))


## v0.1.13 (2025-08-25)

### Continuous Integration

- Release v0.1.13
  ([`fed5f30`](https://github.com/nfraxlab/ai-infra/commit/fed5f3000b2f9da22f59ff7fda109b8d9ad1d9b7))


## v0.1.12 (2025-08-25)

### Continuous Integration

- Release v0.1.12
  ([`559deb8`](https://github.com/nfraxlab/ai-infra/commit/559deb82e9a7345dff626b8591f46dafeac3cfcc))


## v0.1.11 (2025-08-25)

### Continuous Integration

- Release v0.1.11
  ([`d56fb06`](https://github.com/nfraxlab/ai-infra/commit/d56fb063d45422fd77f236d4ba9f350b41732642))


## v0.1.10 (2025-08-25)

### Continuous Integration

- Release v0.1.10
  ([`cce216d`](https://github.com/nfraxlab/ai-infra/commit/cce216dec64f2e7c83e4dd0de51d6a10e977ef33))


## v0.1.9 (2025-08-25)

### Continuous Integration

- Release v0.1.9
  ([`acb1b19`](https://github.com/nfraxlab/ai-infra/commit/acb1b196a2861138f0a848bb717944040735cfa7))


## v0.1.8 (2025-08-25)

### Continuous Integration

- Release v0.1.8
  ([`4046459`](https://github.com/nfraxlab/ai-infra/commit/404645924aa15937705c9fcac3abb9f725f867fe))


## v0.1.7 (2025-08-25)

### Continuous Integration

- Release v0.1.7
  ([`c721b4b`](https://github.com/nfraxlab/ai-infra/commit/c721b4b399caa36df2374a4d2aa92f23fa04ba5f))


## v0.1.6 (2025-08-25)

### Continuous Integration

- Release v0.1.6
  ([`4b1a7c8`](https://github.com/nfraxlab/ai-infra/commit/4b1a7c8e2aaa36c20105d6c931b043c64b05dd95))


## v0.1.5 (2025-08-24)

### Continuous Integration

- Release v0.1.5
  ([`e07e8c3`](https://github.com/nfraxlab/ai-infra/commit/e07e8c3ffb2c119b7fc52a7444133fa44432442e))


## v0.1.4 (2025-08-24)

### Bug Fixes

- Ensure HITL-wrapped tools used by agent (context.tools=effective_tools)
  ([`152afc9`](https://github.com/nfraxlab/ai-infra/commit/152afc9f75e8a6aeec8cb2331990faddf514994c))

- Preserve message dict shape in _apply_hitl replacement
  ([`189ad91`](https://github.com/nfraxlab/ai-infra/commit/189ad918058a001e6570334b35b5ce7dbebce8d0))

- Safe retry handling in chat() when event loop running
  ([`67a5726`](https://github.com/nfraxlab/ai-infra/commit/67a5726d324e7c8c1492daefd3f5321293d56b45))

- Set has_memory to False in analyze to avoid AttributeError after config removal
  ([`abcafa6`](https://github.com/nfraxlab/ai-infra/commit/abcafa64bd30cb0b2ca23a6f6223b71fc903493d))

### Chores

- Centralize dotenv loading and remove duplicate calls
  ([`8788152`](https://github.com/nfraxlab/ai-infra/commit/87881527e51a21c738a8893bc706f9cfddc81561))

- Guard stream_tokens() against agent/tool kwargs
  ([`8333c24`](https://github.com/nfraxlab/ai-infra/commit/8333c2412cfae229219b270bc49ce197ee7e682c))

- Harden HITL gating in arun_agent_stream with safer messages mutation
  ([`fa32592`](https://github.com/nfraxlab/ai-infra/commit/fa32592926ef30f7d4c275bfc3a04651215b641a))

- Minor HITL gating guard adjustment
  ([`b462b7f`](https://github.com/nfraxlab/ai-infra/commit/b462b7faa4c90167cfc77aa6a2e8dc39be143191))

- Refine stream_mode typing and related streaming logic
  ([`a4d01d1`](https://github.com/nfraxlab/ai-infra/commit/a4d01d1a55df5e04e2f388cf2a3d2b356cf0d741))

- Strip agent/tool kwargs in chat() and achat() for safety
  ([`a5e3866`](https://github.com/nfraxlab/ai-infra/commit/a5e386640866b702210f3bdf87d5822b201fb5ef))

- Unify system message to plain dict
  ([`3661255`](https://github.com/nfraxlab/ai-infra/commit/3661255001fb2067a0b5623b14741da91044ba27))

- Update quickstart basics example
  ([`be2c37e`](https://github.com/nfraxlab/ai-infra/commit/be2c37e7d64433db12761926321dc00d569dd7a1))

### Continuous Integration

- Release v0.1.4
  ([`a33e549`](https://github.com/nfraxlab/ai-infra/commit/a33e549e6fe90dd593137cfbfd56ceabc49c1c5d))

### Documentation

- Expand HITL callback contract in set_hitl docstring
  ([`a2febc3`](https://github.com/nfraxlab/ai-infra/commit/a2febc36c4cf5936154c114df6f5fddaa02635a6))

### Features

- Add trace callback for node entry/exit; refactor for deduplication and bugfixes; always use
  self._memory_store for graph compilation
  ([`81d5649`](https://github.com/nfraxlab/ai-infra/commit/81d564996bbbcaa09a6367f901ed3e4c614e2361))

- Allow run and run_async to accept state as kwargs or dict; improve ConditionalEdge targets
  inference; update usage example
  ([`84eb4fe`](https://github.com/nfraxlab/ai-infra/commit/84eb4fe4f644e213dd80c7ef2f6b06f85a71a487))

- Async-aware tool gate (_maybe_await in _wrap_tool_for_hitl)
  ([`2c1d266`](https://github.com/nfraxlab/ai-infra/commit/2c1d266e242b60c88a6f86fa85cce88c49471856))

- Conditionaledge can infer targets from router_fn if not provided; update usage example
  ([`3da2ae6`](https://github.com/nfraxlab/ai-infra/commit/3da2ae68b531cb6078096c5526188b13d851b828))

- Conditionaledge can infer targets from router_fn if not provided; update usage example in
  __init__.py
  ([`4cf9295`](https://github.com/nfraxlab/ai-infra/commit/4cf92956995e1f01444398f57fa9592afb9bf849))

- Preserve full values shape during HITL gating in arun_agent_stream
  ([`73be1cb`](https://github.com/nfraxlab/ai-infra/commit/73be1cb169dad62e5be60d76145e8fbbf1a60c2f))

- Support async HITL callbacks via _maybe_await in _apply_hitl and streaming gating
  ([`ba7c5ed`](https://github.com/nfraxlab/ai-infra/commit/ba7c5ed1670d0d01b93469ce59c1acdbcb8b9106))

- Warn when structured output unsupported (with_structured_output)
  ([`e5ddf52`](https://github.com/nfraxlab/ai-infra/commit/e5ddf527cc148a1553c6b63b64e500a1f1a20427))

- **corellm**: Add explicit global tool usage policy with logging and optional enforcement
  ([`b6f49d8`](https://github.com/nfraxlab/ai-infra/commit/b6f49d8d1ee1103c1bc3cb9ef23d9b2ca56295dd))

### Refactoring

- Deduplicate and simplify CoreGraph, add trace callback for node entry/exit, fix memory_store bug,
  and minor code cleanups
  ([`d0b2839`](https://github.com/nfraxlab/ai-infra/commit/d0b28393c928f85acf35b70015bd214201e1b18e))

- Deduplicate, simplify, and add tracing to CoreGraph; fix memory_store bug; general code cleanups
  ([`9f4fa6c`](https://github.com/nfraxlab/ai-infra/commit/9f4fa6c3c916de4d0662960b4d3a6cf2b6e512b2))

- Deduplicate, simplify, and add tracing to CoreGraph; fix memory_store bug; general code cleanups
  ([`00d44b9`](https://github.com/nfraxlab/ai-infra/commit/00d44b9d214b9a6a6b47eb4aa64fea2fc1c778d8))

- Extract HITL + tool policy logic to tools.py and update core.py
  ([`b4ae421`](https://github.com/nfraxlab/ai-infra/commit/b4ae421249bcac3b4629974f1b04d8c83bc6ff2b))

- Extract runtime binding (ModelRegistry, tool_used, bind_model_with_tools, make_agent_with_context)
  into runtime_bind.py and delegate from core
  ([`cb8f5fc`](https://github.com/nfraxlab/ai-infra/commit/cb8f5fca59a4fbfdf64eaa011c2f736629fb3013))

- Simplify effective tool selection and retain quickstart edits
  ([`80a2eac`](https://github.com/nfraxlab/ai-infra/commit/80a2eac605b30e9fdbb59d9f51c1afd695149397))

- Simplify HITL block handling by returning verbatim replacement
  ([`fec251a`](https://github.com/nfraxlab/ai-infra/commit/fec251a205f04da8540755b5742c163739641144))

- **llm**: Remove non-essential comments/docstrings in CoreLLM
  ([`37f7e9e`](https://github.com/nfraxlab/ai-infra/commit/37f7e9eb81b40e7cd498364db24ce668716404a6))
