# GitHub release checklist

## Before creating the repository

- Confirm the repository name and owning GitHub organization.
- Confirm that retaining the upstream Microsoft MIT copyright is acceptable.
- Decide the maintainer contact used for security reports.
- Confirm the README clone URL and `pyproject.toml` project URLs target the intended repository.
- Run the full tests and leakage checks from the README.
- Confirm no raw/derived videos, participant data, model weights, credentials, or internal paths are tracked.

## Suggested repository settings

- Default branch: `main`.
- Require pull requests and the `ci / test` status check.
- Require at least one project-team review for pipeline or schema changes.
- Enable secret scanning and dependency alerts.
- Disable GitHub Pages unless review HTML contains only public/synthetic data.
- Add CODEOWNERS for `video_annotation_pipeline/`, `configs/`, and `docs/data-contract.md`.

## First release

1. Tag the current schema and CLI as `v0.1.0`.
2. Record the tested Python, ffmpeg, VLM, and GPU environment in release notes.
3. Link only synthetic examples; keep real Ego4D outputs in controlled storage.
4. State that model-generated labels require quality review before training or publication.
5. Publish a small aggregate validation report without source media or participant identifiers.
