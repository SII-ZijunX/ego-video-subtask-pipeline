# Contributing

1. Create a focused branch and keep data, videos, credentials, and model weights out of Git.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Run `pytest`, `python -m compileall video_annotation_pipeline`, and the leakage checks in the README.
4. Add tests for boundary reconciliation, duration gates, or output-schema changes.
5. Describe behavior and data-contract changes in the pull request.

Do not weaken the 3-second final-clip gate or allow review-only clips into the
training manifest without an explicit design discussion and new tests.
