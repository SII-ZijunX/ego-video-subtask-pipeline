# Security and data handling

- Never commit API keys, private model paths, signed URLs, participant identifiers, or raw videos.
- Read API keys only from the environment variable named by `backend.api_key_env`.
- Treat review pages as sensitive when they reference non-public datasets.
- Run generated HTML only from a trusted local directory; do not publish it by default.
- Report security or privacy issues privately to the repository maintainers.

Ego datasets may contain faces, voices, homes, workplaces, and incidental
personal information. Users are responsible for dataset licenses, consent,
access controls, retention, and any required de-identification.
