# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please **do not** open a public issue.

Instead, please report it via:
- GitHub Security Advisory: [Report a vulnerability](https://github.com/ZhouYinLong-lab/PromptLite-Seg/security/advisories/new)
- Or create a private discussion in the Security tab

I will respond within 48 hours with an assessment and expected resolution timeline.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| Latest  | :white_check_mark: |

## Security Best Practices

- Treat VOC archives, Parquet files, SAM checkpoints, and output directories as
  untrusted until their hashes or project ownership marker have been verified.
- Do not use `--replace` on a directory that was not created by the project
  materializer; unmanaged paths are intentionally rejected.
- Use only the fixed-hash official SAM checkpoint documented by the project.
- Enable Dependabot alerts for automated vulnerability notifications
- Review dependency updates regularly

## Known Dependency Advisory

The frozen CUDA reproduction environment uses PyTorch 2.11.0 and dependency
scanning reports a low-severity advisory affecting `torch.jit.script` on
untrusted input. PromptLite-Seg does not call TorchScript/JIT and does not load
unverified checkpoints. This is a scoped mitigation, not a claim that the
dependency is generally unaffected. Any deployment that accepts untrusted
models or enables JIT should upgrade to a fixed PyTorch build and rerun the
benchmark validation.
