# Security Policy

## Scope

jspace is a research tool for mechanistic interpretability experiments on language models.
It runs locally and does not include any network services, authentication, or user data handling.

The main security consideration is **model weights and API tokens**: this repo never stores,
logs, or transmits HuggingFace tokens or model weights.

## Reporting a Vulnerability

If you find a security issue (e.g. a dependency with a known CVE that affects this project,
or an unsafe code pattern), please email [kameshkanna@gmail.com] rather than opening a
public issue. Include:

- A description of the issue and its potential impact
- Steps to reproduce or a proof of concept if applicable

You will receive a response within 72 hours. If the issue is confirmed, a fix will be
released and the reporter credited in the changelog unless anonymity is requested.

## Dependency security

Dependencies are pinned in `pyproject.toml`. Run `pip install -e ".[dev]"` to install the
exact versions. Check for known vulnerabilities with:

```bash
pip install pip-audit
pip-audit
```
