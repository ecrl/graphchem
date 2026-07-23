# Security Policy

## Supported versions

Security fixes are prioritized for the latest published release on PyPI and the
current `master` tip. Older minor versions may not receive backports.

| Version | Supported |
| ------- | --------- |
| 2.3.x (latest tip / PyPI) | Yes |
| < 2.3 | No |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Prefer one of:

1. [GitHub Security Advisories](https://github.com/ecrl/graphchem/security/advisories/new)
   for this repository (private disclosure), or
2. Email the maintainer at travis.j.kessler@gmail.com with a description,
   impact, and reproduction steps if available.

You should receive an acknowledgment within a reasonable time. We will
coordinate a fix and disclosure timeline with you when appropriate.

## Publishing

PyPI releases use **trusted publishing (OIDC)** via the
`Publish GraphChem to PyPI` GitHub Actions workflow on published GitHub
Releases. No long-lived PyPI API tokens are stored in this repository.
Maintainers must configure a GitHub Environment named `pypi` and a matching
PyPI trusted publisher for project `graphchem`.
