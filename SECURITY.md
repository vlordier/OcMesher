# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue to disclose a security
vulnerability.

Instead, report it privately through one of the following channels:

1. **GitHub private vulnerability reporting** — use the
   [Security tab](https://github.com/princeton-vl/OcMesher/security/advisories/new)
   on this repository.
2. **Email** — send details to the Princeton VL lab maintainers.  Contact
   information can be found in the Princeton VL lab website or via the authors
   listed in `pyproject.toml`.

### What to Include

- A concise description of the vulnerability.
- Steps to reproduce or a minimal proof-of-concept.
- The potential impact (e.g. remote code execution, denial of service).
- Any suggested mitigation if you have one.

## Response Timeline

| Event                          | Target SLA |
| ------------------------------ | ---------- |
| Acknowledgement of report      | 3 business days |
| Confirmed / triaged            | 7 business days |
| Fix released or patch issued   | 30 days (critical), 90 days (others) |

## Scope

This policy covers the Python package (`ocmesher/`) and the C++ shared library
(`ocmesher/source/`).  Third-party dependencies are out of scope; please report
those directly to the upstream project.

## Disclosure Policy

Once a fix is available we will publish a GitHub Security Advisory and credit
the reporter (unless they prefer to remain anonymous).
