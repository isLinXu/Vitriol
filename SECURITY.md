## Security Policy

Thank you for helping keep Vitriol and its users safe.

## Supported Versions

Security fixes are typically applied to the latest version on the default branch.

Older releases may not receive security updates.

## Reporting A Vulnerability

Please do **not** open a public issue for suspected vulnerabilities.

Instead, report privately to the maintainers and include:

- A clear description of the issue
- Affected components or commands
- Reproduction steps or a proof of concept
- Expected impact
- Any suggested mitigation, if known

If the issue involves remote-code execution, model loading, or untrusted artifacts, include the exact command or workflow that triggers it.

## What To Expect

Maintainers will try to:

- Acknowledge the report within a reasonable time
- Reproduce and assess the issue
- Decide whether the report is a vulnerability or a hardening request
- Coordinate a fix and responsible disclosure when appropriate

## Scope Notes

The following areas deserve special care in this repository:

- `trust_remote_code` model loading paths
- CLI and API surfaces that accept user-provided model IDs or file paths
- Visualization and HTML-serving commands
- Optional integrations and web UI components
- Any benchmark or patching path that executes model-specific code

## Safe Usage Reminder

For safer environments such as CI or shared systems:

- Prefer `--no-trust-remote-code` when compatibility allows
- Use trusted local model paths where possible
- Avoid running unreviewed third-party model code
