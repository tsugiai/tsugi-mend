# Security Policy

## Supported Versions

`tsugi-mend` is currently pre-alpha software. Security fixes are handled on the
latest public release line.

| Version | Supported |
| ------- | --------- |
| 0.1.x | Yes |

## Reporting a Vulnerability

Please report security issues privately to the public maintainer contact listed
for this package:

Tong Liu <tong@tsugicinema.com>

Do not open a public GitHub issue for a suspected vulnerability. Include the
affected version or commit, a short description of the impact, reproduction
steps when available, and any relevant logs or configuration snippets that do
not contain secrets.

The maintainer will acknowledge credible reports, investigate the issue, and
coordinate a fix or disclosure timeline appropriate to the severity.

## Sideband Trust Model

`tsugi-mend` includes a sideband control channel for rack-level progress and
health metadata. In the 0.1.x line, that channel is plain TCP with JSON payloads
and is intended for a trusted intra-cluster network. It has no transport
authentication by default.

Run the sideband listener only on a trusted private network or behind equivalent
network controls. Do not expose it directly to the public internet or to
untrusted tenants. Opt-in authentication and TLS support are planned for a
future release.
