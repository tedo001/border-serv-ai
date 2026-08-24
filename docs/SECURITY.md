# Security and privacy

This platform can identify people by face and vehicles by registration, and its
output can put someone in front of an armed response. The controls below are
not incidental.

## Authentication

- **People** authenticate with a username and password, receiving a short-lived
  JWT access token plus a refresh token. Passwords are SHA-256 pre-hashed then
  bcrypt-hashed (cost 12); the pre-hash exists because bcrypt silently
  truncates at 72 bytes and would otherwise accept two different passphrases
  sharing a prefix.
- **Machines** authenticate with an API key, stored only as a SHA-256 digest and
  shown exactly once at creation. A C2 gateway never needs an operator's password.
- Login failures are **indistinguishable** regardless of cause, so operator
  accounts cannot be enumerated.
- Lockout is keyed on username *and* source address together: keying on address
  alone locks out a whole post behind one NAT; keying on username alone lets an
  attacker lock a real operator out of their console.
- Signing keys shorter than 32 bytes are rejected at construction, not warned
  about, so a weak secret cannot reach production quietly.

## Authorisation

A strict hierarchy, not a permission matrix — border force structure is
hierarchical, and a matrix invites misconfiguration.

| Role | May |
|---|---|
| `viewer` | Read events, view live video |
| `operator` | + acknowledge alerts, manage the vehicle watchlist |
| `supervisor` | + configure cameras, zones and rules; enrol faces |
| `admin` | + manage users, API keys, system settings |

An unrecognised role resolves to `viewer` — a typo reduces access, never grants it.

Face enrolment requires `supervisor` deliberately: adding a person to a
biometric watchlist is a materially different act from acknowledging an alert.

## Tokens in URLs

A browser `<img>` cannot send an `Authorization` header, so media endpoints
accept `?token=`. This is scoped tightly: **read-only GETs on media paths only**
(`/snapshot`, `/clip`, `/stream.mjpeg`, `/stream`). Every other endpoint
rejects a query token, so a URL leaked into a proxy log or `Referer` cannot be
replayed into a state-changing call.

## Audit trail

Every security-relevant action is recorded with actor, action, target, source
address and outcome: logins, user and key management, camera reconfiguration,
and **every watchlist change**. In a system that can identify people by face,
"who added this person, when, and on what authority" must be answerable months
later — which is why `reference` (a case file or intelligence report) is carried
on every watchlist entry.

## Evidence integrity

SHA-256 per artefact, plus a per-day hash chain across manifests, so both
alteration and deletion are detectable. Verify from the CLI or the console:

```bash
ibvap evidence verify-chain 2026-08-24
```

This is tamper-**evident**, not tamper-proof. Anyone with write access to the
directory could rebuild the chain; defeating that requires an append-only store
or external notarisation, which is a deployment decision.

## Privacy controls

| Setting | Effect |
|---|---|
| `privacy.blur_unmatched_faces` | Blur faces in stored evidence unless they matched a watchlist |
| `privacy.store_only_watchlist_faces` | Never persist embeddings of passers-by |
| `privacy.face_crop_retention_days` | Purge raw face crops on a shorter clock than general evidence |
| `privacy.audit_biometric_matches` | Record every watchlist match for later review |

Face embeddings **never leave the node as data**. The watchlist API returns
identities and match decisions, never vectors — so no API consumer can
accumulate its own gallery.

Recognition is also gated on quality: a face too small, too blurred or too far
off-frontal is rejected outright rather than matched at low confidence, and a
probe must beat the runner-up by a margin. A face that is 0.61 against two
different people is not a match — it is an ambiguous face.

## Network posture

- A node binds `0.0.0.0` on an **isolated operational network**. Do not expose
  it to the internet; put a reverse proxy with TLS in front if remote access is
  required.
- RTSP credentials are **redacted** from every log line — those logs get shipped
  to a central collector.
- The bootstrap password is printed to the console, never logged, for the same
  reason.
- Outbound webhooks are HMAC-SHA256 signed with the timestamp inside the signed
  material, so a captured delivery cannot be replayed.
- The container runs unprivileged with a read-only root filesystem; the systemd
  unit applies equivalent hardening, since the process decodes untrusted video
  from an operational network.

## Reporting

Report vulnerabilities privately to the repository maintainers rather than
opening a public issue.
