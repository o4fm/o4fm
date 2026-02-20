# O4FM Digital Protocol (v0)

This document defines the digital payload protocol carried inside O4FM PHY frames.

Related analog-link spec:
- `/docs/analog-protocol.md`

## Layering

- PHY frame payload is fixed at `128` bytes.
- Logical frame max size is `1792` bytes.
- One logical frame is split across multiple PHY frames for transmission.

## Logical Frame Header

All integer fields are big-endian.

| Field | Size | Description |
|---|---:|---|
| `magic` | 4B | ASCII `O4FM` |
| `version` | 1B | Protocol version, current `1` |
| `header_len` | 1B | Header length in bytes, current `40` |
| `total_size` | 2B | Full logical frame length (header + payload), `<= 1792` |
| `callsign` | 16B | ASCII callsign, padded/truncated to 16 bytes |
| `flags` | 8B | Bit flags for control features |
| `mode` | 8B | Mode ID (see below) |

Header total: `40` bytes.  
Max logical payload: `1792 - 40 = 1752` bytes.

## Mode IDs

- `1`: Voice mode (Opus payload)
- `2`: Text mode (UTF-8 payload)
- `3`: IP mode (reserved for IPv4/IPv6 adaptation)

### Voice Mode Payload (v0)

- Payload is a stream of `u16_be packet_len` + `Opus packet bytes`.
- Opus settings are currently configured out-of-band (CLI setting on both sides).
- Current default profile is mono 8 kHz, 20 ms frames, 7 kbps target bitrate.
- Optional voice DSP profile is out-of-band: `basic` (default) or `passthrough` (no enhancement).

## Transmission Rules

- Sender constructs logical frame first.
- Sender fragments encoded logical frame into 128-byte PHY payload chunks.
- Receiver reassembles byte stream from decoded PHY data frames.
- Receiver scans for `magic`, validates length/version, and decodes logical frame.
- Receiver outputs logical payload bytes to upper layer.

## Low-Rate Negotiation (Probe/Capability/Select/Commit)

- Link starts in a low-rate control phase and exchanges supported profiles.
- A selected profile is committed before entering `Data` state.
- Session runtime parameters are derived from the committed negotiation profile.

Negotiation profile fields (encoded in capability/select/commit payloads):

| Field | Size | Description |
|---|---:|---|
| `profile_id` | 1B | Local profile index |
| `modulation` | 1B | FSK order enum |
| `symbol_rate` | 1B | Symbol-rate enum |
| `fec_scheme` | 1B | FEC scheme enum |
| `code_n` | 2B | FEC codeword length |
| `code_k` | 2B | FEC payload length |
| `interleaver_depth` | 1B | Interleaver depth |
| `max_iterations` | 1B | Decoder max iterations |
| `voice_bitrate_bps` | 4B | Voice target bitrate in bps |

Current implementation exports the committed profile to the link layer via an
`EnterProfile { profile }` action so the receiver/transmitter can switch
demod/mod/FEC/voice settings from handshake result instead of manual alignment.

## Unidirectional Burst Mode (No Return Channel Required)

The protocol also supports a one-way mode for channels where no duplex/half-duplex
response path is available.

Per-burst flow:

- Training tones (known pattern)
- Sync (from frame + CRC validation path)
- Data frame payload

The receiver can decode passively by:

- estimating tone center/spacing from the training section,
- demodulating with the estimated tone plan,
- validating decoded frames with CRC.

In this mode, reliability is provided by FEC and sender-side repetition policy,
not by ACK-based retransmission.

## Notes

- This v0 header does not yet include logical-frame CRC or per-fragment index fields.
- Integrity currently relies on PHY frame CRC and logical header validation.
- Future versions can extend `flags`/`mode` behavior while keeping backward compatibility through `version`.
