# O4FM Analog Link Protocol (v0)

This document defines the analog-link-side profile for O4FM over standard FM radio audio ports.  
It does not replace the digital frame protocol; it defines audio capture/playback behavior, processing chains, and operating modes.

## Status

- `implemented`: CPAL real-time audio I/O, WAV offline I/O, optional voice pre/post processing.
- `planned`: unified in-band analog-link control signaling (for example pilot/training and auto-threshold negotiation).

## Layering

- RF layer: commercial/amateur FM voice radios (external mic/speaker audio ports).
- Analog audio layer: PCM capture/playback (CPAL or WAV).
- Digital transport layer: O4FM logical frame + PHY/FEC (see digital protocol document).

Note: the analog link layer does not define encryption/authentication and is expected to follow amateur radio regulations.

## Audio Interface Profile (v0)

- PCM format: `i16`, mono preferred.
- Real-time default sample rate: `48_000 Hz` (configurable).
- Offline mode: arbitrary WAV input is allowed and resampled as needed.
- Channel constraints: digital pipeline currently requires `1` channel; passthrough can use multiple channels.

## Operating Modes

### 1) Passthrough Mode

- Goal: minimal processing path for link validation and device bring-up.
- Behavior: input audio is forwarded directly to output with only required buffering.
- Typical use: gain/echo/latency/clipping diagnostics.

### 2) Digital Relay Mode

- Goal: carry digital baseband through analog audio ports.
- Behavior: `demod -> FEC decode -> link -> FEC encode -> mod`.
- Constraints: current implementation requires mono, and peer parameters must match (modulation/symbol rate/FEC).

### 3) Voice Payload Mode

- Goal: voice content encoded into payload and transported by the digital layer.
- Codec: currently Opus (default 8 kHz mono, 20 ms frame, ~7 kbps target).
- Processing chain: optional `basic` (high-pass + slow AGC + light limiter) or `passthrough` (no processing).

## Silence/Gap and Sync

- Short silence gaps are allowed between frames (`inter-frame-silence-ms`).
- Receiver segmentation/synchronization is stream-oriented and does not depend on fixed mute lengths.
- `zero-threshold` controls near-zero segmentation gating; tune it against real channel noise.

## Handshake-Derived Session Parameters

- The low-rate control exchange is used to commit a runtime profile before data transfer.
- Committed parameters include modulation, symbol rate, FEC parameters, and voice bitrate target.
- Receiver behavior should switch according to the committed profile rather than static local defaults.

## One-Way Operation Over Continuous Noise

- A return channel is not required.
- Each transmitted burst carries a known training-tone prefix before frame payload.
- The receiver estimates `tone_center_hz` and `tone_spacing_hz` from this prefix and
  demodulates the following frame segment with the estimated plan.
- Frame acceptance still requires CRC success; noise-only segments should not pass.

## Device Selection

- The implementation provides device enumeration and stable IDs for input/output selection.
- Recommended flow:
  1. List devices and confirm expected audio path.
  2. Pin input/output device IDs.
  3. Run TX/RX with the same profile on both sides.

## Interop Guidance (Recommended)

- Start with `passthrough` to validate the end-to-end audio path, then switch to `digital`.
- Start from conservative parameters (lower threshold, default frame gap), then tighten.
- For voice, validate end-to-end latency/jitter first, then tune bitrate and preprocessing.

## Non-goals (v0)

- No in-band auto-negotiation format is defined at analog layer.
- No analog-layer FEC is defined (FEC is in the digital transport layer).
- No full-duplex voice or echo-cancellation protocol is defined.

## Related

- Digital protocol: `/docs/digital-protocol.md`
