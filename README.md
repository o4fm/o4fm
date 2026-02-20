# o4fm

Open-source audio-port digital protocol bootstrap for VHF/UHF FM radios.

## Current scope (v1 bootstrap)

- 4-FSK over 48 kHz PCM at fixed 4800 symbols
- Fixed data frame payload: 128 bytes
- Fixed logical MTU: 1792 bytes (14 frames)
- Low-rate handshake + data phase state machine
- LDPC-like pluggable FEC API (bootstrap implementation)
- Stop-and-wait ARQ
- Rust workspace with `no_std`-friendly core crate

## Digital payload protocol

- Spec: `/docs/digital-protocol.md`
- Logical frame header includes `magic + total_size + callsign(16B) + flags(8B) + mode(8B)`
- Logical frame max size: `1792B`
- PHY payload size: `128B` (logical frame fragmented across PHY frames)

## Workspace

- `/crates/o4fm-core`: profile types, frame codec, CRC, whitening
- `/crates/o4fm-phy`: BFSK mod/demod and soft-bit export
- `/crates/o4fm-fec`: FEC encode/decode + interleaver API
- `/crates/o4fm-link`: handshake + link state machine
- `/crates/o4fm-audio`: host audio abstraction and mock loopback
- `/crates/o4fm-radio-io`: PTT/COR abstraction and mock backend
- `/apps/o4fm-node`: reference link-state demo
- `/apps/o4fm-lab`: BER/FER-oriented lab harness

## Build & test

```bash
cargo test --workspace
cargo run -p o4fm-node -- --mode=demo
cargo run -p o4fm-lab
```

`o4fm-node` audio modes:

```bash
# List audio devices with stable IDs
cargo run -p o4fm-node -- --list-devices

# Demo negotiation + debug waveform dump
cargo run -p o4fm-node -- --mode=demo --demo-wave-out=target/o4fm-node-demo.wav

# Realtime audio passthrough (CPAL)
cargo run -p o4fm-node -- --mode=cpal --sample-rate=48000 --channels=1 --frame-samples=480 --seconds=10

# Realtime digital relay path (demod/FEC/link/remod)
cargo run -p o4fm-node -- --mode=cpal --pipeline=digital --sample-rate=48000 --channels=1 --frame-samples=480 --seconds=10

# Select specific devices by ID
cargo run -p o4fm-node -- --mode=cpal --pipeline=digital --input-device-id=\"<ID>\" --output-device-id=\"<ID>\"

# Offline WAV processing
cargo run -p o4fm-node -- --mode=wav --in=input.wav --out=output.wav --sample-rate=48000 --frame-samples=480 --gain=1.0

# Convert arbitrary binary file to protocol-framed WAV signal
cargo run -p o4fm-node -- --mode=bin2wav --in-bin=input.bin --out-wav=output.wav --inter-frame-silence-ms=2 --callsign=BH4GTN --lp-mode=text --lp-flags=0x0

# Recover binary data from protocol WAV signal
cargo run -p o4fm-node -- --mode=wav2bin --in-wav=input.wav --out-bin=output.bin --zero-threshold=2

# Voice TX: PCM WAV -> Opus(7kbps) -> O4FM WAV
cargo run -p o4fm-node -- --mode=voice-tx --in-wav=speech.wav --out-wav=rf.wav --opus-bitrate=7000 --opus-frame-ms=20 --voice-dsp=basic --callsign=BH4GTN --lp-flags=0x0

# Voice RX: O4FM WAV -> Opus -> PCM WAV (8 kHz mono)
cargo run -p o4fm-node -- --mode=voice-rx --in-wav=rf.wav --out-wav=speech_out.wav --opus-bitrate=7000 --voice-dsp=basic --zero-threshold=2

# Completely bypass voice enhancement (no AGC/HPF/post-processing)
cargo run -p o4fm-node -- --mode=voice-tx --in-wav=speech.wav --out-wav=rf.wav --voice-dsp=passthrough
```

## Benchmarks (Criterion HTML)

```bash
cargo bench -p o4fm-link --bench critical_metrics
```

Report entry page:

`/target/criterion/report/index.html`

## Note

The FEC internals are intentionally lightweight for v1 scaffolding. The public API is designed to allow swapping in a full QC-LDPC implementation in a follow-up iteration.

## CI/CD (GitHub Actions)

- CI workflow: `.github/workflows/ci.yml`
  - Linux unit tests on every `push` and `pull_request`
- Packaging workflow: `.github/workflows/packages.yml`
  - Trigger on `workflow_dispatch` and `v*` tags
  - Native packages:
    - Linux `x86_64`
    - macOS `x86_64` and `arm64`
    - Windows `x86_64`
  - Cross Linux packages:
    - `aarch64-unknown-linux-gnu`
    - `armv7-unknown-linux-gnueabihf`
