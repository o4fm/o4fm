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
cargo run -p o4fm-node -- --mode=bin2wav --in-bin=input.bin --out-wav=output.wav --inter-frame-silence-ms=2

# Recover binary data from protocol WAV signal
cargo run -p o4fm-node -- --mode=wav2bin --in-wav=input.wav --out-bin=output.bin --zero-threshold=2
```

## Benchmarks (Criterion HTML)

```bash
cargo bench -p o4fm-link --bench critical_metrics
```

Report entry page:

`/Users/deltonding/Projects/github.com/dsh0416/o4fm/target/criterion/report/index.html`

## Note

The FEC internals are intentionally lightweight for v1 scaffolding. The public API is designed to allow swapping in a full QC-LDPC implementation in a follow-up iteration.
