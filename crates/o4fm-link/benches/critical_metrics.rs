use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use heapless::Vec as HVec;
use o4fm_core::{
    DATA_FRAME_PAYLOAD_BYTES, FecProfile, Frame, FrameHeader, FrameKind, LinkProfile,
    MAX_PAYLOAD_BYTES, RadioProfile, SymbolRate,
};
use o4fm_fec::{decode_ldpc, encode_ldpc};
use o4fm_link::{LinkEvent, LinkMachine};
use o4fm_phy::{demodulate, modulate};
use std::time::Instant;

fn bench_fec(c: &mut Criterion) {
    let fec = FecProfile::default();
    let payload: Vec<u8> = (0..usize::from(fec.code_k))
        .map(|i| u8::from(((i * 7) % 13) < 6))
        .collect();

    let encoded = encode_ldpc(&payload, &fec);
    let llr: Vec<f32> = encoded
        .iter()
        .map(|&b| if b == 1 { 4.0 } else { -4.0 })
        .collect();

    let mut group = c.benchmark_group("fec_ldpc");
    group.throughput(Throughput::Elements(u64::from(fec.code_k)));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let _ = encode_ldpc(&payload, &fec);
        })
    });

    group.bench_function("decode", |b| {
        b.iter(|| {
            let _ = decode_ldpc(&llr, &fec).expect("decode must succeed");
        })
    });

    group.finish();
}

fn bench_phy(c: &mut Criterion) {
    let mut group = c.benchmark_group("phy_fsk");
    let mut radio = RadioProfile::default();
    radio.symbol_rate = o4fm_core::SymbolRate::R4800;
    let bits: Vec<u8> = (0..256).map(|i| u8::from((i % 5) < 2)).collect();
    let pcm = modulate(&bits, &radio).expect("modulate");
    group.throughput(Throughput::Elements(bits.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("modulate", 4800_u32),
        &bits,
        |b, in_bits| {
            b.iter(|| {
                let _ = modulate(in_bits, &radio).expect("modulate");
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("demodulate", 4800_u32),
        &pcm,
        |b, in_pcm| {
            b.iter(|| {
                let _ = demodulate(in_pcm, &radio).expect("demodulate");
            })
        },
    );

    group.finish();
}

fn bench_frame_codec(c: &mut Criterion) {
    let header = FrameHeader {
        version: 1,
        profile_id: 1,
        sequence: 9,
        fec_id: 1,
        kind: FrameKind::Data,
    };
    let payload = [0x42_u8; DATA_FRAME_PAYLOAD_BYTES];
    let frame = Frame::new(header, &payload).expect("frame");
    let encoded = frame.encode(true).expect("encode");

    let mut group = c.benchmark_group("frame_codec");
    group.throughput(Throughput::Bytes(payload.len() as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let _ = frame.encode(true).expect("encode");
        })
    });

    group.bench_function("decode", |b| {
        b.iter(|| {
            let _ = Frame::decode(&encoded, true).expect("decode");
        })
    });

    group.finish();
}

fn bench_link_state_machine(c: &mut Criterion) {
    let mut group = c.benchmark_group("link_state_machine");

    group.bench_function("handshake+txrequest", |b| {
        b.iter_batched(
            || LinkMachine::new(LinkProfile::default()),
            |mut link| {
                let _ = link.link_tick(LinkEvent::PttReady);

                let probe = Frame::new(
                    FrameHeader {
                        version: 1,
                        profile_id: 0,
                        sequence: 0,
                        fec_id: 0,
                        kind: FrameKind::Probe,
                    },
                    &[],
                )
                .expect("probe frame");
                let _ = link.link_tick(LinkEvent::RxFrame(probe));

                let cap = Frame::new(
                    FrameHeader {
                        version: 1,
                        profile_id: 0,
                        sequence: 0,
                        fec_id: 0,
                        kind: FrameKind::Capability,
                    },
                    &[1, 2, 5],
                )
                .expect("cap frame");
                let _ = link.link_tick(LinkEvent::RxFrame(cap));

                let mut payload = HVec::<u8, MAX_PAYLOAD_BYTES>::new();
                payload
                    .extend_from_slice(b"benchmark-payload")
                    .expect("payload fits");
                let _ = link.link_tick(LinkEvent::TxRequest { payload });
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0_u8; bits.len().div_ceil(8)];
    for (idx, &bit) in bits.iter().enumerate() {
        if bit & 1 == 1 {
            out[idx / 8] |= 1 << (7 - idx % 8);
        }
    }
    out
}

fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for bit in (0..8).rev() {
            out.push((byte >> bit) & 1);
        }
    }
    out
}

fn split_signal_segments_adaptive(samples: &[i16], zero_threshold: i16) -> Vec<(usize, usize)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    if peak == 0 {
        return Vec::new();
    }
    let adaptive = i16::try_from((peak / 24).max(16)).unwrap_or(i16::MAX);
    let gate = adaptive.max(zero_threshold.min(adaptive.saturating_mul(2)));
    let min_active = 48usize;
    let gap_hysteresis = 16usize;

    let mut out = Vec::<(usize, usize)>::new();
    let mut start = None::<usize>;
    let mut quiet_run = 0usize;
    for (idx, &sample) in samples.iter().enumerate() {
        let active = sample.abs() > gate;
        match (start, active) {
            (None, true) => {
                start = Some(idx);
                quiet_run = 0;
            }
            (Some(_), true) => quiet_run = 0,
            (Some(s), false) => {
                quiet_run += 1;
                if quiet_run >= gap_hysteresis {
                    let end = idx + 1 - quiet_run;
                    if end > s && end - s >= min_active {
                        out.push((s, end));
                    }
                    start = None;
                    quiet_run = 0;
                }
            }
            (None, false) => {}
        }
    }
    if let Some(s) = start
        && samples.len().saturating_sub(s) >= min_active
    {
        out.push((s, samples.len()));
    }
    out
}

fn find_data_frames_from_byte_stream(bytes: &[u8]) -> Vec<Frame> {
    let mut idx = 0usize;
    let mut out = Vec::<Frame>::new();
    while idx + 8 <= bytes.len() {
        let payload_len = usize::from(bytes[idx + 5]);
        if payload_len > MAX_PAYLOAD_BYTES {
            idx += 1;
            continue;
        }
        let frame_len = 6 + payload_len + 2;
        if idx + frame_len > bytes.len() {
            break;
        }

        let window = &bytes[idx..idx + frame_len];
        match Frame::decode(window, true) {
            Ok(frame) => {
                if frame.header.kind == FrameKind::Data {
                    out.push(frame);
                }
                idx += frame_len;
            }
            Err(_) => idx += 1,
        }
    }
    out
}

fn decode_best_data_frame_from_segment(
    samples: &[i16],
    radio: &RadioProfile,
    sps: usize,
    expected_seq: Option<u8>,
) -> Option<Frame> {
    let mut best = None::<Frame>;
    let mut best_payload_len = 0usize;

    for offset in 0..sps {
        if offset >= samples.len() {
            break;
        }
        let usable = (samples.len() - offset) / sps * sps;
        if usable < sps * 8 {
            continue;
        }
        let block = &samples[offset..offset + usable];
        let demod = match demodulate(block, radio) {
            Ok(d) => d,
            Err(_) => continue,
        };

        for inverted in [false, true] {
            let hard_bits: Vec<u8> = if inverted {
                demod
                    .soft_bits
                    .iter()
                    .map(|llr| u8::from(*llr < 0.0))
                    .collect()
            } else {
                demod
                    .soft_bits
                    .iter()
                    .map(|llr| u8::from(*llr > 0.0))
                    .collect()
            };

            for lead_trim in 0..8usize {
                if hard_bits.len() <= lead_trim {
                    break;
                }
                let candidate_bits = &hard_bits[lead_trim..];
                let usable_bits = candidate_bits.len() & !7;
                if usable_bits < 64 {
                    continue;
                }
                let candidate_bytes = bits_to_bytes(&candidate_bits[..usable_bits]);
                for frame in find_data_frames_from_byte_stream(&candidate_bytes) {
                    if let Some(expect) = expected_seq
                        && frame.header.sequence != expect
                    {
                        continue;
                    }
                    let payload_len = frame.payload.len();
                    if payload_len > best_payload_len {
                        best_payload_len = payload_len;
                        best = Some(frame);
                    }
                }
            }
        }
    }

    best
}

fn decode_payload_with_gap_jitter(
    samples: &[i16],
    radio: &RadioProfile,
    zero_threshold: i16,
) -> Vec<u8> {
    let sps = usize::try_from(48_000_u32 / radio.symbol_rate.as_hz()).unwrap_or(10);
    let mut expected_seq = None::<u8>;
    let mut out = Vec::<u8>::new();
    for (start, end) in split_signal_segments_adaptive(samples, zero_threshold) {
        if let Some(frame) =
            decode_best_data_frame_from_segment(&samples[start..end], radio, sps, expected_seq)
        {
            out.extend_from_slice(frame.payload.as_slice());
            expected_seq = Some((frame.header.sequence + 1) & 0x0F);
        }
    }
    out
}

fn build_gap_jitter_waveform(
    payload: &[u8],
    radio: &RadioProfile,
    gap_pattern_ms: &[u32],
) -> Vec<i16> {
    let mut pcm = Vec::<i16>::new();
    let mut seq = 0u8;
    for (frame_idx, chunk) in payload.chunks(DATA_FRAME_PAYLOAD_BYTES).enumerate() {
        let frame = Frame::new(
            FrameHeader {
                version: 1,
                profile_id: 0,
                sequence: seq,
                fec_id: 1,
                kind: FrameKind::Data,
            },
            chunk,
        )
        .expect("frame");
        let encoded = frame.encode(true).expect("encode");
        let bits = bytes_to_bits(&encoded);
        let tx = modulate(&bits, radio).expect("mod");
        pcm.extend_from_slice(&tx);

        let gap_ms = gap_pattern_ms[frame_idx % gap_pattern_ms.len()];
        let silence = (48_000_u64 * u64::from(gap_ms) / 1000) as usize;
        pcm.extend(std::iter::repeat_n(0_i16, silence));
        seq = (seq + 1) & 0x0F;
    }
    pcm
}

fn bench_decode_gap_jitter_realtime(c: &mut Criterion) {
    let mut radio = RadioProfile::default();
    radio.symbol_rate = SymbolRate::R4800;

    let payload: Vec<u8> = (0..1024).map(|i| ((i * 53 + 17) & 0xFF) as u8).collect();
    let gap_patterns: [(&str, [u32; 4]); 2] = [("mild", [2, 3, 2, 4]), ("stress", [0, 2, 11, 2])];
    let thresholds: [i16; 4] = [0, 2, 64, 2048];

    let mut group = c.benchmark_group("decode_gap_jitter_realtime");
    group.throughput(Throughput::Bytes(payload.len() as u64));

    for (name, gaps) in gap_patterns {
        let samples = build_gap_jitter_waveform(&payload, &radio, &gaps);
        let baseline = decode_payload_with_gap_jitter(&samples, &radio, thresholds[0]);
        assert!(!baseline.is_empty(), "baseline decode is empty for {name}");
        let audio_secs = samples.len() as f64 / 48_000.0;

        group.bench_with_input(
            BenchmarkId::new("decode", name),
            &samples,
            |b, in_samples| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for i in 0..iters {
                        let th = thresholds[(i as usize) % thresholds.len()];
                        let decoded = decode_payload_with_gap_jitter(in_samples, &radio, th);
                        assert_eq!(
                            decoded, baseline,
                            "decode changed under threshold jitter th={th}, scenario={name}"
                        );
                    }
                    let elapsed = start.elapsed();
                    let realtime_x = (audio_secs * iters as f64) / elapsed.as_secs_f64();
                    assert!(
                        realtime_x >= 1.0,
                        "decode realtime below 1x for {name}: {realtime_x:.2}x"
                    );
                    elapsed
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fec,
    bench_phy,
    bench_frame_codec,
    bench_link_state_machine,
    bench_decode_gap_jitter_realtime
);
criterion_main!(benches);
