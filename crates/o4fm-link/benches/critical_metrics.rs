use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use heapless::Vec as HVec;
use o4fm_core::{
    DATA_FRAME_PAYLOAD_BYTES, FecProfile, Frame, FrameHeader, FrameKind, LinkProfile, RadioProfile,
    MAX_PAYLOAD_BYTES,
};
use o4fm_fec::{decode_ldpc, encode_ldpc};
use o4fm_link::{LinkEvent, LinkMachine};
use o4fm_phy::{demodulate, modulate};

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

    group.bench_with_input(BenchmarkId::new("modulate", 4800_u32), &bits, |b, in_bits| {
        b.iter(|| {
            let _ = modulate(in_bits, &radio).expect("modulate");
        })
    });

    group.bench_with_input(BenchmarkId::new("demodulate", 4800_u32), &pcm, |b, in_pcm| {
        b.iter(|| {
            let _ = demodulate(in_pcm, &radio).expect("demodulate");
        })
    });

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

criterion_group!(
    benches,
    bench_fec,
    bench_phy,
    bench_frame_codec,
    bench_link_state_machine
);
criterion_main!(benches);
