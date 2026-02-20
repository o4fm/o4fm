use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use heapless::Vec as HVec;
use o4fm_audio::{list_device_infos, AudioConfig, AudioIo, CpalRealtimeAudio, WavFileAudio};
use o4fm_core::{
    DATA_FRAME_PAYLOAD_BYTES, FecProfile, Frame, FrameHeader, FrameKind, LinkProfile,
    LOGICAL_MTU_BYTES, MAX_PAYLOAD_BYTES, Modulation, RadioProfile, SymbolRate,
};
use o4fm_fec::{decode_ldpc, encode_ldpc};
use o4fm_link::{LinkAction, LinkEvent, LinkMachine};
use o4fm_phy::{demodulate, modulate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(());
    }

    if has_flag(&args, "--list-devices") {
        print_devices()?;
        return Ok(());
    }

    let mode = arg_value(&args, "--mode").unwrap_or("demo");
    match mode {
        "demo" => run_demo(&args)?,
        "cpal" => run_cpal_mode(&args)?,
        "wav" => run_wav_mode(&args)?,
        "bin2wav" => run_bin_to_wav_mode(&args)?,
        "wav2bin" => run_wav_to_bin_mode(&args)?,
        other => {
            eprintln!("unknown mode: {other}");
            print_help();
        }
    }

    Ok(())
}

fn print_devices() -> Result<(), Box<dyn std::error::Error>> {
    let devices = list_device_infos()?;
    println!("available audio devices:");
    for dev in devices {
        println!(
            "  id={} | in={} out={} | {}",
            dev.id, dev.supports_input, dev.supports_output, dev.name
        );
    }
    Ok(())
}

fn run_demo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut a = LinkMachine::new(LinkProfile::default());
    let mut b = LinkMachine::new(LinkProfile::default());
    let mut captured_frames = Vec::<Frame>::new();
    let wave_out = arg_value(args, "--demo-wave-out").unwrap_or("target/o4fm-node-demo.wav");

    println!("o4fm-node demo (dual-peer negotiation)");
    let mut a_actions = a.link_tick(LinkEvent::PttReady);
    capture_send_frames(&a_actions, &mut captured_frames);
    emit(a_actions.clone());
    let mut b_actions = HVec::<LinkAction, 8>::new();

    for _ in 0..8 {
        let mut progressed = false;

        let frames_to_b = extract_frames(&a_actions);
        a_actions.clear();
        for frame in frames_to_b {
            let out = b.link_tick(LinkEvent::RxFrame(frame));
            if !out.is_empty() {
                progressed = true;
                capture_send_frames(&out, &mut captured_frames);
                emit(out.clone());
                for action in out {
                    let _ = b_actions.push(action);
                }
            }
        }

        let frames_to_a = extract_frames(&b_actions);
        b_actions.clear();
        for frame in frames_to_a {
            let out = a.link_tick(LinkEvent::RxFrame(frame));
            if !out.is_empty() {
                progressed = true;
                capture_send_frames(&out, &mut captured_frames);
                emit(out.clone());
                for action in out {
                    let _ = a_actions.push(action);
                }
            }
        }

        if !progressed {
            break;
        }
        if a.state() == o4fm_link::LinkState::Data && b.state() == o4fm_link::LinkState::Data {
            break;
        }
    }

    println!("  peer-a state={:?} active_profile={:?}", a.state(), a.active_profile());
    println!("  peer-b state={:?} active_profile={:?}", b.state(), b.active_profile());

    let mut payload = HVec::<u8, MAX_PAYLOAD_BYTES>::new();
    payload.extend_from_slice(b"hello o4fm").expect("payload fits");
    let tx_actions = a.link_tick(LinkEvent::TxRequest { payload });
    capture_send_frames(&tx_actions, &mut captured_frames);
    emit(tx_actions);

    write_demo_wave(&captured_frames, wave_out)?;
    println!("  demo wave written: {wave_out}");
    Ok(())
}

fn extract_frames(actions: &HVec<LinkAction, 8>) -> Vec<Frame> {
    actions
        .iter()
        .filter_map(|action| match action {
            LinkAction::SendFrame(frame) => Some(frame.clone()),
            _ => None,
        })
        .collect()
}

fn run_cpal_mode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate_hz = parse_or(arg_value(args, "--sample-rate"), 48_000_u32);
    let frame_samples = parse_or(arg_value(args, "--frame-samples"), 480_usize);
    let channels = parse_or(arg_value(args, "--channels"), 1_u16).max(1);
    let seconds = parse_or(arg_value(args, "--seconds"), 10_u64).max(1);
    let pipeline = arg_value(args, "--pipeline").unwrap_or("digital");
    let input_device_id = arg_value(args, "--input-device-id");
    let output_device_id = arg_value(args, "--output-device-id");

    let cfg = AudioConfig {
        sample_rate_hz,
        channels,
        frame_samples,
    };

    let mut audio = CpalRealtimeAudio::new_with_device_ids(cfg, input_device_id, output_device_id)?;

    match pipeline {
        "passthrough" => run_cpal_passthrough(&mut audio, cfg, seconds),
        "digital" => run_cpal_digital(&mut audio, cfg, seconds, args),
        other => {
            eprintln!("unknown pipeline: {other}, use digital|passthrough");
            Ok(())
        }
    }
}

fn run_cpal_passthrough(
    audio: &mut CpalRealtimeAudio,
    cfg: AudioConfig,
    seconds: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = vec![0_i16; cfg.frame_samples * usize::from(cfg.channels)];

    let started = Instant::now();
    let deadline = Duration::from_secs(seconds);
    let mut total_in = 0usize;
    let mut total_out = 0usize;

    println!(
        "o4fm-node cpal passthrough: {} Hz, {} ch, frame={} samples, duration={}s",
        cfg.sample_rate_hz, cfg.channels, cfg.frame_samples, seconds
    );

    while started.elapsed() < deadline {
        let n = audio.read_frame(&mut buf)?;
        if n > 0 {
            total_in += n;
            audio.write_frame(&buf[..n])?;
            total_out += n;
        } else {
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    println!("cpal passthrough done");
    println!("  samples_in:  {total_in}");
    println!("  samples_out: {total_out}");

    Ok(())
}

fn run_cpal_digital(
    audio: &mut CpalRealtimeAudio,
    cfg: AudioConfig,
    seconds: u64,
    _args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if cfg.channels != 1 {
        return Err("digital pipeline currently requires --channels=1".into());
    }

    let mut radio = RadioProfile::default();
    radio.modulation = Modulation::FourFsk;
    radio.symbol_rate = SymbolRate::R4800;

    let fec = FecProfile::default();
    let bps = radio.modulation.bits_per_symbol();
    let sps = usize::try_from(cfg.sample_rate_hz / radio.symbol_rate.as_hz())
        .map_err(|_| "invalid samples-per-symbol")?;
    let symbols_per_block = usize::from(fec.code_n).div_ceil(bps);
    let block_samples = symbols_per_block * sps;

    let mut link = LinkMachine::new(LinkProfile::default());
    let startup_actions = link.link_tick(LinkEvent::PttReady);
    let mut link_actions = startup_actions.len();
    let mut rx_buf = vec![0_i16; cfg.frame_samples];
    let mut rx_accum = Vec::<i16>::new();

    let started = Instant::now();
    let deadline = Duration::from_secs(seconds);

    let mut blocks_seen = 0usize;
    let mut decode_ok = 0usize;
    let mut decode_fail = 0usize;
    let mut tx_samples = 0usize;
    println!(
        "o4fm-node cpal digital: {} Hz, symbol={} bps, block={} samples, duration={}s",
        cfg.sample_rate_hz,
        radio.symbol_rate.as_hz(),
        block_samples,
        seconds
    );

    while started.elapsed() < deadline {
        let n = audio.read_frame(&mut rx_buf)?;
        if n == 0 {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        rx_accum.extend_from_slice(&rx_buf[..n]);

        while rx_accum.len() >= block_samples {
            blocks_seen += 1;

            let block: Vec<i16> = rx_accum.drain(..block_samples).collect();
            let demod = demodulate(&block, &radio)?;

            if demod.soft_bits.len() < usize::from(fec.code_n) {
                decode_fail += 1;
                continue;
            }
            let llr = &demod.soft_bits[..usize::from(fec.code_n)];

            match decode_ldpc(llr, &fec) {
                Ok(decoded_bits) => {
                    decode_ok += 1;

                    let bytes = bits_to_bytes(&decoded_bits);
                    let mut payload = HVec::<u8, MAX_PAYLOAD_BYTES>::new();
                    let copy_len = bytes.len().min(MAX_PAYLOAD_BYTES);
                    let _ = payload.extend_from_slice(&bytes[..copy_len]);
                    let actions = link.link_tick(LinkEvent::TxRequest { payload });
                    link_actions += actions.len();

                    // Re-encode and remodulate as digital relay path.
                    let reencoded = encode_ldpc(&decoded_bits, &fec);
                    let tx_pcm = modulate(&reencoded, &radio)?;
                    audio.write_frame(&tx_pcm)?;
                    tx_samples += tx_pcm.len();
                }
                Err(_) => {
                    decode_fail += 1;
                }
            }
        }
    }

    println!("cpal digital done");
    println!("  blocks_seen:     {blocks_seen}");
    println!("  decode_ok:       {decode_ok}");
    println!("  decode_fail:     {decode_fail}");
    println!("  link_actions:    {link_actions}");
    println!("  tx_samples:      {tx_samples}");

    Ok(())
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

fn run_wav_mode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let input = arg_value(args, "--in").ok_or("--in=... is required in wav mode")?;
    let output = arg_value(args, "--out").ok_or("--out=... is required in wav mode")?;

    let sample_rate_hz = parse_or(arg_value(args, "--sample-rate"), 48_000_u32);
    let frame_samples = parse_or(arg_value(args, "--frame-samples"), 480_usize).max(1);
    let gain = parse_or(arg_value(args, "--gain"), 1.0_f32);

    let mut audio = WavFileAudio::open(input, output, sample_rate_hz)?;
    let mut buf = vec![0_i16; frame_samples];
    let mut total = 0usize;

    println!(
        "o4fm-node wav mode: in={} out={} sample_rate={} frame={} gain={:.3}",
        input, output, sample_rate_hz, frame_samples, gain
    );

    loop {
        let n = audio.read_frame(&mut buf)?;
        if n == 0 {
            break;
        }

        if (gain - 1.0).abs() > f32::EPSILON {
            for sample in &mut buf[..n] {
                let v = f32::from(*sample) * gain;
                *sample = v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            }
        }

        audio.write_frame(&buf[..n])?;
        total += n;
    }

    audio.finalize()?;
    println!("wav session done");
    println!("  samples_processed: {total}");

    Ok(())
}

fn run_bin_to_wav_mode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let in_bin = arg_value(args, "--in-bin").ok_or("--in-bin=... is required in bin2wav mode")?;
    let out_wav =
        arg_value(args, "--out-wav").ok_or("--out-wav=... is required in bin2wav mode")?;
    let inter_frame_silence_ms =
        parse_or(arg_value(args, "--inter-frame-silence-ms"), 2_u32).clamp(0, 2000);

    let mut radio = RadioProfile::default();
    radio.modulation = Modulation::FourFsk;
    radio.symbol_rate = SymbolRate::R4800;
    let silence_samples = (48_000_u64 * u64::from(inter_frame_silence_ms) / 1000) as usize;

    if let Some(parent) = Path::new(out_wav).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let input = fs::read(in_bin)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_wav, spec)?;

    let mut seq = 0u8;
    let mut frames = 0usize;
    let mut logical_packets = 0usize;
    let mut pcm_samples = 0usize;
    for mtu_chunk in input.chunks(LOGICAL_MTU_BYTES) {
        logical_packets += 1;
        for frame_payload in mtu_chunk.chunks(DATA_FRAME_PAYLOAD_BYTES) {
            let frame = Frame::new(
                FrameHeader {
                    version: 1,
                    profile_id: 0,
                    sequence: seq,
                    fec_id: 1,
                    kind: FrameKind::Data,
                },
                frame_payload,
            )
            .ok_or_else(|| io::Error::other("chunk too large for frame"))?;

            let encoded = frame
                .encode(true)
                .map_err(|_| io::Error::other("frame encode failed"))?;
            let bits = bytes_to_bits(&encoded);
            let pcm = modulate(&bits, &radio).map_err(|_| {
                io::Error::other(
                    "modulate failed (likely invalid modulation/symbol-rate for ~10kHz channel)",
                )
            })?;

            for sample in pcm {
                writer.write_sample(sample)?;
                pcm_samples += 1;
            }
            for _ in 0..silence_samples {
                writer.write_sample(0_i16)?;
                pcm_samples += 1;
            }

            seq = (seq + 1) & 0x0F;
            frames += 1;
        }
    }

    writer.finalize()?;
    println!("bin2wav done");
    println!("  in_bin:            {in_bin}");
    println!("  out_wav:           {out_wav}");
    println!("  bytes_in:          {}", input.len());
    println!("  frames_out:        {frames}");
    println!("  logical_packets:   {logical_packets}");
    println!("  frame_payload:     {}B", DATA_FRAME_PAYLOAD_BYTES);
    println!("  logical_mtu:       {}B", LOGICAL_MTU_BYTES);
    println!("  symbol_rate:       {}", radio.symbol_rate.as_hz());
    println!("  modulation:        {}FSK", radio.modulation.order());
    println!("  pcm_samples_total: {pcm_samples}");

    Ok(())
}

fn run_wav_to_bin_mode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let in_wav = arg_value(args, "--in-wav").ok_or("--in-wav=... is required in wav2bin mode")?;
    let out_bin =
        arg_value(args, "--out-bin").ok_or("--out-bin=... is required in wav2bin mode")?;
    let zero_threshold = parse_or(arg_value(args, "--zero-threshold"), 2_i16).max(0);

    let mut radio = RadioProfile::default();
    radio.modulation = Modulation::FourFsk;
    radio.symbol_rate = SymbolRate::R4800;
    let bps = radio.modulation.bits_per_symbol();
    let sps = usize::try_from(48_000_u32 / radio.symbol_rate.as_hz())
        .map_err(|_| io::Error::other("invalid samples-per-symbol"))?;
    let min_silence_run = sps.saturating_mul(4).max(8);

    let mut reader = hound::WavReader::open(in_wav)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("wav2bin currently requires mono WAV".into());
    }

    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect()
            } else {
                let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| {
                        let v = s.unwrap_or(0) as f32 / scale;
                        (v * f32::from(i16::MAX))
                            .round()
                            .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
                            as i16
                    })
                    .collect()
            }
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| {
                (s.unwrap_or(0.0) * f32::from(i16::MAX))
                    .round()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
            })
            .collect(),
    };

    let segments = split_signal_segments(&samples, zero_threshold, min_silence_run);
    let mut out = Vec::<u8>::new();
    let mut decoded_frames = 0usize;
    let mut dropped_segments = 0usize;

    for (start, end) in segments {
        let mut len = end.saturating_sub(start);
        if len < sps * 8 {
            dropped_segments += 1;
            continue;
        }
        len -= len % sps;
        if len == 0 {
            dropped_segments += 1;
            continue;
        }
        let block = &samples[start..start + len];
        if let Some(frame) = decode_frame_from_segment(block, &radio, sps, bps) {
            if frame.header.kind == FrameKind::Data {
                out.extend_from_slice(frame.payload.as_slice());
                decoded_frames += 1;
            } else {
                dropped_segments += 1;
            }
        } else {
            dropped_segments += 1;
        }
    }

    if let Some(parent) = Path::new(out_bin).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_bin, &out)?;

    println!("wav2bin done");
    println!("  in_wav:            {in_wav}");
    println!("  out_bin:           {out_bin}");
    println!("  samples_in:        {}", samples.len());
    println!("  decoded_frames:    {decoded_frames}");
    println!("  dropped_segments:  {dropped_segments}");
    println!("  bytes_out:         {}", out.len());
    println!(
        "  logical_packets:   {}",
        out.len().div_ceil(LOGICAL_MTU_BYTES)
    );
    println!("  frame_payload:     {}B", DATA_FRAME_PAYLOAD_BYTES);
    println!("  logical_mtu:       {}B", LOGICAL_MTU_BYTES);
    println!("  symbol_rate:       {}", radio.symbol_rate.as_hz());
    println!("  modulation:        {}FSK", radio.modulation.order());

    Ok(())
}

fn emit<const N: usize>(actions: HVec<LinkAction, N>) {
    for action in actions {
        println!("  action: {action:?}");
    }
}

fn capture_send_frames<const N: usize>(actions: &HVec<LinkAction, N>, out: &mut Vec<Frame>) {
    for action in actions {
        if let LinkAction::SendFrame(frame) = action {
            out.push(frame.clone());
        }
    }
}

fn split_signal_segments(
    samples: &[i16],
    zero_threshold: i16,
    min_silence_run: usize,
) -> Vec<(usize, usize)> {
    let mut segments = Vec::<(usize, usize)>::new();
    let mut in_signal = false;
    let mut start = 0usize;
    let mut zero_run = 0usize;

    for (idx, &sample) in samples.iter().enumerate() {
        let is_zero = sample.unsigned_abs() <= zero_threshold as u16;
        if in_signal {
            if is_zero {
                zero_run += 1;
                if zero_run >= min_silence_run {
                    let end = idx + 1 - zero_run;
                    if end > start {
                        segments.push((start, end));
                    }
                    in_signal = false;
                    zero_run = 0;
                }
            } else {
                zero_run = 0;
            }
        } else if !is_zero {
            in_signal = true;
            start = idx;
            zero_run = 0;
        }
    }

    if in_signal && start < samples.len() {
        segments.push((start, samples.len()));
    }

    segments
}

fn decode_frame_from_segment(
    samples: &[i16],
    radio: &RadioProfile,
    sps: usize,
    bits_per_symbol: usize,
) -> Option<Frame> {
    let max_offset = sps.min(samples.len().saturating_sub(sps));
    for offset in 0..=max_offset {
        let usable = samples.len().saturating_sub(offset) / sps * sps;
        if usable < sps * 8 {
            continue;
        }
        let demod = demodulate(&samples[offset..offset + usable], radio).ok()?;
        if let Some(frame) = decode_frame_from_llr(&demod.soft_bits, bits_per_symbol) {
            return Some(frame);
        }
    }
    None
}

fn decode_frame_from_llr(soft_bits: &[f32], bits_per_symbol: usize) -> Option<Frame> {
    for inverted in [false, true] {
        let hard_bits: Vec<u8> = if inverted {
            soft_bits.iter().map(|llr| u8::from(*llr < 0.0)).collect()
        } else {
            soft_bits.iter().map(|llr| u8::from(*llr > 0.0)).collect()
        };

        // M-FSK packs bits into symbols, so we may have up to (bps-1) padded bits at tail.
        for trim in 0..bits_per_symbol {
            if hard_bits.len() <= trim {
                break;
            }
            let candidate_bits = &hard_bits[..hard_bits.len() - trim];
            let candidate_bytes = bits_to_bytes(candidate_bits);
            if let Ok(frame) = Frame::decode(&candidate_bytes, true) {
                return Some(frame);
            }
        }
    }
    None
}

fn write_demo_wave(frames: &[Frame], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let out_path = Path::new(output_path);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_path, spec)?;

    for frame in frames {
        // Pad short silence between frames to visualize boundaries in debuggers.
        for _ in 0..480 {
            writer.write_sample(0_i16)?;
        }

        let payload = frame
            .encode(true)
            .map_err(|_| io::Error::other("frame encode failed"))?;
        let bits = bytes_to_bits(&payload);
        let mut radio = RadioProfile::default();
        radio.symbol_rate = SymbolRate::R4800;
        let pcm = modulate(&bits, &radio).map_err(|_| io::Error::other("modulate failed"))?;
        for sample in pcm {
            writer.write_sample(sample)?;
        }
    }

    writer.finalize()?;
    Ok(())
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .find_map(|arg| arg.strip_prefix(&format!("{key}=")))
}

fn parse_or<T: std::str::FromStr>(value: Option<&str>, default: T) -> T {
    value.and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn print_help() {
    println!("o4fm-node modes:");
    println!("  --list-devices");
    println!("  --mode=demo --demo-wave-out=target/o4fm-node-demo.wav");
    println!(
        "  --mode=cpal --pipeline=digital|passthrough --sample-rate=48000 --channels=1 --frame-samples=480 --seconds=10"
    );
    println!("             --input-device-id=<ID> --output-device-id=<ID>");
    println!("             (fixed PHY: 4FSK@4800)");
    println!(
        "  --mode=wav --in=input.wav --out=output.wav --sample-rate=48000 --frame-samples=480 --gain=1.0"
    );
    println!(
        "  --mode=bin2wav --in-bin=input.bin --out-wav=output.wav --inter-frame-silence-ms=2  (fixed PHY: 4FSK@4800, frame=128B, MTU=1792B)"
    );
    println!(
        "  --mode=wav2bin --in-wav=input.wav --out-bin=output.bin --zero-threshold=2  (fixed PHY: 4FSK@4800, frame=128B, MTU=1792B)"
    );
}
