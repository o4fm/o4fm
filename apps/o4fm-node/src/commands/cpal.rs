use std::time::{Duration, Instant};

use heapless::Vec as HVec;
use o4fm_audio::{AudioConfig, AudioIo, CpalRealtimeAudio};
use o4fm_core::{FecProfile, LinkProfile, MAX_PAYLOAD_BYTES, RadioProfile};
use o4fm_fec::{decode_ldpc, encode_ldpc};
use o4fm_link::{LinkAction, LinkEvent, LinkMachine};
use o4fm_phy::{demodulate, modulate};

use crate::cli::{CpalArgs, Pipeline};
use crate::commands::build_supported_profiles;
use crate::o4fm_payload::bits_to_bytes;

const TRAINING_SYMBOLS_PER_TONE: usize = 4;

pub(crate) fn run_cpal_mode(args: &CpalArgs) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate_hz = args.sample_rate;
    let frame_samples = args.frame_samples;
    let channels = args.channels.max(1);
    let seconds = args.seconds.max(1);
    let pipeline = args.pipeline;
    let input_device_id = args.input_device_id.as_deref();
    let output_device_id = args.output_device_id.as_deref();

    let cfg = AudioConfig {
        sample_rate_hz,
        channels,
        frame_samples,
    };

    let mut audio = CpalRealtimeAudio::new_with_device_ids(cfg, input_device_id, output_device_id)?;

    match pipeline {
        Pipeline::Passthrough => run_cpal_passthrough(&mut audio, cfg, seconds),
        Pipeline::Digital => run_cpal_digital(&mut audio, cfg, seconds, args),
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
    args: &CpalArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if cfg.channels != 1 {
        return Err("digital pipeline currently requires --channels=1".into());
    }

    let supported = build_supported_profiles(args.voice_bitrate);
    let mut link = LinkMachine::new_with_supported_profiles(LinkProfile::default(), &supported);
    let mut active = link.active_profile();
    let mut radio = active.to_radio_profile();
    let mut fec = active.to_fec_profile();
    let mut voice_bitrate_bps = active.voice_bitrate_bps;
    let mut geom = compute_geometry(cfg.sample_rate_hz, &radio, &fec)?;

    let startup_actions = link.link_tick(LinkEvent::PttReady);
    apply_enter_profile_actions(
        &startup_actions,
        cfg.sample_rate_hz,
        &mut active,
        &mut radio,
        &mut fec,
        &mut voice_bitrate_bps,
        &mut geom,
    )?;
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
        "o4fm-node cpal digital: {} Hz, mod={}FSK symbol={} block={} samples voice={}bps duration={}s",
        cfg.sample_rate_hz,
        radio.modulation.order(),
        radio.symbol_rate.as_hz(),
        geom.burst_samples,
        voice_bitrate_bps,
        seconds
    );

    while started.elapsed() < deadline {
        let n = audio.read_frame(&mut rx_buf)?;
        if n == 0 {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        rx_accum.extend_from_slice(&rx_buf[..n]);

        while rx_accum.len() >= geom.burst_samples {
            blocks_seen += 1;

            let block: Vec<i16> = rx_accum.drain(..geom.burst_samples).collect();
            let demod =
                demod_with_training_fallback(&block, &radio, geom.sps, geom.training_samples)?;

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
                    apply_enter_profile_actions(
                        &actions,
                        cfg.sample_rate_hz,
                        &mut active,
                        &mut radio,
                        &mut fec,
                        &mut voice_bitrate_bps,
                        &mut geom,
                    )?;

                    // Re-encode and remodulate as digital relay path.
                    let reencoded = encode_ldpc(&decoded_bits, &fec);
                    let tx_pcm = modulate_with_training_prefix(&reencoded, &radio)?;
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
    println!(
        "  active_profile:  id={} mod={}FSK sym={} fec=LDPC({}/{}) voice={}bps",
        active.profile_id,
        active.modulation.order(),
        active.symbol_rate.as_hz(),
        active.code_n,
        active.code_k,
        active.voice_bitrate_bps
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct RuntimeGeometry {
    sps: usize,
    training_samples: usize,
    burst_samples: usize,
}

fn compute_geometry(
    sample_rate_hz: u32,
    radio: &RadioProfile,
    fec: &FecProfile,
) -> Result<RuntimeGeometry, Box<dyn std::error::Error>> {
    let bps = radio.modulation.bits_per_symbol();
    let sps = usize::try_from(sample_rate_hz / radio.symbol_rate.as_hz())
        .map_err(|_| "invalid samples-per-symbol")?;
    let symbols_per_block = usize::from(fec.code_n).div_ceil(bps);
    let data_samples = symbols_per_block * sps;
    let training_samples = usize::from(radio.modulation.order()) * TRAINING_SYMBOLS_PER_TONE * sps;
    Ok(RuntimeGeometry {
        sps,
        training_samples,
        burst_samples: training_samples + data_samples,
    })
}

fn apply_enter_profile_actions(
    actions: &HVec<LinkAction, 8>,
    sample_rate_hz: u32,
    active: &mut o4fm_core::NegotiationProfile,
    radio: &mut RadioProfile,
    fec: &mut FecProfile,
    voice_bitrate_bps: &mut u32,
    geom: &mut RuntimeGeometry,
) -> Result<(), Box<dyn std::error::Error>> {
    for action in actions {
        if let LinkAction::EnterProfile { profile } = action {
            *active = *profile;
            *radio = profile.to_radio_profile();
            *fec = profile.to_fec_profile();
            *voice_bitrate_bps = profile.voice_bitrate_bps;
            *geom = compute_geometry(sample_rate_hz, radio, fec)?;
            println!(
                "  negotiated profile committed: id={} mod={}FSK sym={} fec=LDPC({}/{}) voice={}bps burst={} samples",
                profile.profile_id,
                profile.modulation.order(),
                profile.symbol_rate.as_hz(),
                profile.code_n,
                profile.code_k,
                profile.voice_bitrate_bps,
                geom.burst_samples
            );
        }
    }
    Ok(())
}

fn modulate_with_training_prefix(
    encoded_bits: &[u8],
    radio: &RadioProfile,
) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let mut burst_bits = build_training_bits(radio);
    burst_bits.extend_from_slice(encoded_bits);
    Ok(modulate(&burst_bits, radio)?)
}

fn build_training_bits(radio: &RadioProfile) -> Vec<u8> {
    let bits_per_symbol = radio.modulation.bits_per_symbol();
    let tone_count = usize::from(radio.modulation.order());
    let mut bits =
        Vec::<u8>::with_capacity(tone_count * TRAINING_SYMBOLS_PER_TONE * bits_per_symbol);
    for tone in 0..tone_count {
        let sym_bin = gray_to_bin(tone as u16) as usize;
        for _ in 0..TRAINING_SYMBOLS_PER_TONE {
            for bit in 0..bits_per_symbol {
                let shift = bits_per_symbol - 1 - bit;
                bits.push(u8::from(((sym_bin >> shift) & 1) != 0));
            }
        }
    }
    bits
}

fn gray_to_bin(mut g: u16) -> u16 {
    let mut b = g;
    while g > 0 {
        g >>= 1;
        b ^= g;
    }
    b
}

fn demod_with_training_fallback(
    burst: &[i16],
    radio: &RadioProfile,
    sps: usize,
    training_samples: usize,
) -> Result<o4fm_phy::DemodResult, Box<dyn std::error::Error>> {
    if burst.len() <= training_samples {
        return Ok(demodulate(burst, radio)?);
    }
    let adjusted = estimate_tone_plan_from_training(burst, radio, sps).unwrap_or(*radio);
    let trimmed = &burst[training_samples..];
    match demodulate(trimmed, &adjusted) {
        Ok(d) => Ok(d),
        Err(_) => Ok(demodulate(trimmed, radio)?),
    }
}

fn estimate_tone_plan_from_training(
    burst: &[i16],
    radio: &RadioProfile,
    sps: usize,
) -> Option<RadioProfile> {
    let tone_count = usize::from(radio.modulation.order());
    if tone_count < 2 {
        return None;
    }
    let train_len = tone_count * TRAINING_SYMBOLS_PER_TONE * sps;
    if burst.len() < train_len {
        return None;
    }

    let expected_center = f32::from(radio.tone_center_hz.max(1));
    let expected_spacing = f32::from(radio.tone_spacing_hz.max(1));
    let span = expected_spacing * (tone_count as f32 - 1.0);
    let start_expected = expected_center - span / 2.0;
    let search_half = (expected_spacing * 0.7).max(250.0);
    let step_hz = 25.0_f32;
    let mut peaks = Vec::<f32>::with_capacity(tone_count);

    for tone in 0..tone_count {
        let sym_start = tone * TRAINING_SYMBOLS_PER_TONE * sps;
        let sym_end = sym_start + TRAINING_SYMBOLS_PER_TONE * sps;
        if sym_end > burst.len() {
            return None;
        }
        let slice = &burst[sym_start..sym_end];
        let expected = start_expected + tone as f32 * expected_spacing;
        let from = (expected - search_half).max(80.0);
        let to = (expected + search_half).min(23_000.0);
        let mut best_f = expected;
        let mut best_e = f32::MIN;
        let mut f = from;
        while f <= to {
            let e = goertzel_energy(slice, f, 48_000.0);
            if e > best_e {
                best_e = e;
                best_f = f;
            }
            f += step_hz;
        }
        peaks.push(best_f);
    }

    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut diffs = Vec::<f32>::with_capacity(tone_count.saturating_sub(1));
    for pair in peaks.windows(2) {
        diffs.push(pair[1] - pair[0]);
    }
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let spacing = diffs[diffs.len() / 2].clamp(100.0, 4_000.0);
    let center = (peaks.iter().sum::<f32>() / peaks.len() as f32).clamp(100.0, 20_000.0);

    let mut out = *radio;
    out.tone_center_hz = center.round() as u16;
    out.tone_spacing_hz = spacing.round() as u16;
    Some(out)
}

fn goertzel_energy(samples: &[i16], target_hz: f32, sample_rate_hz: f32) -> f32 {
    let w = 2.0 * std::f32::consts::PI * target_hz / sample_rate_hz;
    let coeff = 2.0 * w.cos();
    let mut q1 = 0.0_f32;
    let mut q2 = 0.0_f32;
    for &s in samples {
        let q0 = coeff * q1 - q2 + f32::from(s);
        q2 = q1;
        q1 = q0;
    }
    q1 * q1 + q2 * q2 - coeff * q1 * q2
}
