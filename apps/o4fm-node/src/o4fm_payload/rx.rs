use o4fm_core::{Frame, MAX_PAYLOAD_BYTES, NegotiationProfile, RadioProfile};
use o4fm_phy::demodulate;

use crate::wav_io::read_wav_mono_i16;

use super::framing::{bits_to_bytes, extract_logical_payloads};
use super::{O4fmRxStats, find_data_frames_from_byte_stream, fixed_negotiation_profile};

const TRAINING_SYMBOLS_PER_TONE: usize = 4;

pub(crate) fn read_o4fm_payload_from_wav(
    in_wav: &str,
    zero_threshold: i16,
    profiles: &[NegotiationProfile],
) -> Result<O4fmRxStats, Box<dyn std::error::Error>> {
    let (samples, _) = read_wav_mono_i16(in_wav)?;
    Ok(decode_o4fm_payload_from_samples_with_profiles(
        &samples,
        zero_threshold,
        profiles,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decode_o4fm_payload_from_samples(
    samples: &[i16],
    zero_threshold: i16,
) -> O4fmRxStats {
    decode_o4fm_payload_from_samples_with_profiles(
        samples,
        zero_threshold,
        core::slice::from_ref(&fixed_negotiation_profile()),
    )
}

pub(crate) fn decode_o4fm_payload_from_samples_with_profiles(
    samples: &[i16],
    zero_threshold: i16,
    profiles: &[NegotiationProfile],
) -> O4fmRxStats {
    let fallback = fixed_negotiation_profile();
    let mut best: Option<O4fmRxStats> = None;

    let selected_profiles = if profiles.is_empty() {
        core::slice::from_ref(&fallback)
    } else {
        profiles
    };

    for profile in selected_profiles {
        let candidate = decode_with_profile(samples, zero_threshold, *profile);
        if is_better_rx(&candidate, best.as_ref()) {
            best = Some(candidate);
        }
    }

    best.unwrap_or_else(|| decode_with_profile(samples, zero_threshold, fallback))
}

fn decode_with_profile(
    samples: &[i16],
    zero_threshold: i16,
    profile: NegotiationProfile,
) -> O4fmRxStats {
    let radio = profile.to_radio_profile();
    let bps = radio.modulation.bits_per_symbol();
    let sps = usize::try_from(48_000_u32 / radio.symbol_rate.as_hz()).unwrap_or(10);
    let (stream_reassembly, stream_decoded_frames) =
        decode_data_streaming(samples, &radio, sps, bps);
    let mut stream_payload = Vec::<u8>::new();
    let (stream_logical_frames, stream_parse_errors, stream_trailing_bytes) =
        extract_logical_payloads(&stream_reassembly, &mut stream_payload);

    let mut seg_reassembly = Vec::<u8>::new();
    let mut seg_decoded_frames = 0usize;
    let mut seg_dropped_segments = 0usize;
    let mut expected_seq = None::<u8>;
    let segments = split_signal_segments_adaptive(samples, zero_threshold);
    for (start, end) in &segments {
        match decode_best_data_frame_from_segment(
            &samples[*start..*end],
            &radio,
            sps,
            bps,
            expected_seq,
        ) {
            Some(frame) => {
                seg_reassembly.extend_from_slice(frame.payload.as_slice());
                seg_decoded_frames += 1;
                expected_seq = Some((frame.header.sequence + 1) & 0x0F);
            }
            None => seg_dropped_segments += 1,
        }
    }
    let mut seg_payload = Vec::<u8>::new();
    let (seg_logical_frames, seg_parse_errors, seg_trailing_bytes) =
        extract_logical_payloads(&seg_reassembly, &mut seg_payload);

    let use_segments = seg_logical_frames > stream_logical_frames
        || (seg_logical_frames == stream_logical_frames
            && seg_payload.len() > stream_payload.len())
        || (seg_logical_frames == stream_logical_frames
            && seg_payload.len() == stream_payload.len()
            && seg_parse_errors < stream_parse_errors)
        || (seg_logical_frames == stream_logical_frames
            && seg_payload.len() == stream_payload.len()
            && seg_parse_errors == stream_parse_errors
            && seg_trailing_bytes < stream_trailing_bytes);

    if use_segments {
        O4fmRxStats {
            payload: seg_payload,
            samples_in: samples.len(),
            decoded_frames: seg_decoded_frames,
            dropped_segments: seg_dropped_segments,
            logical_frames: seg_logical_frames,
            parse_errors: seg_parse_errors,
            trailing_bytes: seg_trailing_bytes,
            symbol_rate_hz: radio.symbol_rate.as_hz(),
            modulation_order: radio.modulation.order(),
        }
    } else {
        O4fmRxStats {
            payload: stream_payload,
            samples_in: samples.len(),
            decoded_frames: stream_decoded_frames,
            dropped_segments: 0,
            logical_frames: stream_logical_frames,
            parse_errors: stream_parse_errors,
            trailing_bytes: stream_trailing_bytes,
            symbol_rate_hz: radio.symbol_rate.as_hz(),
            modulation_order: radio.modulation.order(),
        }
    }
}

fn is_better_rx(candidate: &O4fmRxStats, current: Option<&O4fmRxStats>) -> bool {
    let Some(cur) = current else {
        return true;
    };
    candidate.logical_frames > cur.logical_frames
        || (candidate.logical_frames == cur.logical_frames
            && candidate.payload.len() > cur.payload.len())
        || (candidate.logical_frames == cur.logical_frames
            && candidate.payload.len() == cur.payload.len()
            && candidate.parse_errors < cur.parse_errors)
        || (candidate.logical_frames == cur.logical_frames
            && candidate.payload.len() == cur.payload.len()
            && candidate.parse_errors == cur.parse_errors
            && candidate.trailing_bytes < cur.trailing_bytes)
        || (candidate.logical_frames == cur.logical_frames
            && candidate.payload.len() == cur.payload.len()
            && candidate.parse_errors == cur.parse_errors
            && candidate.trailing_bytes == cur.trailing_bytes
            && candidate.decoded_frames > cur.decoded_frames)
}

fn decode_data_streaming(
    samples: &[i16],
    radio: &RadioProfile,
    sps: usize,
    _bits_per_symbol: usize,
) -> (Vec<u8>, usize) {
    let mut best_payload = Vec::<u8>::new();
    let mut best_decoded = 0usize;
    let mut best_logical_frames = 0usize;
    let mut best_logical_bytes = 0usize;
    let mut best_parse_errors = usize::MAX;
    let mut best_trailing_bytes = usize::MAX;

    for offset in 0..sps {
        if offset >= samples.len() {
            break;
        }
        let usable = (samples.len() - offset) / sps * sps;
        if usable < sps * 8 {
            continue;
        }
        let block = &samples[offset..offset + usable];
        let training_samples = training_sample_count(radio, sps);
        let adjusted_radio = estimate_tone_plan_from_training(block, radio, sps).unwrap_or(*radio);

        for (trim, active_radio) in [(0usize, *radio), (training_samples, adjusted_radio)] {
            if block.len() <= trim + sps * 8 {
                continue;
            }
            let demod = match demodulate(&block[trim..], &active_radio) {
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
                    let (payload, decoded) = extract_data_frames_from_byte_stream(&candidate_bytes);
                    let mut logical_payload = Vec::<u8>::new();
                    let (logical_frames, parse_errors, trailing_bytes) =
                        extract_logical_payloads(&payload, &mut logical_payload);
                    let logical_bytes = logical_payload.len();
                    let is_better = logical_frames > best_logical_frames
                        || (logical_frames == best_logical_frames
                            && logical_bytes > best_logical_bytes)
                        || (logical_frames == best_logical_frames
                            && logical_bytes == best_logical_bytes
                            && parse_errors < best_parse_errors)
                        || (logical_frames == best_logical_frames
                            && logical_bytes == best_logical_bytes
                            && parse_errors == best_parse_errors
                            && trailing_bytes < best_trailing_bytes)
                        || (logical_frames == best_logical_frames
                            && logical_bytes == best_logical_bytes
                            && parse_errors == best_parse_errors
                            && trailing_bytes == best_trailing_bytes
                            && decoded > best_decoded);
                    if is_better {
                        best_logical_frames = logical_frames;
                        best_logical_bytes = logical_bytes;
                        best_parse_errors = parse_errors;
                        best_trailing_bytes = trailing_bytes;
                        best_decoded = decoded;
                        best_payload = payload;
                    }
                }
            }
        }
    }

    (best_payload, best_decoded)
}

fn extract_data_frames_from_byte_stream(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut idx = 0usize;
    let mut out = Vec::<u8>::new();
    let mut decoded = 0usize;

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
                if frame.header.kind == o4fm_core::FrameKind::Data {
                    out.extend_from_slice(frame.payload.as_slice());
                    decoded += 1;
                }
                idx += frame_len;
            }
            Err(_) => idx += 1,
        }
    }

    (out, decoded)
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

fn decode_best_data_frame_from_segment(
    samples: &[i16],
    radio: &RadioProfile,
    sps: usize,
    _bits_per_symbol: usize,
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
        let training_samples = training_sample_count(radio, sps);
        let adjusted_radio = estimate_tone_plan_from_training(block, radio, sps).unwrap_or(*radio);
        for (trim, active_radio) in [(0usize, *radio), (training_samples, adjusted_radio)] {
            if block.len() <= trim + sps * 8 {
                continue;
            }
            let data_block = &block[trim..];
            let demod = match demodulate(data_block, &active_radio) {
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
    }

    best
}

fn training_sample_count(radio: &RadioProfile, sps: usize) -> usize {
    usize::from(radio.modulation.order()) * TRAINING_SYMBOLS_PER_TONE * sps
}

fn estimate_tone_plan_from_training(
    block: &[i16],
    radio: &RadioProfile,
    sps: usize,
) -> Option<RadioProfile> {
    let tone_count = usize::from(radio.modulation.order());
    if tone_count < 2 {
        return None;
    }
    let train_len = training_sample_count(radio, sps);
    if block.len() < train_len {
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
        if sym_end > block.len() {
            return None;
        }
        let slice = &block[sym_start..sym_end];
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
