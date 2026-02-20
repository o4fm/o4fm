use o4fm_core::{
    DEFAULT_AFC_RANGE_HZ, DEFAULT_NOISE_ADAPT_K_Q8, DEFAULT_RX_BANDWIDTH_HZ,
    DEFAULT_TONE_CENTER_HZ, DEFAULT_TONE_SPACING_HZ, DEFAULT_VOICE_BITRATE_BPS, FecScheme, Frame,
    Modulation, NegotiationProfile, SymbolRate,
};

mod framing;
mod rx;
mod tx;

pub(crate) use framing::bits_to_bytes;
#[cfg(test)]
pub(crate) use rx::decode_o4fm_payload_from_samples;
pub(crate) use rx::read_o4fm_payload_from_wav;
pub(crate) use tx::{capture_send_frames, emit, write_demo_wave, write_o4fm_payload_to_wav};

pub(crate) struct O4fmTxStats {
    pub(crate) frames_out: usize,
    pub(crate) logical_packets: usize,
    pub(crate) pcm_samples_total: usize,
    pub(crate) symbol_rate_hz: u32,
    pub(crate) modulation_order: u16,
}

pub(crate) struct O4fmRxStats {
    pub(crate) payload: Vec<u8>,
    pub(crate) samples_in: usize,
    pub(crate) decoded_frames: usize,
    pub(crate) dropped_segments: usize,
    pub(crate) logical_frames: usize,
    pub(crate) parse_errors: usize,
    pub(crate) trailing_bytes: usize,
    pub(crate) symbol_rate_hz: u32,
    pub(crate) modulation_order: u16,
}

fn fixed_negotiation_profile() -> NegotiationProfile {
    NegotiationProfile {
        profile_id: 0,
        modulation: Modulation::FourFsk,
        symbol_rate: SymbolRate::R4800,
        fec_scheme: FecScheme::Ldpc,
        code_n: 256,
        code_k: 128,
        interleaver_depth: 8,
        max_iterations: 16,
        voice_bitrate_bps: DEFAULT_VOICE_BITRATE_BPS,
        tone_center_hz: DEFAULT_TONE_CENTER_HZ,
        tone_spacing_hz: DEFAULT_TONE_SPACING_HZ,
        rx_bandwidth_hz: DEFAULT_RX_BANDWIDTH_HZ,
        afc_range_hz: DEFAULT_AFC_RANGE_HZ,
        preamble_symbols: 64,
        sync_word: 0xD3_91_7A_C5,
        noise_adapt_k_q8: DEFAULT_NOISE_ADAPT_K_Q8,
        continuous_noise_mode: true,
    }
}

fn find_data_frames_from_byte_stream(bytes: &[u8]) -> Vec<Frame> {
    let mut idx = 0usize;
    let mut out = Vec::<Frame>::new();
    while idx + 8 <= bytes.len() {
        let payload_len = usize::from(bytes[idx + 5]);
        if payload_len > o4fm_core::MAX_PAYLOAD_BYTES {
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
                    out.push(frame);
                }
                idx += frame_len;
            }
            Err(_) => idx += 1,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use o4fm_core::{LOGICAL_MODE_TEXT, callsign_ascii16};

    use super::{
        decode_o4fm_payload_from_samples, read_o4fm_payload_from_wav, write_o4fm_payload_to_wav,
    };
    use crate::wav_io::read_wav_mono_i16;

    fn temp_wav_path(name: &str) -> String {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        p.push(format!("o4fm_{name}_{pid}_{nanos}.wav"));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn streaming_decode_is_stable_across_zero_threshold_values() {
        let wav_path = temp_wav_path("zero_threshold");
        let payload: Vec<u8> = (0..1024).map(|i| ((i * 37) & 0xFF) as u8).collect();

        write_o4fm_payload_to_wav(
            &payload,
            &wav_path,
            callsign_ascii16("T0TEST"),
            0,
            LOGICAL_MODE_TEXT,
            0,
            super::fixed_negotiation_profile(),
        )
        .expect("write test waveform");

        for threshold in [0_i16, 2, 64, 2048] {
            let rx = read_o4fm_payload_from_wav(
                &wav_path,
                threshold,
                &[super::fixed_negotiation_profile()],
            )
            .expect("streaming decode must succeed");
            assert_eq!(
                rx.payload, payload,
                "payload mismatch for zero-threshold={threshold}"
            );
            assert_eq!(
                rx.parse_errors, 0,
                "unexpected parse error for threshold={threshold}"
            );
        }

        let _ = std::fs::remove_file(&wav_path);
    }

    #[test]
    fn streaming_decode_is_stable_under_zero_threshold_jitter() {
        let payload: Vec<u8> = (0..1024).map(|i| ((i * 53 + 17) & 0xFF) as u8).collect();
        let callsign = callsign_ascii16("T0JITR");
        let test_gaps_ms = [0_u32, 2_u32];

        for gap_ms in test_gaps_ms {
            let wav_path = temp_wav_path(&format!("zero_threshold_jitter_gap{gap_ms}"));
            write_o4fm_payload_to_wav(
                &payload,
                &wav_path,
                callsign,
                0,
                LOGICAL_MODE_TEXT,
                gap_ms,
                super::fixed_negotiation_profile(),
            )
            .expect("write jitter waveform");
            let (samples, _) = read_wav_mono_i16(&wav_path).expect("read jitter waveform once");
            let baseline = decode_o4fm_payload_from_samples(&samples, 0);

            // Deterministic pseudo-random threshold jitter sequence.
            let mut state = 0x1234_5678_u32 ^ gap_ms;
            for _ in 0..4 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let threshold = ((state >> 8) & 0x7FFF) as i16;
                let rx = decode_o4fm_payload_from_samples(&samples, threshold);
                assert_eq!(
                    rx.payload, baseline.payload,
                    "payload changed under jittered threshold={threshold}, gap_ms={gap_ms}"
                );
                assert_eq!(
                    rx.parse_errors, baseline.parse_errors,
                    "parse_errors changed under jittered threshold={threshold}, gap_ms={gap_ms}"
                );
            }

            let _ = std::fs::remove_file(&wav_path);
        }
    }
}
