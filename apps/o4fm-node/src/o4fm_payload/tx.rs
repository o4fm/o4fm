use std::fs;
use std::io;
use std::path::Path;

use heapless::Vec as HVec;
use o4fm_core::{
    DATA_FRAME_PAYLOAD_BYTES, Frame, FrameHeader, FrameKind, LOGICAL_MAX_PAYLOAD_BYTES,
    LogicalFrame, NegotiationProfile, RadioProfile, SymbolRate,
};
use o4fm_link::LinkAction;
use o4fm_phy::modulate;

use super::O4fmTxStats;
use super::framing::bytes_to_bits;

const TRAINING_SYMBOLS_PER_TONE: usize = 4;

pub(crate) fn write_o4fm_payload_to_wav(
    payload: &[u8],
    out_wav: &str,
    callsign: [u8; 16],
    flags: u64,
    mode: u64,
    inter_frame_silence_ms: u32,
    profile: NegotiationProfile,
) -> Result<O4fmTxStats, Box<dyn std::error::Error>> {
    let radio = profile.to_radio_profile();
    let silence_samples = (48_000_u64 * u64::from(inter_frame_silence_ms) / 1000) as usize;

    if let Some(parent) = Path::new(out_wav).parent()
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
    let mut writer = hound::WavWriter::create(out_wav, spec)?;

    let mut seq = 0u8;
    let mut frames = 0usize;
    let mut logical_packets = 0usize;
    let mut pcm_samples = 0usize;
    for mtu_chunk in payload.chunks(LOGICAL_MAX_PAYLOAD_BYTES) {
        logical_packets += 1;
        let logical = LogicalFrame::new(callsign, flags, mode, mtu_chunk)
            .ok_or_else(|| io::Error::other("logical frame too large"))?;
        let encoded_logical = logical
            .encode()
            .map_err(|_| io::Error::other("logical frame encode failed"))?;
        for frame_payload in encoded_logical.as_slice().chunks(DATA_FRAME_PAYLOAD_BYTES) {
            let frame = Frame::new(
                FrameHeader {
                    version: 1,
                    profile_id: profile.profile_id,
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
            let frame_bits = bytes_to_bits(&encoded);
            let mut burst_bits = build_training_bits(&radio);
            burst_bits.extend_from_slice(&frame_bits);
            let pcm = modulate(&burst_bits, &radio)
                .map_err(|_| io::Error::other("modulate failed for selected PHY"))?;

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
    Ok(O4fmTxStats {
        frames_out: frames,
        logical_packets,
        pcm_samples_total: pcm_samples,
        symbol_rate_hz: radio.symbol_rate.as_hz(),
        modulation_order: radio.modulation.order(),
    })
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

pub(crate) fn emit<const N: usize>(actions: HVec<LinkAction, N>) {
    for action in actions {
        println!("  action: {action:?}");
    }
}

pub(crate) fn capture_send_frames<const N: usize>(
    actions: &HVec<LinkAction, N>,
    out: &mut Vec<Frame>,
) {
    for action in actions {
        if let LinkAction::SendFrame(frame) = action {
            out.push(frame.clone());
        }
    }
}

pub(crate) fn write_demo_wave(
    frames: &[Frame],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
