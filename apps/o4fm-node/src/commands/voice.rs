use std::io;

use o4fm_core::{LOGICAL_MODE_VOICE, callsign_ascii16};
use opus::{
    Application as OpusApplication, Bitrate as OpusBitrate, Channels as OpusChannels,
    Decoder as OpusDecoder, Encoder as OpusEncoder,
};

use crate::cli::{VoiceDspArg, VoiceRxArgs, VoiceTxArgs};
use crate::commands::{build_supported_profiles, callsign_to_string};
use crate::o4fm_payload::{read_o4fm_payload_from_wav, write_o4fm_payload_to_wav};
use crate::wav_io::{read_wav_mono_i16, resample_linear_i16, write_wav_mono_i16};

pub(crate) fn run_voice_tx_mode(args: &VoiceTxArgs) -> Result<(), Box<dyn std::error::Error>> {
    let in_wav = args.in_wav.as_str();
    let out_wav = args.out_wav.as_str();
    let inter_frame_silence_ms = args.inter_frame_silence_ms.clamp(0, 2000);
    let callsign = callsign_ascii16(args.callsign.as_str());
    let flags = args.lp_flags;
    let opus_bitrate = args.opus_bitrate.clamp(500, 510_000);
    let frame_ms = parse_opus_frame_ms(args.opus_frame_ms).unwrap_or(20);
    let dsp_mode = voice_dsp_from_arg(args.voice_dsp);

    let (samples, sample_rate) = read_wav_mono_i16(in_wav)?;
    let mut speech_8k = resample_linear_i16(&samples, sample_rate, 8_000);
    apply_voice_preprocess(&mut speech_8k, dsp_mode);
    let samples_per_frame = 8_000_usize * frame_ms / 1000;

    let mut enc = OpusEncoder::new(8_000, OpusChannels::Mono, OpusApplication::Voip)
        .map_err(|e| io::Error::other(format!("opus encoder init failed: {e}")))?;
    enc.set_bitrate(OpusBitrate::Bits(opus_bitrate))
        .map_err(|e| io::Error::other(format!("opus set bitrate failed: {e}")))?;
    enc.set_vbr(false)
        .map_err(|e| io::Error::other(format!("opus set vbr failed: {e}")))?;
    enc.set_complexity(10)
        .map_err(|e| io::Error::other(format!("opus set complexity failed: {e}")))?;

    let mut pcm_frame = vec![0_i16; samples_per_frame];
    let mut packet_buf = vec![0_u8; 512];
    let mut encoded_voice = Vec::<u8>::new();
    let mut opus_packets = 0usize;
    for chunk in speech_8k.chunks(samples_per_frame) {
        pcm_frame.fill(0);
        pcm_frame[..chunk.len()].copy_from_slice(chunk);
        let packet_len = enc
            .encode(&pcm_frame, &mut packet_buf)
            .map_err(|e| io::Error::other(format!("opus encode failed: {e}")))?;
        let packet_len_u16 =
            u16::try_from(packet_len).map_err(|_| io::Error::other("opus packet too large"))?;
        encoded_voice.extend_from_slice(&packet_len_u16.to_be_bytes());
        encoded_voice.extend_from_slice(&packet_buf[..packet_len]);
        opus_packets += 1;
    }

    let tx = write_o4fm_payload_to_wav(
        &encoded_voice,
        out_wav,
        callsign,
        flags,
        LOGICAL_MODE_VOICE,
        inter_frame_silence_ms,
        build_supported_profiles(opus_bitrate as u32)[0],
    )?;

    println!("voice-tx done");
    println!("  in_wav:            {in_wav}");
    println!("  out_wav:           {out_wav}");
    println!("  input_rate_hz:     {sample_rate}");
    println!("  speech_rate_hz:    8000");
    println!("  callsign:          {}", callsign_to_string(&callsign));
    println!("  lp_flags:          0x{flags:016X}");
    println!("  codec:             opus");
    println!("  voice_dsp:         {}", voice_dsp_name(dsp_mode));
    println!("  opus_bitrate:      {opus_bitrate}");
    println!("  opus_frame_ms:     {frame_ms}");
    println!("  opus_frame_samp:   {samples_per_frame}");
    println!("  opus_packets:      {opus_packets}");
    println!("  voice_bytes:       {}", encoded_voice.len());
    println!("  logical_packets:   {}", tx.logical_packets);
    println!("  frames_out:        {}", tx.frames_out);
    println!("  pcm_samples_total: {}", tx.pcm_samples_total);

    Ok(())
}

pub(crate) fn run_voice_rx_mode(args: &VoiceRxArgs) -> Result<(), Box<dyn std::error::Error>> {
    let in_wav = args.in_wav.as_str();
    let out_wav = args.out_wav.as_str();
    let zero_threshold = args.zero_threshold.max(0);
    let _opus_bitrate = args.opus_bitrate.clamp(500, 510_000);
    let dsp_mode = voice_dsp_from_arg(args.voice_dsp);

    let profiles = build_supported_profiles(args.opus_bitrate.max(0) as u32);
    let rx = read_o4fm_payload_from_wav(in_wav, zero_threshold, &profiles)?;
    let mut dec = OpusDecoder::new(8_000, OpusChannels::Mono)
        .map_err(|e| io::Error::other(format!("opus decoder init failed: {e}")))?;
    let mut speech = vec![0_i16; 960];
    let mut decoded = Vec::<i16>::new();

    let mut voice_packets = 0usize;
    let mut decode_errors = 0usize;
    let mut pos = 0usize;
    while pos + 2 <= rx.payload.len() {
        let packet_len = usize::from(u16::from_be_bytes([rx.payload[pos], rx.payload[pos + 1]]));
        if packet_len == 0 || pos + 2 + packet_len > rx.payload.len() {
            break;
        }
        let packet = &rx.payload[pos + 2..pos + 2 + packet_len];
        match dec.decode(packet, &mut speech, false) {
            Ok(samples_per_channel) => {
                decoded.extend_from_slice(&speech[..samples_per_channel]);
                voice_packets += 1;
            }
            Err(_) => {
                decode_errors += 1;
            }
        }
        pos += 2 + packet_len;
    }
    let trailing_voice_bytes = rx.payload.len().saturating_sub(pos);
    apply_voice_postprocess(&mut decoded, dsp_mode);

    write_wav_mono_i16(out_wav, 8_000, &decoded)?;

    println!("voice-rx done");
    println!("  in_wav:            {in_wav}");
    println!("  out_wav:           {out_wav}");
    println!("  speech_rate_hz:    8000");
    println!("  codec:             opus");
    println!("  voice_dsp:         {}", voice_dsp_name(dsp_mode));
    println!("  logical_frames:    {}", rx.logical_frames);
    println!("  voice_packets:     {voice_packets}");
    println!("  decode_errors:     {decode_errors}");
    println!("  trailing_voice_b:  {trailing_voice_bytes}");
    println!("  decoded_samples:   {}", decoded.len());
    println!("  parse_errors:      {}", rx.parse_errors);
    println!("  dropped_segments:  {}", rx.dropped_segments);

    Ok(())
}

#[derive(Clone, Copy)]
enum VoiceDspMode {
    Passthrough,
    Basic,
}

fn voice_dsp_from_arg(mode: VoiceDspArg) -> VoiceDspMode {
    match mode {
        VoiceDspArg::Passthrough => VoiceDspMode::Passthrough,
        VoiceDspArg::Basic => VoiceDspMode::Basic,
    }
}

fn voice_dsp_name(mode: VoiceDspMode) -> &'static str {
    match mode {
        VoiceDspMode::Passthrough => "passthrough",
        VoiceDspMode::Basic => "basic",
    }
}

fn apply_voice_preprocess(samples: &mut [i16], mode: VoiceDspMode) {
    if matches!(mode, VoiceDspMode::Passthrough) || samples.is_empty() {
        return;
    }

    // 1st-order DC block / high-pass to tame low-frequency rumble.
    let mut x_prev = 0.0_f32;
    let mut y_prev = 0.0_f32;
    for s in samples.iter_mut() {
        let x = f32::from(*s);
        let y = x - x_prev + 0.995 * y_prev;
        x_prev = x;
        y_prev = y;
        *s = y.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
    }

    // Slow AGC towards target RMS.
    let rms = rms_i16(samples);
    if rms > 1.0 {
        let target = 3500.0_f32;
        let gain = (target / rms).clamp(0.5, 3.0);
        for s in samples.iter_mut() {
            let y = f32::from(*s) * gain;
            *s = y.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
        }
    }
}

fn apply_voice_postprocess(samples: &mut [i16], mode: VoiceDspMode) {
    if matches!(mode, VoiceDspMode::Passthrough) || samples.is_empty() {
        return;
    }
    // Light limiter on decoder output.
    for s in samples.iter_mut() {
        let x = f32::from(*s) / f32::from(i16::MAX);
        let y = x.tanh() * 0.95;
        *s = (y * f32::from(i16::MAX))
            .round()
            .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
    }
}

fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples
        .iter()
        .map(|&s| {
            let x = f32::from(s);
            x * x
        })
        .sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn parse_opus_frame_ms(value: usize) -> Option<usize> {
    match value {
        10 => Some(10),
        20 => Some(20),
        40 => Some(40),
        60 => Some(60),
        _ => None,
    }
}
