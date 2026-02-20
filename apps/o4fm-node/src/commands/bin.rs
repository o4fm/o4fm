use std::fs;
use std::path::Path;

use o4fm_core::{DATA_FRAME_PAYLOAD_BYTES, LOGICAL_MTU_BYTES, callsign_ascii16};

use crate::cli::{Bin2WavArgs, Wav2BinArgs};
use crate::commands::{
    build_supported_profiles, callsign_to_string, find_profile_by_id, logical_mode_name,
    lp_mode_to_u64,
};
use crate::o4fm_payload::{read_o4fm_payload_from_wav, write_o4fm_payload_to_wav};

pub(crate) fn run_bin_to_wav_mode(args: &Bin2WavArgs) -> Result<(), Box<dyn std::error::Error>> {
    let in_bin = args.in_bin.as_str();
    let out_wav = args.out_wav.as_str();
    let inter_frame_silence_ms = args.inter_frame_silence_ms.clamp(0, 2000);

    let input = fs::read(in_bin)?;
    let callsign = callsign_ascii16(args.callsign.as_str());
    let flags = args.lp_flags;
    let mode = lp_mode_to_u64(args.lp_mode);
    let supported = build_supported_profiles(args.voice_bitrate);
    let selected = find_profile_by_id(&supported, args.profile_id)
        .ok_or_else(|| format!("profile-id {} is not in supported set", args.profile_id))?;
    let tx = write_o4fm_payload_to_wav(
        &input,
        out_wav,
        callsign,
        flags,
        mode,
        inter_frame_silence_ms,
        selected,
    )?;
    println!("bin2wav done");
    println!("  in_bin:            {in_bin}");
    println!("  out_wav:           {out_wav}");
    println!("  bytes_in:          {}", input.len());
    println!("  frames_out:        {}", tx.frames_out);
    println!("  logical_packets:   {}", tx.logical_packets);
    println!("  callsign:          {}", callsign_to_string(&callsign));
    println!("  lp_flags:          0x{flags:016X}");
    println!("  lp_mode:           {}", logical_mode_name(mode));
    println!("  frame_payload:     {DATA_FRAME_PAYLOAD_BYTES}B");
    println!("  logical_mtu:       {LOGICAL_MTU_BYTES}B");
    println!("  symbol_rate:       {}", tx.symbol_rate_hz);
    println!("  modulation:        {}FSK", tx.modulation_order);
    println!("  profile_id:        {}", selected.profile_id);
    println!("  voice_bitrate:     {} bps", selected.voice_bitrate_bps);
    println!("  pcm_samples_total: {}", tx.pcm_samples_total);

    Ok(())
}

pub(crate) fn run_wav_to_bin_mode(args: &Wav2BinArgs) -> Result<(), Box<dyn std::error::Error>> {
    let in_wav = args.in_wav.as_str();
    let out_bin = args.out_bin.as_str();
    let zero_threshold = args.zero_threshold.max(0);
    let supported = build_supported_profiles(args.voice_bitrate);
    let selected_profiles = if let Some(profile_id) = args.profile_id {
        let selected = find_profile_by_id(&supported, profile_id)
            .ok_or_else(|| format!("profile-id {profile_id} is not in supported set"))?;
        vec![selected]
    } else {
        supported.to_vec()
    };

    let rx = read_o4fm_payload_from_wav(in_wav, zero_threshold, &selected_profiles)?;
    if let Some(parent) = Path::new(out_bin).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let out = rx.payload.clone();
    fs::write(out_bin, &out)?;

    println!("wav2bin done");
    println!("  in_wav:            {in_wav}");
    println!("  out_bin:           {out_bin}");
    println!("  samples_in:        {}", rx.samples_in);
    println!("  decoded_frames:    {}", rx.decoded_frames);
    println!("  dropped_segments:  {}", rx.dropped_segments);
    println!("  logical_frames:    {}", rx.logical_frames);
    println!("  parse_errors:      {}", rx.parse_errors);
    println!("  trailing_bytes:    {}", rx.trailing_bytes);
    println!("  bytes_out:         {}", out.len());
    println!(
        "  logical_packets:   {}",
        out.len().div_ceil(LOGICAL_MTU_BYTES)
    );
    println!("  frame_payload:     {DATA_FRAME_PAYLOAD_BYTES}B");
    println!("  logical_mtu:       {LOGICAL_MTU_BYTES}B");
    println!("  symbol_rate:       {}", rx.symbol_rate_hz);
    println!("  modulation:        {}FSK", rx.modulation_order);
    println!("  profiles_tried:    {}", selected_profiles.len());

    Ok(())
}
