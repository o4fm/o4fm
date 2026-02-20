use heapless::Vec as HVec;
use o4fm_core::{
    DEFAULT_AFC_RANGE_HZ, DEFAULT_NOISE_ADAPT_K_Q8, DEFAULT_RX_BANDWIDTH_HZ,
    DEFAULT_TONE_CENTER_HZ, DEFAULT_TONE_SPACING_HZ, DEFAULT_VOICE_BITRATE_BPS, FecScheme,
    LOGICAL_MODE_IP, LOGICAL_MODE_TEXT, LOGICAL_MODE_VOICE, MAX_NEGOTIATION_PROFILES, Modulation,
    NegotiationProfile, SymbolRate,
};

use crate::cli::LpMode;

pub(crate) mod bin;
pub(crate) mod cpal;
pub(crate) mod demo;
pub(crate) mod tui;
pub(crate) mod voice;
pub(crate) mod wav;

pub(crate) fn lp_mode_to_u64(mode: LpMode) -> u64 {
    match mode {
        LpMode::Voice => LOGICAL_MODE_VOICE,
        LpMode::Text => LOGICAL_MODE_TEXT,
        LpMode::Ip => LOGICAL_MODE_IP,
    }
}

pub(crate) fn logical_mode_name(mode: u64) -> &'static str {
    match mode {
        LOGICAL_MODE_VOICE => "voice",
        LOGICAL_MODE_TEXT => "text",
        LOGICAL_MODE_IP => "ip",
        _ => "custom",
    }
}

pub(crate) fn callsign_to_string(callsign: &[u8; 16]) -> String {
    String::from_utf8_lossy(callsign).trim().to_string()
}

pub(crate) fn build_supported_profiles(
    voice_bitrate_bps: u32,
) -> HVec<NegotiationProfile, MAX_NEGOTIATION_PROFILES> {
    let mut out = HVec::<NegotiationProfile, MAX_NEGOTIATION_PROFILES>::new();
    let bitrate = if voice_bitrate_bps == 0 {
        DEFAULT_VOICE_BITRATE_BPS
    } else {
        voice_bitrate_bps
    };

    let _ = out.push(NegotiationProfile {
        profile_id: 0,
        modulation: Modulation::FourFsk,
        symbol_rate: SymbolRate::R4800,
        fec_scheme: FecScheme::Ldpc,
        code_n: 256,
        code_k: 128,
        interleaver_depth: 8,
        max_iterations: 16,
        voice_bitrate_bps: bitrate,
        tone_center_hz: DEFAULT_TONE_CENTER_HZ,
        tone_spacing_hz: DEFAULT_TONE_SPACING_HZ,
        rx_bandwidth_hz: DEFAULT_RX_BANDWIDTH_HZ,
        afc_range_hz: DEFAULT_AFC_RANGE_HZ,
        preamble_symbols: 64,
        sync_word: 0xD3_91_7A_C5,
        noise_adapt_k_q8: DEFAULT_NOISE_ADAPT_K_Q8,
        continuous_noise_mode: true,
    });

    out
}

pub(crate) fn find_profile_by_id(
    profiles: &[NegotiationProfile],
    profile_id: u8,
) -> Option<NegotiationProfile> {
    profiles
        .iter()
        .copied()
        .find(|p| p.profile_id == profile_id)
}
