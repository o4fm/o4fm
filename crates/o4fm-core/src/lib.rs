#![cfg_attr(not(feature = "std"), no_std)]

use heapless::Vec;
use serde::{Deserialize, Serialize};

pub const DATA_FRAME_PAYLOAD_BYTES: usize = 128;
pub const LOGICAL_MTU_BYTES: usize = 1792;
pub const LOGICAL_MTU_FRAMES: usize = LOGICAL_MTU_BYTES / DATA_FRAME_PAYLOAD_BYTES;
pub const LOGICAL_MAGIC: [u8; 4] = *b"O4FM";
pub const LOGICAL_PROTO_VERSION: u8 = 1;
pub const LOGICAL_CALLSIGN_BYTES: usize = 16;
pub const LOGICAL_FLAGS_BYTES: usize = 8;
pub const LOGICAL_MODE_BYTES: usize = 8;
pub const LOGICAL_HEADER_BYTES: usize =
    4 + 1 + 1 + 2 + LOGICAL_CALLSIGN_BYTES + LOGICAL_FLAGS_BYTES + LOGICAL_MODE_BYTES;
pub const LOGICAL_MAX_PAYLOAD_BYTES: usize = LOGICAL_MTU_BYTES - LOGICAL_HEADER_BYTES;
pub const LOGICAL_MODE_VOICE: u64 = 1;
pub const LOGICAL_MODE_TEXT: u64 = 2;
pub const LOGICAL_MODE_IP: u64 = 3;
pub const MAX_PAYLOAD_BYTES: usize = DATA_FRAME_PAYLOAD_BYTES;
pub const MAX_FRAME_BYTES: usize = 192;
pub const MAX_NEGOTIATION_PROFILES: usize = 8;
pub const DEFAULT_VOICE_BITRATE_BPS: u32 = 7_000;
pub const DEFAULT_TONE_CENTER_HZ: u16 = 6_000;
pub const DEFAULT_TONE_SPACING_HZ: u16 = 2_000;
pub const DEFAULT_RX_BANDWIDTH_HZ: u16 = 10_000;
pub const DEFAULT_AFC_RANGE_HZ: u16 = 600;
pub const DEFAULT_NOISE_ADAPT_K_Q8: u8 = 96;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modulation {
    FourFsk,
    EightFsk,
}

impl Modulation {
    #[must_use]
    pub fn order(self) -> u16 {
        match self {
            Self::FourFsk => 4,
            Self::EightFsk => 8,
        }
    }

    #[must_use]
    pub fn bits_per_symbol(self) -> usize {
        match self {
            Self::FourFsk => 2,
            Self::EightFsk => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolRate {
    R4800,
}

impl SymbolRate {
    #[must_use]
    pub fn as_hz(self) -> u32 {
        match self {
            Self::R4800 => 4800,
        }
    }
}

impl SymbolRate {
    #[must_use]
    pub fn from_hz(hz: u32) -> Option<Self> {
        match hz {
            4800 => Some(Self::R4800),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FecScheme {
    Ldpc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadioProfile {
    pub modulation: Modulation,
    pub symbol_rate: SymbolRate,
    pub deviation_hz: u16,
    pub tone_center_hz: u16,
    pub tone_spacing_hz: u16,
    pub rx_bandwidth_hz: u16,
    pub afc_range_hz: u16,
    pub preamble_symbols: u16,
    pub sync_word: u32,
    pub noise_adapt_k_q8: u8,
    pub continuous_noise_mode: bool,
    pub whitening: bool,
    pub max_payload_bytes: u16,
    pub bt_tenths: u8,
}

impl Default for RadioProfile {
    fn default() -> Self {
        Self {
            modulation: Modulation::FourFsk,
            symbol_rate: SymbolRate::R4800,
            deviation_hz: 3_500,
            tone_center_hz: DEFAULT_TONE_CENTER_HZ,
            tone_spacing_hz: DEFAULT_TONE_SPACING_HZ,
            rx_bandwidth_hz: DEFAULT_RX_BANDWIDTH_HZ,
            afc_range_hz: DEFAULT_AFC_RANGE_HZ,
            preamble_symbols: 64,
            sync_word: 0xD3_91_7A_C5,
            noise_adapt_k_q8: DEFAULT_NOISE_ADAPT_K_Q8,
            continuous_noise_mode: true,
            whitening: true,
            max_payload_bytes: DATA_FRAME_PAYLOAD_BYTES as u16,
            bt_tenths: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FecProfile {
    pub scheme: FecScheme,
    pub code_n: u16,
    pub code_k: u16,
    pub interleaver_depth: u8,
    pub max_iterations: u8,
}

impl Default for FecProfile {
    fn default() -> Self {
        Self {
            scheme: FecScheme::Ldpc,
            code_n: 256,
            code_k: 128,
            interleaver_depth: 8,
            max_iterations: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkProfile {
    pub window_size: u8,
    pub ack_timeout_ms: u16,
    pub max_retransmissions: u8,
    pub probe_retries: u8,
}

impl Default for LinkProfile {
    fn default() -> Self {
        Self {
            window_size: 1,
            ack_timeout_ms: 750,
            max_retransmissions: 5,
            probe_retries: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    Probe = 0,
    Capability = 1,
    Data = 2,
    Ack = 3,
    RetransmitRequest = 4,
    Select = 5,
    Commit = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegotiationProfile {
    pub profile_id: u8,
    pub modulation: Modulation,
    pub symbol_rate: SymbolRate,
    pub fec_scheme: FecScheme,
    pub code_n: u16,
    pub code_k: u16,
    pub interleaver_depth: u8,
    pub max_iterations: u8,
    pub voice_bitrate_bps: u32,
    pub tone_center_hz: u16,
    pub tone_spacing_hz: u16,
    pub rx_bandwidth_hz: u16,
    pub afc_range_hz: u16,
    pub preamble_symbols: u16,
    pub sync_word: u32,
    pub noise_adapt_k_q8: u8,
    pub continuous_noise_mode: bool,
}

impl NegotiationProfile {
    #[must_use]
    pub fn to_radio_profile(self) -> RadioProfile {
        let mut out = RadioProfile::default();
        out.modulation = self.modulation;
        out.symbol_rate = self.symbol_rate;
        out.tone_center_hz = self.tone_center_hz;
        out.tone_spacing_hz = self.tone_spacing_hz;
        out.rx_bandwidth_hz = self.rx_bandwidth_hz;
        out.afc_range_hz = self.afc_range_hz;
        out.preamble_symbols = self.preamble_symbols;
        out.sync_word = self.sync_word;
        out.noise_adapt_k_q8 = self.noise_adapt_k_q8;
        out.continuous_noise_mode = self.continuous_noise_mode;
        out
    }

    #[must_use]
    pub fn to_fec_profile(self) -> FecProfile {
        FecProfile {
            scheme: self.fec_scheme,
            code_n: self.code_n,
            code_k: self.code_k,
            interleaver_depth: self.interleaver_depth,
            max_iterations: self.max_iterations,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiationCodecError {
    InvalidFormat,
    BufferTooSmall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: u8,
    pub profile_id: u8,
    pub sequence: u8,
    pub fec_id: u8,
    pub kind: FrameKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Vec<u8, MAX_PAYLOAD_BYTES>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCodecError {
    BufferTooSmall,
    PayloadTooLarge,
    InvalidFormat,
    CrcMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalFrameHeader {
    pub total_size: u16,
    pub callsign: [u8; LOGICAL_CALLSIGN_BYTES],
    pub flags: u64,
    pub mode: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalFrame {
    pub header: LogicalFrameHeader,
    pub payload: Vec<u8, LOGICAL_MAX_PAYLOAD_BYTES>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalFrameCodecError {
    BufferTooSmall,
    PayloadTooLarge,
    InvalidFormat,
    MagicMismatch,
    UnsupportedVersion,
    LengthMismatch,
}

impl Frame {
    #[must_use]
    pub fn new(header: FrameHeader, payload: &[u8]) -> Option<Self> {
        let mut v = Vec::<u8, MAX_PAYLOAD_BYTES>::new();
        if v.extend_from_slice(payload).is_err() {
            return None;
        }
        Some(Self { header, payload: v })
    }

    pub fn encode(&self, whitening: bool) -> Result<Vec<u8, MAX_FRAME_BYTES>, FrameCodecError> {
        if self.payload.len() > MAX_PAYLOAD_BYTES {
            return Err(FrameCodecError::PayloadTooLarge);
        }

        let payload_len_u8 =
            u8::try_from(self.payload.len()).map_err(|_| FrameCodecError::PayloadTooLarge)?;

        let mut out = Vec::<u8, MAX_FRAME_BYTES>::new();
        push(&mut out, self.header.version)?;
        push(&mut out, self.header.profile_id)?;
        push(&mut out, self.header.sequence)?;
        push(&mut out, self.header.fec_id)?;
        push(&mut out, self.header.kind as u8)?;
        push(&mut out, payload_len_u8)?;
        out.extend_from_slice(&self.payload)
            .map_err(|_| FrameCodecError::BufferTooSmall)?;

        if whitening {
            whiten_in_place(&mut out[6..]);
        }

        let crc = crc16_ccitt_false(&out);
        push(&mut out, (crc >> 8) as u8)?;
        push(&mut out, (crc & 0xFF) as u8)?;
        Ok(out)
    }

    pub fn decode(buf: &[u8], whitening: bool) -> Result<Self, FrameCodecError> {
        if buf.len() < 8 {
            return Err(FrameCodecError::InvalidFormat);
        }

        let received_crc = (u16::from(buf[buf.len() - 2]) << 8) | u16::from(buf[buf.len() - 1]);
        let computed_crc = crc16_ccitt_false(&buf[..buf.len() - 2]);
        if received_crc != computed_crc {
            return Err(FrameCodecError::CrcMismatch);
        }

        let kind = match buf[4] {
            0 => FrameKind::Probe,
            1 => FrameKind::Capability,
            2 => FrameKind::Data,
            3 => FrameKind::Ack,
            4 => FrameKind::RetransmitRequest,
            5 => FrameKind::Select,
            6 => FrameKind::Commit,
            _ => return Err(FrameCodecError::InvalidFormat),
        };

        let payload_len = usize::from(buf[5]);
        if payload_len > MAX_PAYLOAD_BYTES {
            return Err(FrameCodecError::PayloadTooLarge);
        }
        let body_end = 6 + payload_len;
        if body_end + 2 != buf.len() {
            return Err(FrameCodecError::InvalidFormat);
        }

        let mut payload = Vec::<u8, MAX_PAYLOAD_BYTES>::new();
        payload
            .extend_from_slice(&buf[6..body_end])
            .map_err(|_| FrameCodecError::BufferTooSmall)?;
        if whitening {
            whiten_in_place(&mut payload);
        }

        Ok(Self {
            header: FrameHeader {
                version: buf[0],
                profile_id: buf[1],
                sequence: buf[2],
                fec_id: buf[3],
                kind,
            },
            payload,
        })
    }
}

impl LogicalFrame {
    #[must_use]
    pub fn new(
        callsign: [u8; LOGICAL_CALLSIGN_BYTES],
        flags: u64,
        mode: u64,
        payload: &[u8],
    ) -> Option<Self> {
        if payload.len() > LOGICAL_MAX_PAYLOAD_BYTES {
            return None;
        }
        let total_size = payload.len() + LOGICAL_HEADER_BYTES;
        let total_size_u16 = u16::try_from(total_size).ok()?;
        let mut v = Vec::<u8, LOGICAL_MAX_PAYLOAD_BYTES>::new();
        if v.extend_from_slice(payload).is_err() {
            return None;
        }
        Some(Self {
            header: LogicalFrameHeader {
                total_size: total_size_u16,
                callsign,
                flags,
                mode,
            },
            payload: v,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8, LOGICAL_MTU_BYTES>, LogicalFrameCodecError> {
        if self.payload.len() > LOGICAL_MAX_PAYLOAD_BYTES {
            return Err(LogicalFrameCodecError::PayloadTooLarge);
        }
        let expected_size = LOGICAL_HEADER_BYTES + self.payload.len();
        if usize::from(self.header.total_size) != expected_size || expected_size > LOGICAL_MTU_BYTES
        {
            return Err(LogicalFrameCodecError::LengthMismatch);
        }

        let mut out = Vec::<u8, LOGICAL_MTU_BYTES>::new();
        push_logical(&mut out, LOGICAL_MAGIC[0])?;
        push_logical(&mut out, LOGICAL_MAGIC[1])?;
        push_logical(&mut out, LOGICAL_MAGIC[2])?;
        push_logical(&mut out, LOGICAL_MAGIC[3])?;
        push_logical(&mut out, LOGICAL_PROTO_VERSION)?;
        push_logical(
            &mut out,
            u8::try_from(LOGICAL_HEADER_BYTES)
                .map_err(|_| LogicalFrameCodecError::InvalidFormat)?,
        )?;
        out.extend_from_slice(&self.header.total_size.to_be_bytes())
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;
        out.extend_from_slice(&self.header.callsign)
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;
        out.extend_from_slice(&self.header.flags.to_be_bytes())
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;
        out.extend_from_slice(&self.header.mode.to_be_bytes())
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;
        out.extend_from_slice(&self.payload)
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;

        Ok(out)
    }

    pub fn decode(buf: &[u8]) -> Result<Self, LogicalFrameCodecError> {
        if buf.len() < LOGICAL_HEADER_BYTES {
            return Err(LogicalFrameCodecError::InvalidFormat);
        }
        if buf[0..4] != LOGICAL_MAGIC {
            return Err(LogicalFrameCodecError::MagicMismatch);
        }
        if buf[4] != LOGICAL_PROTO_VERSION {
            return Err(LogicalFrameCodecError::UnsupportedVersion);
        }
        if usize::from(buf[5]) != LOGICAL_HEADER_BYTES {
            return Err(LogicalFrameCodecError::InvalidFormat);
        }

        let total_size = usize::from(u16::from_be_bytes([buf[6], buf[7]]));
        if !(LOGICAL_HEADER_BYTES..=LOGICAL_MTU_BYTES).contains(&total_size) {
            return Err(LogicalFrameCodecError::LengthMismatch);
        }
        if buf.len() != total_size {
            return Err(LogicalFrameCodecError::LengthMismatch);
        }

        let mut callsign = [0_u8; LOGICAL_CALLSIGN_BYTES];
        callsign.copy_from_slice(&buf[8..8 + LOGICAL_CALLSIGN_BYTES]);
        let flags_start = 8 + LOGICAL_CALLSIGN_BYTES;
        let mode_start = flags_start + LOGICAL_FLAGS_BYTES;
        let payload_start = mode_start + LOGICAL_MODE_BYTES;
        let flags = u64::from_be_bytes(
            buf[flags_start..flags_start + LOGICAL_FLAGS_BYTES]
                .try_into()
                .map_err(|_| LogicalFrameCodecError::InvalidFormat)?,
        );
        let mode = u64::from_be_bytes(
            buf[mode_start..mode_start + LOGICAL_MODE_BYTES]
                .try_into()
                .map_err(|_| LogicalFrameCodecError::InvalidFormat)?,
        );
        let mut payload = Vec::<u8, LOGICAL_MAX_PAYLOAD_BYTES>::new();
        payload
            .extend_from_slice(&buf[payload_start..total_size])
            .map_err(|_| LogicalFrameCodecError::BufferTooSmall)?;

        Ok(Self {
            header: LogicalFrameHeader {
                total_size: u16::try_from(total_size)
                    .map_err(|_| LogicalFrameCodecError::LengthMismatch)?,
                callsign,
                flags,
                mode,
            },
            payload,
        })
    }
}

const NEGOTIATION_PROFILE_BYTES: usize = 30;

pub fn encode_capability_payload(
    profiles: &[NegotiationProfile],
) -> Result<Vec<u8, MAX_PAYLOAD_BYTES>, NegotiationCodecError> {
    let mut out = Vec::<u8, MAX_PAYLOAD_BYTES>::new();
    out.push(u8::try_from(profiles.len()).map_err(|_| NegotiationCodecError::BufferTooSmall)?)
        .map_err(|_| NegotiationCodecError::BufferTooSmall)?;
    for profile in profiles {
        encode_profile_into(&mut out, profile)?;
    }
    Ok(out)
}

pub fn decode_capability_payload(
    payload: &[u8],
) -> Result<Vec<NegotiationProfile, MAX_NEGOTIATION_PROFILES>, NegotiationCodecError> {
    if payload.is_empty() {
        return Err(NegotiationCodecError::InvalidFormat);
    }
    let count = usize::from(payload[0]);
    let expected = 1 + count * NEGOTIATION_PROFILE_BYTES;
    if payload.len() != expected || count > MAX_NEGOTIATION_PROFILES {
        return Err(NegotiationCodecError::InvalidFormat);
    }

    let mut out = Vec::<NegotiationProfile, MAX_NEGOTIATION_PROFILES>::new();
    let mut idx = 1usize;
    for _ in 0..count {
        let profile = decode_profile(&payload[idx..idx + NEGOTIATION_PROFILE_BYTES])?;
        out.push(profile)
            .map_err(|_| NegotiationCodecError::BufferTooSmall)?;
        idx += NEGOTIATION_PROFILE_BYTES;
    }

    Ok(out)
}

pub fn encode_selected_profile_payload(
    profile: &NegotiationProfile,
) -> Result<Vec<u8, MAX_PAYLOAD_BYTES>, NegotiationCodecError> {
    let mut out = Vec::<u8, MAX_PAYLOAD_BYTES>::new();
    encode_profile_into(&mut out, profile)?;
    Ok(out)
}

pub fn decode_selected_profile_payload(
    payload: &[u8],
) -> Result<NegotiationProfile, NegotiationCodecError> {
    if payload.len() != NEGOTIATION_PROFILE_BYTES {
        return Err(NegotiationCodecError::InvalidFormat);
    }
    decode_profile(payload)
}

fn encode_profile_into(
    out: &mut Vec<u8, MAX_PAYLOAD_BYTES>,
    profile: &NegotiationProfile,
) -> Result<(), NegotiationCodecError> {
    let modulation = match profile.modulation {
        Modulation::FourFsk => 0,
        Modulation::EightFsk => 1,
    };
    let symbol_rate = match profile.symbol_rate {
        SymbolRate::R4800 => 0,
    };
    let fec_scheme = match profile.fec_scheme {
        FecScheme::Ldpc => 0,
    };

    for byte in [
        profile.profile_id,
        modulation,
        symbol_rate,
        fec_scheme,
        (profile.code_n >> 8) as u8,
        (profile.code_n & 0xFF) as u8,
        (profile.code_k >> 8) as u8,
        (profile.code_k & 0xFF) as u8,
        profile.interleaver_depth,
        profile.max_iterations,
        ((profile.voice_bitrate_bps >> 24) & 0xFF) as u8,
        ((profile.voice_bitrate_bps >> 16) & 0xFF) as u8,
        ((profile.voice_bitrate_bps >> 8) & 0xFF) as u8,
        (profile.voice_bitrate_bps & 0xFF) as u8,
        (profile.tone_center_hz >> 8) as u8,
        (profile.tone_center_hz & 0xFF) as u8,
        (profile.tone_spacing_hz >> 8) as u8,
        (profile.tone_spacing_hz & 0xFF) as u8,
        (profile.rx_bandwidth_hz >> 8) as u8,
        (profile.rx_bandwidth_hz & 0xFF) as u8,
        (profile.afc_range_hz >> 8) as u8,
        (profile.afc_range_hz & 0xFF) as u8,
        (profile.preamble_symbols >> 8) as u8,
        (profile.preamble_symbols & 0xFF) as u8,
        ((profile.sync_word >> 24) & 0xFF) as u8,
        ((profile.sync_word >> 16) & 0xFF) as u8,
        ((profile.sync_word >> 8) & 0xFF) as u8,
        (profile.sync_word & 0xFF) as u8,
        profile.noise_adapt_k_q8,
        u8::from(profile.continuous_noise_mode),
    ] {
        out.push(byte)
            .map_err(|_| NegotiationCodecError::BufferTooSmall)?;
    }
    Ok(())
}

fn decode_profile(payload: &[u8]) -> Result<NegotiationProfile, NegotiationCodecError> {
    if payload.len() != NEGOTIATION_PROFILE_BYTES {
        return Err(NegotiationCodecError::InvalidFormat);
    }
    let modulation = match payload[1] {
        0 => Modulation::FourFsk,
        1 => Modulation::EightFsk,
        _ => return Err(NegotiationCodecError::InvalidFormat),
    };
    let symbol_rate = match payload[2] {
        0 => SymbolRate::R4800,
        _ => return Err(NegotiationCodecError::InvalidFormat),
    };
    let fec_scheme = match payload[3] {
        0 => FecScheme::Ldpc,
        _ => return Err(NegotiationCodecError::InvalidFormat),
    };
    let code_n = (u16::from(payload[4]) << 8) | u16::from(payload[5]);
    let code_k = (u16::from(payload[6]) << 8) | u16::from(payload[7]);
    let voice_bitrate_bps = (u32::from(payload[10]) << 24)
        | (u32::from(payload[11]) << 16)
        | (u32::from(payload[12]) << 8)
        | u32::from(payload[13]);
    let tone_center_hz = (u16::from(payload[14]) << 8) | u16::from(payload[15]);
    let tone_spacing_hz = (u16::from(payload[16]) << 8) | u16::from(payload[17]);
    let rx_bandwidth_hz = (u16::from(payload[18]) << 8) | u16::from(payload[19]);
    let afc_range_hz = (u16::from(payload[20]) << 8) | u16::from(payload[21]);
    let preamble_symbols = (u16::from(payload[22]) << 8) | u16::from(payload[23]);
    let sync_word = (u32::from(payload[24]) << 24)
        | (u32::from(payload[25]) << 16)
        | (u32::from(payload[26]) << 8)
        | u32::from(payload[27]);
    let noise_adapt_k_q8 = payload[28];
    let continuous_noise_mode = payload[29] != 0;
    Ok(NegotiationProfile {
        profile_id: payload[0],
        modulation,
        symbol_rate,
        fec_scheme,
        code_n,
        code_k,
        interleaver_depth: payload[8],
        max_iterations: payload[9],
        voice_bitrate_bps,
        tone_center_hz,
        tone_spacing_hz,
        rx_bandwidth_hz,
        afc_range_hz,
        preamble_symbols,
        sync_word,
        noise_adapt_k_q8,
        continuous_noise_mode,
    })
}

fn push(out: &mut Vec<u8, MAX_FRAME_BYTES>, b: u8) -> Result<(), FrameCodecError> {
    out.push(b).map_err(|_| FrameCodecError::BufferTooSmall)
}

fn push_logical(out: &mut Vec<u8, LOGICAL_MTU_BYTES>, b: u8) -> Result<(), LogicalFrameCodecError> {
    out.push(b)
        .map_err(|_| LogicalFrameCodecError::BufferTooSmall)
}

#[must_use]
pub fn callsign_ascii16(input: &str) -> [u8; LOGICAL_CALLSIGN_BYTES] {
    let mut out = [b' '; LOGICAL_CALLSIGN_BYTES];
    for (idx, b) in input
        .as_bytes()
        .iter()
        .take(LOGICAL_CALLSIGN_BYTES)
        .enumerate()
    {
        out[idx] = b.to_ascii_uppercase();
    }
    out
}

#[must_use]
pub fn crc16_ccitt_false(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in bytes {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if (crc & 0x8000) != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

pub fn whiten_in_place(bytes: &mut [u8]) {
    let mut lfsr: u8 = 0x7D;
    for byte in bytes {
        let mut mask = 0u8;
        for bit in 0..8 {
            let tap = ((lfsr >> 6) ^ (lfsr >> 5)) & 0x01;
            lfsr = ((lfsr << 1) | tap) & 0x7F;
            mask |= tap << bit;
        }
        *byte ^= mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_vector() {
        assert_eq!(crc16_ccitt_false(b"123456789"), 0x29B1);
    }

    #[test]
    fn whiten_round_trip() {
        let mut data = [0x11, 0x22, 0x33, 0x44];
        let src = data;
        whiten_in_place(&mut data);
        whiten_in_place(&mut data);
        assert_eq!(data, src);
    }

    #[test]
    fn frame_round_trip() {
        let header = FrameHeader {
            version: 1,
            profile_id: 0,
            sequence: 7,
            fec_id: 1,
            kind: FrameKind::Data,
        };
        let frame = Frame::new(header, &[1, 2, 3, 4]).expect("payload fits");
        let enc = frame.encode(true).expect("encode ok");
        let dec = Frame::decode(&enc, true).expect("decode ok");
        assert_eq!(dec.header.sequence, 7);
        assert_eq!(dec.payload.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn capability_payload_round_trip() {
        let profiles = [NegotiationProfile {
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
        }];
        let encoded = encode_capability_payload(&profiles).expect("encode payload");
        let decoded = decode_capability_payload(&encoded).expect("decode payload");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].symbol_rate, SymbolRate::R4800);
        assert_eq!(decoded[0].voice_bitrate_bps, DEFAULT_VOICE_BITRATE_BPS);
        assert_eq!(decoded[0].tone_center_hz, DEFAULT_TONE_CENTER_HZ);
    }

    #[test]
    fn logical_frame_round_trip() {
        let frame = LogicalFrame::new(
            callsign_ascii16("N0CALL"),
            0x12,
            LOGICAL_MODE_TEXT,
            b"hello",
        )
        .expect("logical payload fits");
        let encoded = frame.encode().expect("logical encode");
        let decoded = LogicalFrame::decode(&encoded).expect("logical decode");
        assert_eq!(decoded.payload.as_slice(), b"hello");
        assert_eq!(decoded.header.mode, LOGICAL_MODE_TEXT);
        assert_eq!(decoded.header.flags, 0x12);
    }
}
