#![cfg_attr(not(feature = "std"), no_std)]

use heapless::Vec;
use serde::{Deserialize, Serialize};

pub const DATA_FRAME_PAYLOAD_BYTES: usize = 128;
pub const LOGICAL_MTU_BYTES: usize = 1792;
pub const LOGICAL_MTU_FRAMES: usize = LOGICAL_MTU_BYTES / DATA_FRAME_PAYLOAD_BYTES;
pub const MAX_PAYLOAD_BYTES: usize = DATA_FRAME_PAYLOAD_BYTES;
pub const MAX_FRAME_BYTES: usize = 192;
pub const MAX_NEGOTIATION_PROFILES: usize = 8;

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
    pub preamble_symbols: u16,
    pub sync_word: u32,
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
            preamble_symbols: 64,
            sync_word: 0xD3_91_7A_C5,
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

const NEGOTIATION_PROFILE_BYTES: usize = 11;

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
        0,
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
    Ok(NegotiationProfile {
        profile_id: payload[0],
        modulation,
        symbol_rate,
        fec_scheme,
        code_n,
        code_k,
        interleaver_depth: payload[8],
        max_iterations: payload[9],
    })
}

fn push(out: &mut Vec<u8, MAX_FRAME_BYTES>, b: u8) -> Result<(), FrameCodecError> {
    out.push(b).map_err(|_| FrameCodecError::BufferTooSmall)
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
        }];
        let encoded = encode_capability_payload(&profiles).expect("encode payload");
        let decoded = decode_capability_payload(&encoded).expect("decode payload");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].symbol_rate, SymbolRate::R4800);
    }
}
