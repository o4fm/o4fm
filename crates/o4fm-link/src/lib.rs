use heapless::Vec;
use o4fm_core::{
    DEFAULT_AFC_RANGE_HZ, DEFAULT_NOISE_ADAPT_K_Q8, DEFAULT_RX_BANDWIDTH_HZ,
    DEFAULT_TONE_CENTER_HZ, DEFAULT_TONE_SPACING_HZ, DEFAULT_VOICE_BITRATE_BPS, FecScheme, Frame,
    FrameHeader, FrameKind, LinkProfile, MAX_NEGOTIATION_PROFILES, MAX_PAYLOAD_BYTES, Modulation,
    NegotiationProfile, SymbolRate, decode_capability_payload, decode_selected_profile_payload,
    encode_capability_payload, encode_selected_profile_payload,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    Idle,
    Probe,
    Negotiating,
    Data,
    Fallback,
    Reacquire,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkEvent {
    RxFrame(Frame),
    TxRequest { payload: Vec<u8, MAX_PAYLOAD_BYTES> },
    Timeout,
    PttReady,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkAction {
    SendFrame(Frame),
    SendAck { sequence: u8 },
    Retransmit(Frame),
    EnterProfile { profile: NegotiationProfile },
}

#[derive(Debug, Clone)]
pub struct LinkMachine {
    state: LinkState,
    profile: LinkProfile,
    sequence: u8,
    retries: u8,
    pending_tx: Option<Frame>,
    supported_profiles: Vec<NegotiationProfile, MAX_NEGOTIATION_PROFILES>,
    remote_profiles: Vec<NegotiationProfile, MAX_NEGOTIATION_PROFILES>,
    pending_profile: Option<NegotiationProfile>,
    active_profile: NegotiationProfile,
    capability_sent: bool,
}

impl LinkMachine {
    #[must_use]
    pub fn new(profile: LinkProfile) -> Self {
        let supported = default_profiles();
        Self::new_with_supported_profiles(profile, &supported)
    }

    #[must_use]
    pub fn new_with_supported_profiles(
        profile: LinkProfile,
        supported: &[NegotiationProfile],
    ) -> Self {
        let mut supported_profiles = Vec::<NegotiationProfile, MAX_NEGOTIATION_PROFILES>::new();
        for p in supported {
            let _ = supported_profiles.push(*p);
        }
        let active_profile = supported_profiles
            .first()
            .copied()
            .unwrap_or(default_profiles()[0]);

        Self {
            state: LinkState::Idle,
            profile,
            sequence: 0,
            retries: 0,
            pending_tx: None,
            supported_profiles,
            remote_profiles: Vec::new(),
            pending_profile: None,
            active_profile,
            capability_sent: false,
        }
    }

    #[must_use]
    pub fn state(&self) -> LinkState {
        self.state
    }

    #[must_use]
    pub fn active_profile(&self) -> NegotiationProfile {
        self.active_profile
    }

    #[must_use]
    pub fn link_tick(&mut self, event: LinkEvent) -> Vec<LinkAction, 8> {
        let mut out = Vec::<LinkAction, 8>::new();

        match (&self.state, event) {
            (LinkState::Idle, LinkEvent::PttReady)
            | (LinkState::Fallback, LinkEvent::PttReady)
            | (LinkState::Reacquire, LinkEvent::PttReady) => {
                self.state = LinkState::Probe;
                self.capability_sent = false;
                self.remote_profiles.clear();
                self.pending_profile = None;
                let probe = self.build_control_frame(FrameKind::Probe, &[]);
                let _ = out.push(LinkAction::SendFrame(probe));
            }

            (LinkState::Idle, LinkEvent::RxFrame(frame))
            | (LinkState::Probe, LinkEvent::RxFrame(frame))
                if frame.header.kind == FrameKind::Probe =>
            {
                self.state = LinkState::Negotiating;
                if !self.capability_sent {
                    if let Ok(payload) = encode_capability_payload(&self.supported_profiles) {
                        let cap = self.build_control_frame(FrameKind::Capability, &payload);
                        let _ = out.push(LinkAction::SendFrame(cap));
                        self.capability_sent = true;
                    }
                }
            }

            (LinkState::Probe, LinkEvent::RxFrame(frame))
            | (LinkState::Negotiating, LinkEvent::RxFrame(frame))
                if frame.header.kind == FrameKind::Capability =>
            {
                self.state = LinkState::Negotiating;
                if let Ok(remote) = decode_capability_payload(frame.payload.as_slice()) {
                    self.remote_profiles = remote;
                    if !self.capability_sent {
                        if let Ok(payload) = encode_capability_payload(&self.supported_profiles) {
                            let cap = self.build_control_frame(FrameKind::Capability, &payload);
                            let _ = out.push(LinkAction::SendFrame(cap));
                            self.capability_sent = true;
                        }
                    }

                    if let Some(selected_local) = self.pick_common_profile() {
                        self.pending_profile = Some(selected_local);
                        if let Ok(sel_payload) = encode_selected_profile_payload(&selected_local) {
                            let select = self.build_control_frame(FrameKind::Select, &sel_payload);
                            let _ = out.push(LinkAction::SendFrame(select));
                        }
                    }
                }
            }

            (LinkState::Negotiating, LinkEvent::RxFrame(frame))
                if frame.header.kind == FrameKind::Select =>
            {
                if let Ok(selected) = decode_selected_profile_payload(frame.payload.as_slice()) {
                    if let Some(local_selected) = self.find_local_match(selected) {
                        self.pending_profile = Some(local_selected);
                        if let Ok(commit_payload) = encode_selected_profile_payload(&local_selected)
                        {
                            let commit =
                                self.build_control_frame(FrameKind::Commit, &commit_payload);
                            let _ = out.push(LinkAction::SendFrame(commit));
                        }
                    }
                }
            }

            (LinkState::Negotiating, LinkEvent::RxFrame(frame))
                if frame.header.kind == FrameKind::Commit =>
            {
                if let Ok(committed) = decode_selected_profile_payload(frame.payload.as_slice()) {
                    if let Some(local_committed) = self.find_local_match(committed) {
                        self.active_profile = local_committed;
                        self.state = LinkState::Data;
                        let _ = out.push(LinkAction::EnterProfile {
                            profile: self.active_profile,
                        });
                    }
                }
            }

            (LinkState::Data, LinkEvent::TxRequest { payload }) => {
                let frame = self.build_data_frame(payload.as_slice());
                self.pending_tx = Some(frame.clone());
                self.retries = 0;
                let _ = out.push(LinkAction::SendFrame(frame));
            }

            (LinkState::Data, LinkEvent::RxFrame(frame))
                if frame.header.kind == FrameKind::Data =>
            {
                let _ = out.push(LinkAction::SendAck {
                    sequence: frame.header.sequence,
                });
            }

            (LinkState::Data, LinkEvent::RxFrame(frame)) if frame.header.kind == FrameKind::Ack => {
                if frame.header.sequence == self.sequence {
                    self.pending_tx = None;
                    self.sequence = (self.sequence + 1) & 0x0F;
                    self.retries = 0;
                }
            }

            (LinkState::Data, LinkEvent::Timeout) => {
                if let Some(frame) = self.pending_tx.clone() {
                    if self.retries < self.profile.max_retransmissions {
                        self.retries += 1;
                        let _ = out.push(LinkAction::Retransmit(frame));
                    } else {
                        self.state = LinkState::Fallback;
                        self.capability_sent = false;
                        let probe = self.build_control_frame(FrameKind::Probe, &[]);
                        let _ = out.push(LinkAction::SendFrame(probe));
                    }
                } else {
                    self.state = LinkState::Reacquire;
                }
            }

            _ => {}
        }

        out
    }

    fn pick_common_profile(&self) -> Option<NegotiationProfile> {
        let mut selected: Option<NegotiationProfile> = None;
        for local in &self.supported_profiles {
            if self
                .remote_profiles
                .iter()
                .any(|r| profiles_compatible(*local, *r))
            {
                if let Some(cur) = selected {
                    if profile_score(*local) > profile_score(cur) {
                        selected = Some(*local);
                    }
                } else {
                    selected = Some(*local);
                }
            }
        }
        selected
    }

    fn find_local_match(&self, remote_selected: NegotiationProfile) -> Option<NegotiationProfile> {
        self.supported_profiles
            .iter()
            .copied()
            .find(|p| profiles_compatible(*p, remote_selected))
    }

    fn build_control_frame(&self, kind: FrameKind, payload: &[u8]) -> Frame {
        Frame::new(
            FrameHeader {
                version: 1,
                profile_id: 0,
                sequence: self.sequence,
                fec_id: 0,
                kind,
            },
            payload,
        )
        .expect("control frame payload must fit")
    }

    fn build_data_frame(&self, payload: &[u8]) -> Frame {
        Frame::new(
            FrameHeader {
                version: 1,
                profile_id: self.active_profile.profile_id,
                sequence: self.sequence,
                fec_id: 1,
                kind: FrameKind::Data,
            },
            payload,
        )
        .expect("data frame payload must fit")
    }
}

fn profile_score(profile: NegotiationProfile) -> u32 {
    let phy = u32::try_from(profile.modulation.bits_per_symbol()).unwrap_or(1)
        * profile.symbol_rate.as_hz();
    phy.saturating_mul(1_000_000)
        .saturating_add(profile.voice_bitrate_bps)
}

fn profiles_compatible(a: NegotiationProfile, b: NegotiationProfile) -> bool {
    a.modulation == b.modulation
        && a.symbol_rate == b.symbol_rate
        && a.fec_scheme == b.fec_scheme
        && a.code_n == b.code_n
        && a.code_k == b.code_k
        && a.interleaver_depth == b.interleaver_depth
        && a.max_iterations == b.max_iterations
        && a.voice_bitrate_bps == b.voice_bitrate_bps
        && a.tone_center_hz == b.tone_center_hz
        && a.tone_spacing_hz == b.tone_spacing_hz
        && a.rx_bandwidth_hz == b.rx_bandwidth_hz
        && a.afc_range_hz == b.afc_range_hz
        && a.preamble_symbols == b.preamble_symbols
        && a.sync_word == b.sync_word
        && a.noise_adapt_k_q8 == b.noise_adapt_k_q8
        && a.continuous_noise_mode == b.continuous_noise_mode
}

fn default_profiles() -> [NegotiationProfile; 1] {
    [NegotiationProfile {
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
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_payload(slice: &[u8]) -> Vec<u8, MAX_PAYLOAD_BYTES> {
        let mut out = Vec::<u8, MAX_PAYLOAD_BYTES>::new();
        out.extend_from_slice(slice).expect("fits");
        out
    }

    #[test]
    fn handshake_to_data_with_select_commit() {
        let mut a = LinkMachine::new(LinkProfile::default());
        let mut b = LinkMachine::new(LinkProfile::default());

        let a_out = a.link_tick(LinkEvent::PttReady);
        assert_eq!(a.state(), LinkState::Probe);
        let probe = match &a_out[0] {
            LinkAction::SendFrame(f) => f.clone(),
            _ => panic!("expected probe frame"),
        };

        let b_out = b.link_tick(LinkEvent::RxFrame(probe));
        assert_eq!(b.state(), LinkState::Negotiating);
        let capability = match &b_out[0] {
            LinkAction::SendFrame(f) => f.clone(),
            _ => panic!("expected capability frame"),
        };

        let a_out = a.link_tick(LinkEvent::RxFrame(capability));
        assert_eq!(a.state(), LinkState::Negotiating);
        let select = a_out
            .iter()
            .find_map(|x| match x {
                LinkAction::SendFrame(f) if f.header.kind == FrameKind::Select => Some(f.clone()),
                _ => None,
            })
            .expect("select expected");

        let b_out = b.link_tick(LinkEvent::RxFrame(select));
        let commit = b_out
            .iter()
            .find_map(|x| match x {
                LinkAction::SendFrame(f) if f.header.kind == FrameKind::Commit => Some(f.clone()),
                _ => None,
            })
            .expect("commit expected");

        let a_out = a.link_tick(LinkEvent::RxFrame(commit));
        assert_eq!(a.state(), LinkState::Data);
        assert!(
            a_out
                .iter()
                .any(|x| matches!(x, LinkAction::EnterProfile { .. }))
        );
    }

    #[test]
    fn data_retransmit_then_fallback() {
        let mut p = LinkProfile::default();
        p.max_retransmissions = 1;
        let mut link = LinkMachine::new(p);
        link.state = LinkState::Data;

        let out = link.link_tick(LinkEvent::TxRequest {
            payload: as_payload(&[1, 2, 3]),
        });
        assert_eq!(out.len(), 1);

        let out = link.link_tick(LinkEvent::Timeout);
        assert_eq!(out.len(), 1);
        assert_eq!(link.state(), LinkState::Data);

        let out = link.link_tick(LinkEvent::Timeout);
        assert_eq!(out.len(), 1);
        assert_eq!(link.state(), LinkState::Fallback);
    }

    #[test]
    fn negotiation_rejects_voice_bitrate_mismatch() {
        let mut a = LinkMachine::new(LinkProfile::default());
        let remote_profiles = [NegotiationProfile {
            voice_bitrate_bps: DEFAULT_VOICE_BITRATE_BPS + 1_000,
            ..default_profiles()[0]
        }];
        let mut b =
            LinkMachine::new_with_supported_profiles(LinkProfile::default(), &remote_profiles);

        let a_probe = a.link_tick(LinkEvent::PttReady);
        let probe = match &a_probe[0] {
            LinkAction::SendFrame(f) => f.clone(),
            _ => panic!("expected probe"),
        };
        let b_cap = b.link_tick(LinkEvent::RxFrame(probe));
        let capability = b_cap
            .iter()
            .find_map(|x| match x {
                LinkAction::SendFrame(f) if f.header.kind == FrameKind::Capability => {
                    Some(f.clone())
                }
                _ => None,
            })
            .expect("capability expected");

        let a_out = a.link_tick(LinkEvent::RxFrame(capability));
        assert!(
            !a_out.iter().any(
                |x| matches!(x, LinkAction::SendFrame(f) if f.header.kind == FrameKind::Select)
            ),
            "select must not be emitted when voice bitrate differs"
        );
    }
}
