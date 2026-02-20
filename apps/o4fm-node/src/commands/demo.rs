use heapless::Vec as HVec;
use o4fm_core::{Frame, LinkProfile, MAX_PAYLOAD_BYTES};
use o4fm_link::{LinkAction, LinkEvent, LinkMachine, LinkState};

use crate::cli::DemoArgs;
use crate::commands::build_supported_profiles;
use crate::o4fm_payload::{capture_send_frames, emit, write_demo_wave};

pub(crate) fn run_demo(args: &DemoArgs) -> Result<(), Box<dyn std::error::Error>> {
    let supported = build_supported_profiles(args.voice_bitrate);
    let mut a = LinkMachine::new_with_supported_profiles(LinkProfile::default(), &supported);
    let mut b = LinkMachine::new_with_supported_profiles(LinkProfile::default(), &supported);
    let mut captured_frames = Vec::<Frame>::new();
    let wave_out = args.demo_wave_out.as_str();

    println!("o4fm-node demo (dual-peer negotiation)");
    let mut a_actions = a.link_tick(LinkEvent::PttReady);
    log_enter_profile("peer-a", &a_actions);
    capture_send_frames(&a_actions, &mut captured_frames);
    emit(a_actions.clone());
    let mut b_actions = HVec::<LinkAction, 8>::new();

    for _ in 0..8 {
        let mut progressed = false;

        let frames_to_b = extract_frames(&a_actions);
        a_actions.clear();
        for frame in frames_to_b {
            let out = b.link_tick(LinkEvent::RxFrame(frame));
            if !out.is_empty() {
                progressed = true;
                log_enter_profile("peer-b", &out);
                capture_send_frames(&out, &mut captured_frames);
                emit(out.clone());
                for action in out {
                    let _ = b_actions.push(action);
                }
            }
        }

        let frames_to_a = extract_frames(&b_actions);
        b_actions.clear();
        for frame in frames_to_a {
            let out = a.link_tick(LinkEvent::RxFrame(frame));
            if !out.is_empty() {
                progressed = true;
                log_enter_profile("peer-a", &out);
                capture_send_frames(&out, &mut captured_frames);
                emit(out.clone());
                for action in out {
                    let _ = a_actions.push(action);
                }
            }
        }

        if !progressed {
            break;
        }
        if a.state() == LinkState::Data && b.state() == LinkState::Data {
            break;
        }
    }

    println!(
        "  peer-a state={:?} active_profile={:?}",
        a.state(),
        a.active_profile()
    );
    println!(
        "  peer-b state={:?} active_profile={:?}",
        b.state(),
        b.active_profile()
    );

    let mut payload = HVec::<u8, MAX_PAYLOAD_BYTES>::new();
    payload
        .extend_from_slice(b"hello o4fm")
        .expect("payload fits");
    let tx_actions = a.link_tick(LinkEvent::TxRequest { payload });
    log_enter_profile("peer-a", &tx_actions);
    capture_send_frames(&tx_actions, &mut captured_frames);
    emit(tx_actions);

    write_demo_wave(&captured_frames, wave_out)?;
    println!("  demo wave written: {wave_out}");
    Ok(())
}

fn log_enter_profile<const N: usize>(peer: &str, actions: &HVec<LinkAction, N>) {
    for action in actions {
        if let LinkAction::EnterProfile { profile } = action {
            println!(
                "  {peer} committed profile: id={} mod={}FSK sym={} fec=LDPC({}/{}) interleave={} iters={} voice={}bps",
                profile.profile_id,
                profile.modulation.order(),
                profile.symbol_rate.as_hz(),
                profile.code_n,
                profile.code_k,
                profile.interleaver_depth,
                profile.max_iterations,
                profile.voice_bitrate_bps
            );
        }
    }
}

fn extract_frames(actions: &HVec<LinkAction, 8>) -> Vec<Frame> {
    actions
        .iter()
        .filter_map(|action| match action {
            LinkAction::SendFrame(frame) => Some(frame.clone()),
            _ => None,
        })
        .collect()
}
