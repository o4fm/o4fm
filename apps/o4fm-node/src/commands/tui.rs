use std::io;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};

use crate::cli::{
    Bin2WavArgs, CpalArgs, DemoArgs, LpMode, Pipeline, TuiArgs, VoiceDspArg, VoiceRxArgs,
    VoiceTxArgs, Wav2BinArgs,
};
use crate::commands::build_supported_profiles;
use crate::commands::{bin, cpal, demo, voice};

#[derive(Clone, Copy)]
enum UiMode {
    Demo,
    CpalDigital,
    CpalPassthrough,
    BinToWav,
    WavToBin,
    VoiceTx,
    VoiceRx,
}

impl UiMode {
    fn name(self) -> &'static str {
        match self {
            Self::Demo => "demo",
            Self::CpalDigital => "cpal:digital",
            Self::CpalPassthrough => "cpal:passthrough",
            Self::BinToWav => "bin-to-wav",
            Self::WavToBin => "wav-to-bin",
            Self::VoiceTx => "voice-tx",
            Self::VoiceRx => "voice-rx",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Demo => Self::CpalDigital,
            Self::CpalDigital => Self::CpalPassthrough,
            Self::CpalPassthrough => Self::BinToWav,
            Self::BinToWav => Self::WavToBin,
            Self::WavToBin => Self::VoiceTx,
            Self::VoiceTx => Self::VoiceRx,
            Self::VoiceRx => Self::Demo,
        }
    }
}

#[derive(Clone, Copy)]
enum Field {
    InputPath,
    OutputPath,
    SampleRate,
    FrameSamples,
    Seconds,
    VoiceBitrate,
    ZeroThreshold,
    InterFrameSilenceMs,
    ProfileId,
}

impl Field {
    const ALL: [Field; 9] = [
        Field::InputPath,
        Field::OutputPath,
        Field::SampleRate,
        Field::FrameSamples,
        Field::Seconds,
        Field::VoiceBitrate,
        Field::ZeroThreshold,
        Field::InterFrameSilenceMs,
        Field::ProfileId,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::InputPath => "input_path",
            Self::OutputPath => "output_path",
            Self::SampleRate => "sample_rate",
            Self::FrameSamples => "frame_samples",
            Self::Seconds => "seconds",
            Self::VoiceBitrate => "voice_bitrate",
            Self::ZeroThreshold => "zero_threshold",
            Self::InterFrameSilenceMs => "inter_frame_silence_ms",
            Self::ProfileId => "profile_id",
        }
    }
}

struct UiState {
    mode: UiMode,
    selected: usize,
    sample_rate: u32,
    frame_samples: usize,
    seconds: u64,
    voice_bitrate: u32,
    zero_threshold: i16,
    inter_frame_silence_ms: u32,
    profile_id: u8,
    last_apply_at: Option<Instant>,
    status: String,
    running: bool,
    editing: bool,
    input_path: String,
    output_path: String,
}

impl UiState {
    fn new() -> Self {
        Self {
            mode: UiMode::Demo,
            selected: 0,
            sample_rate: 48_000,
            frame_samples: 480,
            seconds: 10,
            voice_bitrate: 7_000,
            zero_threshold: 2,
            inter_frame_silence_ms: 2,
            profile_id: 0,
            last_apply_at: None,
            status: "Ready".to_string(),
            running: false,
            editing: false,
            input_path: "input.wav".to_string(),
            output_path: "output.wav".to_string(),
        }
    }

    fn selected_field(&self) -> Field {
        Field::ALL[self.selected]
    }

    fn apply(&mut self) {
        self.last_apply_at = Some(Instant::now());
        let profiles = build_supported_profiles(self.voice_bitrate);
        let has_selected = profiles.iter().any(|p| p.profile_id == self.profile_id);
        self.status = if has_selected {
            format!(
                "Applied mode={} profile_id={} ({} profile(s))",
                self.mode.name(),
                self.profile_id,
                profiles.len()
            )
        } else {
            format!(
                "Applied mode={} but profile_id={} is invalid for current profile set",
                self.mode.name(),
                self.profile_id
            )
        };
    }

    fn nudge_selected(&mut self, delta: i32) {
        match self.selected_field() {
            Field::InputPath | Field::OutputPath => {}
            Field::SampleRate => {
                self.sample_rate = nudge_u32(self.sample_rate, delta, 8_000, 96_000, 1_000);
            }
            Field::FrameSamples => {
                self.frame_samples = nudge_usize(self.frame_samples, delta, 80, 4_800, 80);
            }
            Field::Seconds => {
                self.seconds = nudge_u64(self.seconds, delta, 1, 3_600, 1);
            }
            Field::VoiceBitrate => {
                self.voice_bitrate = nudge_u32(self.voice_bitrate, delta, 500, 510_000, 500);
            }
            Field::ZeroThreshold => {
                self.zero_threshold = nudge_i16(self.zero_threshold, delta, 0, 32767, 1);
            }
            Field::InterFrameSilenceMs => {
                self.inter_frame_silence_ms =
                    nudge_u32(self.inter_frame_silence_ms, delta, 0, 2_000, 1);
            }
            Field::ProfileId => {
                self.profile_id = nudge_u8(self.profile_id, delta, 0, 7, 1);
            }
        }
    }

    fn config_rows(&self) -> Vec<String> {
        vec![
            self.input_path.clone(),
            self.output_path.clone(),
            format!("sample_rate={} Hz", self.sample_rate),
            format!("frame_samples={}", self.frame_samples),
            format!("seconds={}", self.seconds),
            format!("voice_bitrate={} bps", self.voice_bitrate),
            format!("zero_threshold={}", self.zero_threshold),
            format!("inter_frame_silence_ms={}", self.inter_frame_silence_ms),
            format!("profile_id={}", self.profile_id),
        ]
    }

    fn selected_path_mut(&mut self) -> Option<&mut String> {
        match self.selected_field() {
            Field::InputPath => Some(&mut self.input_path),
            Field::OutputPath => Some(&mut self.output_path),
            _ => None,
        }
    }
}

enum WorkerEvent {
    Finished(String),
}

struct WorkerRuntime {
    tx: Sender<WorkerEvent>,
    rx: Receiver<WorkerEvent>,
    handle: Option<JoinHandle<()>>,
}

impl WorkerRuntime {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            handle: None,
        }
    }

    fn start(&mut self, snapshot: RunSnapshot) -> Result<(), String> {
        if self.handle.is_some() {
            return Err("A task is already running".to_string());
        }
        let tx = self.tx.clone();
        self.handle = Some(thread::spawn(move || {
            let msg = match run_snapshot(snapshot) {
                Ok(()) => "Task finished successfully".to_string(),
                Err(e) => format!("Task failed: {e}"),
            };
            let _ = tx.send(WorkerEvent::Finished(msg));
        }));
        Ok(())
    }

    fn poll(&mut self) -> Option<WorkerEvent> {
        self.rx.try_recv().ok()
    }

    fn reap_if_finished(&mut self) {
        if let Some(handle) = &self.handle
            && handle.is_finished()
        {
            let handle = self.handle.take().expect("checked is_some");
            let _ = handle.join();
        }
    }
}

#[derive(Clone)]
struct RunSnapshot {
    mode: UiMode,
    input_path: String,
    output_path: String,
    sample_rate: u32,
    frame_samples: usize,
    seconds: u64,
    voice_bitrate: u32,
    zero_threshold: i16,
    inter_frame_silence_ms: u32,
    profile_id: u8,
}

pub(crate) fn run_tui_mode(args: &TuiArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tick_hz = args.tick_hz.max(1);
    let tick = Duration::from_millis(1000 / tick_hz);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, tick);

    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    tick: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = UiState::new();
    let mut runtime = WorkerRuntime::new();

    loop {
        runtime.reap_if_finished();
        while let Some(evt) = runtime.poll() {
            match evt {
                WorkerEvent::Finished(msg) => {
                    state.running = false;
                    state.status = msg;
                }
            }
        }

        terminal.draw(|f| draw_ui(f, &state))?;

        if event::poll(tick)? {
            let evt = event::read()?;
            if let Event::Key(key) = evt {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('e') => {
                        if matches!(state.selected_field(), Field::InputPath | Field::OutputPath) {
                            state.editing = !state.editing;
                            state.status = if state.editing {
                                "Path edit mode ON".to_string()
                            } else {
                                "Path edit mode OFF".to_string()
                            };
                        }
                    }
                    KeyCode::Tab => {
                        if !state.editing {
                            state.mode = state.mode.next();
                        }
                    }
                    KeyCode::Char('r') => {
                        if !state.editing {
                            state.apply();
                        }
                    }
                    KeyCode::Char('s') => {
                        if !state.editing {
                            let snapshot = RunSnapshot {
                                mode: state.mode,
                                input_path: state.input_path.clone(),
                                output_path: state.output_path.clone(),
                                sample_rate: state.sample_rate,
                                frame_samples: state.frame_samples,
                                seconds: state.seconds,
                                voice_bitrate: state.voice_bitrate,
                                zero_threshold: state.zero_threshold,
                                inter_frame_silence_ms: state.inter_frame_silence_ms,
                                profile_id: state.profile_id,
                            };
                            match runtime.start(snapshot) {
                                Ok(()) => {
                                    state.running = true;
                                    state.status = format!(
                                        "Task started for mode={} (background thread)",
                                        state.mode.name()
                                    );
                                }
                                Err(e) => state.status = e,
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        if state.editing
                            && let Some(path) = state.selected_path_mut()
                        {
                            path.pop();
                        }
                    }
                    KeyCode::Enter | KeyCode::Esc => {
                        if state.editing {
                            state.editing = false;
                            state.status = "Path edit mode OFF".to_string();
                        }
                    }
                    KeyCode::Char(ch) => {
                        if state.editing
                            && let Some(path) = state.selected_path_mut()
                        {
                            path.push(ch);
                        }
                    }
                    KeyCode::Up => {
                        if !state.editing {
                            if state.selected == 0 {
                                state.selected = Field::ALL.len() - 1;
                            } else {
                                state.selected -= 1;
                            }
                        }
                    }
                    KeyCode::Down => {
                        if !state.editing {
                            state.selected = (state.selected + 1) % Field::ALL.len();
                        }
                    }
                    KeyCode::Left => {
                        if !state.editing {
                            state.nudge_selected(-1);
                        }
                    }
                    KeyCode::Right => {
                        if !state.editing {
                            state.nudge_selected(1);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

fn draw_ui(frame: &mut Frame<'_>, state: &UiState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(7),
        ])
        .split(frame.area());

    let run_state = if state.running { "RUNNING" } else { "IDLE" };
    let header = Paragraph::new(format!(
        "O4FM Node TUI | mode={} | state={} | edit={} | press q to quit",
        state.mode.name(),
        run_state,
        state.editing
    ))
    .block(Block::default().borders(Borders::ALL).title("Status"));
    frame.render_widget(header, layout[0]);

    let help = Paragraph::new("Tab: switch mode | Up/Down: select field | Left/Right: adjust | e: edit path | Enter/Esc: end edit | r: apply | s: start")
        .block(Block::default().borders(Borders::ALL).title("Keys"));
    frame.render_widget(help, layout[1]);

    let items: Vec<ListItem<'_>> = state
        .config_rows()
        .into_iter()
        .enumerate()
        .map(|(idx, row)| {
            let marker = if idx == state.selected { "> " } else { "  " };
            ListItem::new(format!("{marker}{}: {row}", Field::ALL[idx].name()))
        })
        .collect();
    let config = List::new(items).block(Block::default().borders(Borders::ALL).title("Config"));
    frame.render_widget(config, layout[2]);

    let profiles = build_supported_profiles(state.voice_bitrate);
    let mut lines = Vec::<String>::new();
    lines.push(format!("status: {}", state.status));
    lines.push(format!("supported_profiles: {}", profiles.len()));
    for p in &profiles {
        lines.push(format!(
            "id={} {}FSK sym={} tone_center={} tone_spacing={} afc={} voice={}",
            p.profile_id,
            p.modulation.order(),
            p.symbol_rate.as_hz(),
            p.tone_center_hz,
            p.tone_spacing_hz,
            p.afc_range_hz,
            p.voice_bitrate_bps
        ));
    }

    let summary = Paragraph::new(lines.join("\n"))
        .block(Block::default().borders(Borders::ALL).title("Profiles"));
    frame.render_widget(summary, layout[3]);
}

fn run_snapshot(snapshot: RunSnapshot) -> Result<(), String> {
    match snapshot.mode {
        UiMode::Demo => {
            let args = DemoArgs {
                demo_wave_out: "target/o4fm-node-demo-tui.wav".to_string(),
                voice_bitrate: snapshot.voice_bitrate,
            };
            demo::run_demo(&args).map_err(|e| e.to_string())
        }
        UiMode::CpalDigital | UiMode::CpalPassthrough => {
            let args = CpalArgs {
                pipeline: match snapshot.mode {
                    UiMode::CpalPassthrough => Pipeline::Passthrough,
                    _ => Pipeline::Digital,
                },
                sample_rate: snapshot.sample_rate,
                channels: 1,
                frame_samples: snapshot.frame_samples,
                seconds: snapshot.seconds,
                voice_bitrate: snapshot.voice_bitrate,
                input_device_id: None,
                output_device_id: None,
            };
            cpal::run_cpal_mode(&args).map_err(|e| e.to_string())
        }
        UiMode::BinToWav => {
            let args = Bin2WavArgs {
                in_bin: snapshot.input_path,
                out_wav: snapshot.output_path,
                inter_frame_silence_ms: snapshot.inter_frame_silence_ms,
                callsign: "NOCALL".to_string(),
                lp_flags: 0,
                lp_mode: LpMode::Text,
                voice_bitrate: snapshot.voice_bitrate,
                profile_id: snapshot.profile_id,
            };
            bin::run_bin_to_wav_mode(&args).map_err(|e| e.to_string())
        }
        UiMode::WavToBin => {
            let args = Wav2BinArgs {
                in_wav: snapshot.input_path,
                out_bin: snapshot.output_path,
                zero_threshold: snapshot.zero_threshold,
                voice_bitrate: snapshot.voice_bitrate,
                profile_id: Some(snapshot.profile_id),
            };
            bin::run_wav_to_bin_mode(&args).map_err(|e| e.to_string())
        }
        UiMode::VoiceTx => {
            let args = VoiceTxArgs {
                in_wav: snapshot.input_path,
                out_wav: snapshot.output_path,
                inter_frame_silence_ms: snapshot.inter_frame_silence_ms,
                callsign: "NOCALL".to_string(),
                lp_flags: 0,
                opus_bitrate: i32::try_from(snapshot.voice_bitrate).unwrap_or(7000),
                opus_frame_ms: 20,
                voice_dsp: VoiceDspArg::Basic,
            };
            voice::run_voice_tx_mode(&args).map_err(|e| e.to_string())
        }
        UiMode::VoiceRx => {
            let args = VoiceRxArgs {
                in_wav: snapshot.input_path,
                out_wav: snapshot.output_path,
                zero_threshold: snapshot.zero_threshold,
                opus_bitrate: i32::try_from(snapshot.voice_bitrate).unwrap_or(7000),
                voice_dsp: VoiceDspArg::Basic,
            };
            voice::run_voice_rx_mode(&args).map_err(|e| e.to_string())
        }
    }
}

fn nudge_u32(value: u32, delta: i32, min: u32, max: u32, step: u32) -> u32 {
    let raw = if delta > 0 {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    raw.clamp(min, max)
}

fn nudge_u64(value: u64, delta: i32, min: u64, max: u64, step: u64) -> u64 {
    let raw = if delta > 0 {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    raw.clamp(min, max)
}

fn nudge_usize(value: usize, delta: i32, min: usize, max: usize, step: usize) -> usize {
    let raw = if delta > 0 {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    raw.clamp(min, max)
}

fn nudge_i16(value: i16, delta: i32, min: i16, max: i16, step: i16) -> i16 {
    let raw = if delta > 0 {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    raw.clamp(min, max)
}

fn nudge_u8(value: u8, delta: i32, min: u8, max: u8, step: u8) -> u8 {
    let raw = if delta > 0 {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    raw.clamp(min, max)
}
