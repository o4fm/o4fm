use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_samples: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            channels: 1,
            frame_samples: 480,
        }
    }
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("backend unavailable")]
    BackendUnavailable,
    #[error("device I/O failure")]
    Io,
    #[error("unsupported sample format")]
    UnsupportedSampleFormat,
    #[error("file I/O failure")]
    FileIo,
    #[error("invalid device id")]
    InvalidDeviceId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub supports_input: bool,
    pub supports_output: bool,
}

pub trait AudioIo {
    /// Read one frame of mono/stereo PCM samples into `out`.
    ///
    /// # Errors
    /// Returns an error if the backend cannot provide samples.
    fn read_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError>;
    /// Write one frame of PCM samples.
    ///
    /// # Errors
    /// Returns an error if the backend cannot accept samples.
    fn write_frame(&mut self, samples: &[i16]) -> Result<(), AudioError>;
}

#[derive(Default)]
pub struct MockLoopbackAudio {
    queue: VecDeque<i16>,
}

impl AudioIo for MockLoopbackAudio {
    fn read_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        let mut copied = 0usize;
        for sample in out.iter_mut() {
            if let Some(s) = self.queue.pop_front() {
                *sample = s;
                copied += 1;
            } else {
                break;
            }
        }
        Ok(copied)
    }

    fn write_frame(&mut self, samples: &[i16]) -> Result<(), AudioError> {
        self.queue.extend(samples.iter().copied());
        Ok(())
    }
}

pub struct CpalRealtimeAudio {
    rx_queue: Arc<Mutex<VecDeque<i16>>>,
    tx_queue: Arc<Mutex<VecDeque<i16>>>,
    _input_stream: Stream,
    _output_stream: Stream,
}

impl CpalRealtimeAudio {
    /// Open realtime audio using default input/output devices.
    ///
    /// # Errors
    /// Returns an error if the CPAL backend or devices are unavailable.
    pub fn new_default(cfg: AudioConfig) -> Result<Self, AudioError> {
        Self::new_with_device_ids(cfg, None, None)
    }

    /// Open realtime audio using optional explicit device IDs.
    ///
    /// # Errors
    /// Returns an error for invalid IDs, missing devices, unsupported formats,
    /// or stream creation failures.
    pub fn new_with_device_ids(
        cfg: AudioConfig,
        input_device_id: Option<&str>,
        output_device_id: Option<&str>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let input_device = if let Some(id) = input_device_id {
            let parsed = cpal::DeviceId::from_str(id).map_err(|_| AudioError::InvalidDeviceId)?;
            let dev = host
                .device_by_id(&parsed)
                .ok_or(AudioError::BackendUnavailable)?;
            if !dev.supports_input() {
                return Err(AudioError::BackendUnavailable);
            }
            dev
        } else {
            host.default_input_device()
                .ok_or(AudioError::BackendUnavailable)?
        };
        let output_device = if let Some(id) = output_device_id {
            let parsed = cpal::DeviceId::from_str(id).map_err(|_| AudioError::InvalidDeviceId)?;
            let dev = host
                .device_by_id(&parsed)
                .ok_or(AudioError::BackendUnavailable)?;
            if !dev.supports_output() {
                return Err(AudioError::BackendUnavailable);
            }
            dev
        } else {
            host.default_output_device()
                .ok_or(AudioError::BackendUnavailable)?
        };

        let input_supported = input_device
            .default_input_config()
            .map_err(|_| AudioError::BackendUnavailable)?;
        let output_supported = output_device
            .default_output_config()
            .map_err(|_| AudioError::BackendUnavailable)?;

        let input_config = StreamConfig {
            channels: cfg.channels.max(1),
            sample_rate: cfg.sample_rate_hz,
            buffer_size: cpal::BufferSize::Default,
        };
        let output_config = StreamConfig {
            channels: cfg.channels.max(1),
            sample_rate: cfg.sample_rate_hz,
            buffer_size: cpal::BufferSize::Default,
        };

        let rx_queue = Arc::new(Mutex::new(VecDeque::with_capacity(cfg.frame_samples * 8)));
        let tx_queue = Arc::new(Mutex::new(VecDeque::with_capacity(cfg.frame_samples * 8)));

        let input_stream = match input_supported.sample_format() {
            SampleFormat::I16 => {
                build_input_stream::<i16>(&input_device, &input_config, Arc::clone(&rx_queue))?
            }
            SampleFormat::U16 => {
                build_input_stream::<u16>(&input_device, &input_config, Arc::clone(&rx_queue))?
            }
            SampleFormat::F32 => {
                build_input_stream::<f32>(&input_device, &input_config, Arc::clone(&rx_queue))?
            }
            _ => return Err(AudioError::UnsupportedSampleFormat),
        };

        let output_stream = match output_supported.sample_format() {
            SampleFormat::I16 => {
                build_output_stream::<i16>(&output_device, &output_config, Arc::clone(&tx_queue))?
            }
            SampleFormat::U16 => {
                build_output_stream::<u16>(&output_device, &output_config, Arc::clone(&tx_queue))?
            }
            SampleFormat::F32 => {
                build_output_stream::<f32>(&output_device, &output_config, Arc::clone(&tx_queue))?
            }
            _ => return Err(AudioError::UnsupportedSampleFormat),
        };

        input_stream.play().map_err(|_| AudioError::Io)?;
        output_stream.play().map_err(|_| AudioError::Io)?;

        Ok(Self {
            rx_queue,
            tx_queue,
            _input_stream: input_stream,
            _output_stream: output_stream,
        })
    }
}

impl AudioIo for CpalRealtimeAudio {
    fn read_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        let mut copied = 0usize;
        let mut q = self.rx_queue.lock().map_err(|_| AudioError::Io)?;
        for sample in out.iter_mut() {
            if let Some(s) = q.pop_front() {
                *sample = s;
                copied += 1;
            } else {
                break;
            }
        }
        Ok(copied)
    }

    fn write_frame(&mut self, samples: &[i16]) -> Result<(), AudioError> {
        let mut q = self.tx_queue.lock().map_err(|_| AudioError::Io)?;
        q.extend(samples.iter().copied());
        // Keep bounded queue to avoid unbounded growth when producer is faster.
        while q.len() > 96_000 {
            q.pop_front();
        }
        Ok(())
    }
}

pub struct WavFileAudio {
    input_samples: Vec<i16>,
    input_cursor: usize,
    output_writer: hound::WavWriter<BufWriter<File>>,
}

impl WavFileAudio {
    /// Open WAV-backed audio I/O for offline processing.
    ///
    /// # Errors
    /// Returns an error if input/output WAV files cannot be opened or parsed.
    pub fn open<PIn: AsRef<Path>, POut: AsRef<Path>>(
        input_path: PIn,
        output_path: POut,
        sample_rate_hz: u32,
    ) -> Result<Self, AudioError> {
        let mut reader = hound::WavReader::open(input_path).map_err(|_| AudioError::FileIo)?;
        let spec = reader.spec();

        let input_samples = match spec.sample_format {
            hound::SampleFormat::Float => {
                let mut v = Vec::new();
                for sample in reader.samples::<f32>() {
                    let s = sample.map_err(|_| AudioError::FileIo)?;
                    let normalized = s.clamp(-1.0, 1.0);
                    v.push(i16::from_sample(normalized));
                }
                v
            }
            hound::SampleFormat::Int => {
                let bps = spec.bits_per_sample;
                let mut v = Vec::new();
                if bps <= 16 {
                    for sample in reader.samples::<i16>() {
                        v.push(sample.map_err(|_| AudioError::FileIo)?);
                    }
                } else {
                    for sample in reader.samples::<i32>() {
                        let s = sample.map_err(|_| AudioError::FileIo)?;
                        v.push(scale_int_sample_to_i16(s, bps));
                    }
                }
                v
            }
        };

        let out_spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let output_writer =
            hound::WavWriter::create(output_path, out_spec).map_err(|_| AudioError::FileIo)?;

        Ok(Self {
            input_samples,
            input_cursor: 0,
            output_writer,
        })
    }

    /// Finalize and flush output WAV file.
    ///
    /// # Errors
    /// Returns an error if finalization of the WAV writer fails.
    pub fn finalize(self) -> Result<(), AudioError> {
        self.output_writer
            .finalize()
            .map_err(|_| AudioError::FileIo)
    }
}

impl AudioIo for WavFileAudio {
    fn read_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        let remaining = self
            .input_samples
            .len()
            .saturating_sub(self.input_cursor)
            .min(out.len());

        if remaining == 0 {
            return Ok(0);
        }

        let end = self.input_cursor + remaining;
        out[..remaining].copy_from_slice(&self.input_samples[self.input_cursor..end]);
        self.input_cursor = end;
        Ok(remaining)
    }

    fn write_frame(&mut self, samples: &[i16]) -> Result<(), AudioError> {
        for &sample in samples {
            self.output_writer
                .write_sample(sample)
                .map_err(|_| AudioError::FileIo)?;
        }
        Ok(())
    }
}

/// List available audio device display names.
///
/// # Errors
/// Returns an error if audio backend device enumeration fails.
pub fn list_devices() -> Result<Vec<String>, AudioError> {
    Ok(list_device_infos()?.into_iter().map(|d| d.name).collect())
}

/// List available audio devices with stable IDs and capabilities.
///
/// # Errors
/// Returns an error if audio backend device enumeration fails.
pub fn list_device_infos() -> Result<Vec<AudioDeviceInfo>, AudioError> {
    let host = cpal::default_host();
    let devices = host.devices().map_err(|_| AudioError::BackendUnavailable)?;
    let mut out = Vec::new();
    for device in devices {
        let name = device.description().map_or_else(
            |_| "unknown-device".to_owned(),
            |desc| desc.name().to_owned(),
        );
        let id = device
            .id()
            .map_or_else(|_| "unknown-id".to_owned(), |id| id.to_string());
        out.push(AudioDeviceInfo {
            id,
            name,
            supports_input: device.supports_input(),
            supports_output: device.supports_output(),
        });
    }
    Ok(out)
}

fn scale_int_sample_to_i16(sample: i32, bits_per_sample: u16) -> i16 {
    let shift = u32::from(bits_per_sample.saturating_sub(16));
    let shifted = if shift == 0 { sample } else { sample >> shift };
    let clamped = shifted.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    i16::try_from(clamped).expect("clamped to i16 range")
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    rx_queue: Arc<Mutex<VecDeque<i16>>>,
) -> Result<Stream, AudioError>
where
    T: cpal::Sample + SizedSample,
    i16: FromSample<T>,
{
    let err_fn = |_err: cpal::StreamError| {};

    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if let Ok(mut q) = rx_queue.lock() {
                    for &s in data {
                        q.push_back(i16::from_sample(s));
                        if q.len() > 96_000 {
                            q.pop_front();
                        }
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|_| AudioError::Io)
}

fn build_output_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    tx_queue: Arc<Mutex<VecDeque<i16>>>,
) -> Result<Stream, AudioError>
where
    T: cpal::Sample + SizedSample + FromSample<i16>,
{
    let err_fn = |_err: cpal::StreamError| {};

    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                if let Ok(mut q) = tx_queue.lock() {
                    for slot in data {
                        let sample = q.pop_front().unwrap_or(0);
                        *slot = T::from_sample(sample);
                    }
                } else {
                    for slot in data {
                        *slot = T::from_sample(0);
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|_| AudioError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_file_io_round_trip() {
        let tmp = std::env::temp_dir();
        let in_path = tmp.join("o4fm_audio_in.wav");
        let out_path = tmp.join("o4fm_audio_out.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&in_path, spec).expect("create input wav");
            for i in 0..128_i16 {
                writer.write_sample(i).expect("write sample");
            }
            writer.finalize().expect("finalize input wav");
        }

        let mut io = WavFileAudio::open(&in_path, &out_path, 48_000).expect("open wav io");
        let mut buf = [0_i16; 64];
        let n = io.read_frame(&mut buf).expect("read frame");
        assert_eq!(n, 64);
        io.write_frame(&buf[..n]).expect("write frame");
        io.finalize().expect("finalize");

        let _ = std::fs::remove_file(in_path);
        let _ = std::fs::remove_file(out_path);
    }
}
