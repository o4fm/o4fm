use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadioBusy {
    Idle,
    Busy,
}

impl Default for RadioBusy {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Error)]
pub enum RadioIoError {
    #[error("not connected")]
    NotConnected,
    #[error("I/O failure")]
    Io,
}

pub trait RadioIo {
    fn set_ptt(&mut self, enabled: bool) -> Result<(), RadioIoError>;
    fn read_cor(&mut self) -> Result<RadioBusy, RadioIoError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockRadioIo {
    ptt: bool,
    cor: RadioBusy,
}

impl MockRadioIo {
    #[must_use]
    pub fn ptt(&self) -> bool {
        self.ptt
    }

    pub fn set_mock_cor(&mut self, cor: RadioBusy) {
        self.cor = cor;
    }
}

impl RadioIo for MockRadioIo {
    fn set_ptt(&mut self, enabled: bool) -> Result<(), RadioIoError> {
        self.ptt = enabled;
        Ok(())
    }

    fn read_cor(&mut self) -> Result<RadioBusy, RadioIoError> {
        Ok(self.cor)
    }
}
