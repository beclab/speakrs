use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::Session;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready, with_execution_mode};
#[cfg(feature = "coreml")]
mod native;
#[cfg(feature = "coreml")]
mod parallel;
mod run;
mod tensor;

/// Errors that can occur during segmentation inference
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SegmentationError {
    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Streaming channel was closed before all windows were sent
    #[error("receiver disconnected")]
    Disconnected(#[from] crossbeam_channel::SendError<Array2<f32>>),
    /// Internal segmentation invariant was violated
    #[error("{context}: {message}")]
    Invariant {
        /// Which step failed
        context: &'static str,
        /// Invariant failure details
        message: String,
    },
    /// Model output was missing or had an unexpected shape
    #[error("{context}: {message}")]
    MalformedOutput {
        /// Which output extraction step failed
        context: &'static str,
        /// Output validation details
        message: String,
    },
    /// Background worker panicked
    #[error("{worker} thread panicked")]
    WorkerPanic {
        /// Worker or thread name
        worker: String,
    },
}

// seg models exported with EnumeratedShapes for batch 1-32 and b64
const PRIMARY_BATCH_SIZE: usize = 32;
#[cfg(feature = "coreml")]
const LARGE_BATCH_SIZE: usize = 64;

/// Sliding-window segmentation model (pyannote segmentation-3.0)
pub struct SegmentationModel {
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    native_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_large_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    cached_single_input_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_batch_input_shape: CachedInputShape,
    input_buffer: ndarray::Array3<f32>,
    primary_batch_input_buffer: ndarray::Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

// SAFETY: SegmentationModel is only used from one thread at a time via &mut self
// SAFETY: the non-Send fields contain Objective-C objects that are only moved, not shared
// SAFETY: SharedCoreMlModel is already Send + Sync
#[cfg(feature = "coreml")]
unsafe impl Send for SegmentationModel {}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: impl AsRef<Path>, step_duration: f32) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, step_duration, ExecutionMode::Cpu)
    }

    /// Load a segmentation-3.0 ONNX model with the requested execution mode
    pub fn with_mode(
        model_path: impl AsRef<Path>,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ModelLoadError> {
        mode.validate()?;
        ensure_ort_ready()?;

        let model_path = model_path.as_ref();
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        #[cfg(feature = "coreml")]
        if matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            Self::validate_native_coreml_assets(model_path, mode)?;
        }

        macro_rules! timed {
            ($expr:expr) => {{
                let start = std::time::Instant::now();
                let value = $expr;
                (value, start.elapsed())
            }};
        }

        let (session, session_elapsed) = timed!(Self::build_session(model_path, mode)?);
        let (primary_batched_session, primary_batched_elapsed) = timed!(
            primary_batched_path(model_path, mode)
                .map(|path| Self::build_session(&path, mode))
                .transpose()?
        );
        #[cfg(feature = "coreml")]
        let (native_session, native_session_elapsed) =
            timed!(Self::load_native_coreml(model_path, mode)?);
        #[cfg(feature = "coreml")]
        let (native_batched_session, native_batched_elapsed) =
            timed!(Self::load_native_coreml_batched(model_path, mode)?);
        #[cfg(feature = "coreml")]
        let (native_large_batched_session, native_large_batched_elapsed) =
            timed!(Self::load_native_coreml_large_batched(model_path, mode)?);

        #[cfg(feature = "coreml")]
        if matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            if native_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_coreml_path(model_path, mode)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
            if native_batched_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_batched_coreml_path(model_path, mode, PRIMARY_BATCH_SIZE)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
            if native_large_batched_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_batched_coreml_path(model_path, mode, LARGE_BATCH_SIZE)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
        }

        #[cfg(feature = "coreml")]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + native_session_elapsed
                + native_batched_elapsed
                + native_large_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                native_single_ms = native_session_elapsed.as_millis(),
                native_b32_ms = native_batched_elapsed.as_millis(),
                native_b64_ms = native_large_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }
        #[cfg(not(feature = "coreml"))]
        {
            let total_ms = (session_elapsed + primary_batched_elapsed).as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }

        Ok(Self {
            mode,
            session,
            primary_batched_session,
            #[cfg(feature = "coreml")]
            native_session,
            #[cfg(feature = "coreml")]
            native_batched_session,
            #[cfg(feature = "coreml")]
            native_large_batched_session,
            #[cfg(feature = "coreml")]
            cached_single_input_shape: CachedInputShape::new("input", &[1, 1, window_samples]),
            #[cfg(feature = "coreml")]
            cached_batch_input_shape: CachedInputShape::new(
                "input",
                &[PRIMARY_BATCH_SIZE, 1, window_samples],
            ),
            input_buffer: ndarray::Array3::zeros((1, 1, window_samples)),
            primary_batch_input_buffer: ndarray::Array3::zeros((
                PRIMARY_BATCH_SIZE,
                1,
                window_samples,
            )),
            window_samples,
            step_samples,
            sample_rate,
        })
    }

    fn build_session(model_path: &Path, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::available_threads().min(6))?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn available_threads() -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Number of audio samples per sliding window
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Number of audio samples the window advances each step
    pub fn step_samples(&self) -> usize {
        self.step_samples
    }

    /// Step size in seconds
    pub fn step_seconds(&self) -> f64 {
        self.step_samples as f64 / self.sample_rate as f64
    }

    /// Execution mode this model was loaded with
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }
}

/// The batched segmentation model this mode should load, if any.
///
/// 🔴 None on OpenVINO, and the decision belongs here rather than only in `required_files`.
/// That list is behind the `online` feature, so a consumer that downloads weights some other
/// way -- which is every consumer that turns default features off -- has the whole repository
/// in its cache and arrives here with the file present. Fixing only the download list would
/// have fixed nothing for them, this crate's own engine included.
///
/// What it avoids: OpenVINO's GPU plugin generates an LSTM kernel referencing
/// OUTPUT1_GET_INDEX and OUTPUT2_GET_INDEX without defining them, the OpenCL compiler rejects
/// the program, and the session fails to build -- taking the pipeline with it. Measured on
/// Arrow Lake-S / OpenVINO 2025.4.1, on models verified bit-identical to their originals on
/// CPU first, the trigger is a static sequence length rather than the batch: batch 1 made
/// static fails the same way, batch 32 with only its batch dimension made dynamic still
/// fails, and batch 32 with a dynamic sequence length compiles and runs. The unbatched model
/// survives because its sample count is dynamic, not because it is smaller.
fn primary_batched_path(model_path: &Path, mode: ExecutionMode) -> Option<PathBuf> {
    batched_model_path(model_path, PRIMARY_BATCH_SIZE)
        .filter(|_| !mode.is_openvino())
        .filter(|path| path.exists())
}

fn batched_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    let path = model_path;
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory holding an empty file named like the batched segmentation model. The
    /// decision under test reads the name and whether the path exists, never the bytes.
    fn models_dir_with_batched(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("speakrs-seg-batched-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(format!("segmentation-3.0-b{PRIMARY_BATCH_SIZE}.onnx")),
            b"",
        )
        .unwrap();
        dir
    }

    #[test]
    fn openvino_declines_the_batched_segmentation_model_even_when_it_is_present() {
        let dir = models_dir_with_batched("openvino");
        let model = dir.join("segmentation-3.0.onnx");

        // The control: every other accelerated mode takes it, so a None below is the mode
        // talking and not a missing file or a mangled name.
        assert!(primary_batched_path(&model, ExecutionMode::Cuda).is_some());
        assert!(primary_batched_path(&model, ExecutionMode::MiGraphX).is_some());

        assert_eq!(
            primary_batched_path(&model, ExecutionMode::OpenVino { device_type: "GPU" }),
            None,
        );
        // The device does not enter into it: the CPU device runs the same plugin stack.
        assert_eq!(
            primary_batched_path(&model, ExecutionMode::OpenVino { device_type: "CPU" }),
            None,
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_batched_model_is_still_none_for_everyone() {
        let dir = std::env::temp_dir().join(format!("speakrs-seg-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("segmentation-3.0.onnx");

        assert_eq!(primary_batched_path(&model, ExecutionMode::Cuda), None);
        assert_eq!(
            primary_batched_path(&model, ExecutionMode::OpenVino { device_type: "GPU" }),
            None,
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
