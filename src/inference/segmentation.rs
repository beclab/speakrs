use crate::inference::{OvTarget, openvino_gpu_plugin, ov_target};
use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::Session;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready, with_execution_mode};
use crate::models::SEGMENTATION_ONNX;
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
    /// Whether segmentation runs batched, which is the session and not the file.
    ///
    /// A consumer that reports this had to ask whether the prepared model exists, and that
    /// answer is now one step short of the truth: the file can be there and its session can
    /// have been declined, which is exactly the case the load path started tolerating.
    pub fn is_batched(&self) -> bool {
        self.primary_batched_session.is_some()
    }

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
        let (primary_batched_session, primary_batched_elapsed) =
            timed!(match primary_batched_path(model_path, mode) {
                None => None,
                // 🔴 On OpenVINO a batched model that will not build turns batching off; it
                // does not stop the pipeline loading. Absence was already the safe direction
                // to fall, and this is the same outcome discovered one step later. There the
                // file is derived at startup by the consumer from whatever export is in the
                // cache, so a new export shape produces a model that passes onnx's checker
                // and that the GPU plugin then refuses; propagating that takes down every
                // install on the next weights revision, for a feature whose absence costs
                // speed and nothing else.
                //
                // 🔴 Every other backend keeps the behaviour it had before this fork: the
                // error propagates and the load fails. Their batched export ships with the
                // weights, so one that is present and will not build is a damaged download,
                // and stopping on it says so where falling back would not. An earlier version
                // of this fell back on every mode -- a corrupt export on CUDA stopped failing
                // the load and started costing throughput in silence, which is a change to
                // backends this fork exists to leave alone.
                //
                // The same asymmetry, for the same reason, is why the embedding loader does
                // not fall back at all: nothing derives its exports either.
                Some(path) => Self::build_session(&path, mode)
                    .map(Some)
                    .or_else(|error| {
                        if tolerates_unbuildable_batched(mode) {
                            tracing::warn!(
                                model = %path.display(),
                                %error,
                                "the batched segmentation model would not build; \
                                 running segmentation one window at a time"
                            );
                            Ok(None)
                        } else {
                            Err(error)
                        }
                    })?,
            });
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
/// On OpenVINO this is a different file, not the stock `-b32` export, and the decision
/// belongs here rather than only in `required_files`. That list is behind the `online`
/// feature, so a consumer that downloads weights some other way -- which is every consumer
/// that turns default features off -- has the whole repository in its cache and arrives here
/// with the stock file present. Deciding this in the download list would decide nothing for
/// them, this crate's own engine included.
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
    if openvino_gpu_plugin(mode) {
        return dynamic_sequence_batched_path(model_path).filter(|path| path.exists());
    }
    batched_model_path(model_path, PRIMARY_BATCH_SIZE).filter(|path| path.exists())
}

/// Whether a batched segmentation model that is present and will not build costs this mode
/// only its batching, rather than its load.
///
/// 🔴 True for OpenVINO alone, because there the file is written by the consumer at startup
/// from whatever export is in the cache: a new export shape can pass onnx's checker and still
/// be refused by the plugin, and propagating that stops every install on the next weights
/// revision over a feature whose absence costs speed. Every other backend gets the export with
/// its weights, so present-and-unbuildable is a damaged download and the load should stop on
/// it -- which is what they did before this fork, and what they must keep doing, since adding
/// a backend is not a licence to change the others.
///
/// 🔴 Narrowed to the devices that can actually refuse: OpenVINO on the processor or the NPU
/// loads the stock export that ships with the weights, so present-and-unbuildable is a damaged
/// download there too and the load should stop on it, as it does for CUDA.
///
/// ⚠️ `UnresolvedAuto` is in, and has to be. Bare `AUTO` takes the stock export -- see
/// `openvino_gpu_plugin` for why -- and may still land on a GPU that refuses it. Leaving it out
/// would mean bare `AUTO` cannot start on a machine with an Intel GPU, which is worse than what
/// it does today.
fn tolerates_unbuildable_batched(mode: ExecutionMode) -> bool {
    match mode {
        ExecutionMode::OpenVino { device_type } => matches!(
            ov_target(device_type),
            OvTarget::GpuOnly | OvTarget::GpuComposite | OvTarget::UnresolvedAuto
        ),
        _ => false,
    }
}

/// Whether this mode reaches OpenVINO's GPU plugin, which is what the substitution above is for.
///
/// 🔴 The device, not the backend. The kernel that cannot be compiled is generated by the GPU
/// plugin; the CPU and NPU plugins are different code and were never measured to have it. Asking
/// only whether the mode is OpenVINO sent every device to a derived model that has to be
/// provisioned outside this crate -- so OpenVINO on the processor, where the stock export was
/// measured to run batched at 62.1 ms per window, silently ran one window at a time instead
/// whenever nobody had prepared the substitute.
///
/// The file name OpenVINO looks for when it batches segmentation.
///
/// Public because whoever provisions the models has to write this exact name, and it is built
/// from SEGMENTATION_ONNX and PRIMARY_BATCH_SIZE rather than spelled out. Two consumers had
/// already spelled it out themselves -- a shell that prepares the file and a binary that
/// reports whether it is there -- so a change to either constant would have made both of them
/// quietly wrong, in a direction whose only symptom is batching being off.
/// The batched segmentation model this mode looks for, which is not the same file for all of
/// them.
///
/// 🔴 Asked rather than spelled, and asked per mode, because a consumer that reports whether
/// batching took has to name the file it looked for -- and naming the OpenVINO one on a CUDA
/// deployment tells the reader to go find a file nothing there wants. It answers for the mode
/// exactly as `primary_batched_path` decides, so the two cannot drift.
pub fn batched_segmentation_file_name_for(mode: ExecutionMode) -> String {
    if openvino_gpu_plugin(mode) {
        return batched_segmentation_file_name();
    }
    let stem = SEGMENTATION_ONNX
        .strip_suffix(".onnx")
        .expect("SEGMENTATION_ONNX names an .onnx file");
    format!("{stem}-b{PRIMARY_BATCH_SIZE}.onnx")
}

pub fn batched_segmentation_file_name() -> String {
    let stem = SEGMENTATION_ONNX
        .strip_suffix(".onnx")
        .expect("SEGMENTATION_ONNX names an .onnx file");
    format!("{stem}-b{PRIMARY_BATCH_SIZE}-dynseq.onnx")
}

/// The batched segmentation model OpenVINO can compile, named apart from the stock one.
///
/// A different filename rather than the same one, because the two are not interchangeable
/// and the stock one is what crashes. Sharing the name would mean a deployment without the
/// prepared model silently picking up the static graph and failing to build a session -- the
/// failure this whole path exists to avoid. Absent, batching is simply off, which is where
/// OpenVINO stood before this and is the safe direction to fall.
///
/// The file is the stock `-b32` export with its sample dimension made dynamic, which is the
/// one edit that matters and is numerically identity: verified bit-for-bit against the
/// original on CPU before either was trusted. Producing it needs an ONNX library, so it is
/// generated where one is available rather than here.
fn dynamic_sequence_batched_path(model_path: &Path) -> Option<PathBuf> {
    // The name is fixed rather than derived from what was passed, so that the one string a
    // provisioning step has to produce has exactly one definition. The stem is still checked,
    // because a caller handing this something that is not a model should get None.
    model_path.file_name()?.to_str()?.strip_suffix(".onnx")?;
    Some(model_path.with_file_name(batched_segmentation_file_name()))
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
    use crate::inference::{OvTarget, ov_target};

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

    /// The name the other two consumers produce for the current export. Neither spells it:
    /// the shell derives it from the batched export it finds in the cache, the engine asks
    /// this function -- so this pins what all three agree on today, and a change to either
    /// constant shows up here rather than as batching silently off.
    #[test]
    fn the_prepared_model_is_named_after_the_export_and_the_batch() {
        assert_eq!(
            batched_segmentation_file_name(),
            "segmentation-3.0-b32-dynseq.onnx"
        );
    }

    /// The name a consumer reports, per mode, against the file the loader actually looks for.
    /// Spelling either one on the consumer side is how the two drift; asserting them together
    /// here is what makes asking cheaper than spelling.
    #[test]
    fn the_reported_file_name_is_the_one_the_loader_looks_for() {
        let dir = models_dir_with_batched("reported");
        let model = dir.join("segmentation-3.0.onnx");
        std::fs::write(dir.join(batched_segmentation_file_name()), b"").unwrap();

        for mode in [
            ExecutionMode::Cpu,
            ExecutionMode::Cuda,
            ExecutionMode::MiGraphX,
            ExecutionMode::OpenVino { device_type: "GPU" },
            ExecutionMode::OpenVino { device_type: "CPU" },
        ] {
            let looked_for = primary_batched_path(&model, mode).unwrap_or_else(|| {
                panic!("{mode:?} found no batched model with both files present")
            });
            assert_eq!(
                looked_for.file_name().unwrap().to_str().unwrap(),
                batched_segmentation_file_name_for(mode),
                "{mode:?} reports a different file from the one it loads",
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every device string this crate can be handed, and what it resolves to.
    ///
    /// 🔴 The rows that `contains("GPU")` got wrong are the point of the table. `BATCH:GPU(4)`
    /// is the documented way to set an explicit batch size and reads as a non-GPU device when
    /// the token is compared whole; `MULTI:CPU,NPU` reaches no GPU and was sent to the derived
    /// model; bare `AUTO` is a runtime decision and fell to the stock export by the accident
    /// that its four letters do not spell GPU.
    #[test]
    fn a_device_string_resolves_to_one_answer() {
        for (device, want) in [
            ("GPU", OvTarget::GpuOnly),
            ("GPU.0", OvTarget::GpuOnly),
            ("GPU.1", OvTarget::GpuOnly),
            ("CPU", OvTarget::Cpu),
            ("NPU", OvTarget::Npu),
            ("HETERO:NPU,GPU", OvTarget::GpuComposite),
            ("AUTO:GPU,CPU", OvTarget::GpuComposite),
            ("AUTO:CPU,GPU", OvTarget::GpuComposite),
            ("BATCH:GPU", OvTarget::GpuComposite),
            ("BATCH:GPU(4)", OvTarget::GpuComposite),
            ("BATCH:GPU(16)", OvTarget::GpuComposite),
            ("MULTI:GPU.1,GPU.0", OvTarget::GpuComposite),
            ("BATCH:CPU(16)", OvTarget::NonGpuComposite),
            ("MULTI:CPU,NPU", OvTarget::NonGpuComposite),
            ("AUTO", OvTarget::UnresolvedAuto),
            ("MULTI", OvTarget::UnresolvedAuto),
            (" GPU ", OvTarget::GpuOnly),
        ] {
            assert_eq!(ov_target(device), want, "{device:?}");
        }
    }

    /// A string this crate does not recognise is treated as reaching the plugin: that costs
    /// batching when the substitute is missing, the other direction costs a session that will
    /// not build. Kept as a test because it is a decision, not a fallthrough.
    #[test]
    fn an_unrecognised_device_string_is_assumed_to_reach_the_gpu() {
        for device in ["gpu", "Gpu", "foobar", "VPU", "FUTURE:GPU"] {
            assert!(
                openvino_gpu_plugin(ExecutionMode::OpenVino {
                    device_type: device
                }),
                "{device:?} should be assumed to reach the plugin",
            );
        }
    }

    /// 🔴 Bare `AUTO` keeps the behaviour it has today, deliberately: the stock export, and a
    /// refusal it survives. Treating it as a GPU would cost a machine with no GPU the batching
    /// it currently gets, to buy a derived model nobody provisions by default.
    #[test]
    fn bare_auto_takes_the_stock_export_and_survives_a_refusal() {
        let mode = ExecutionMode::OpenVino {
            device_type: "AUTO",
        };
        assert!(
            !openvino_gpu_plugin(mode),
            "bare AUTO must not be sent to the derived model",
        );
        assert!(
            tolerates_unbuildable_batched(mode),
            "bare AUTO may still land on a GPU, so a refusal must cost batching, not the load",
        );
    }

    /// 🔴 Adding a backend must not change the others. A batched export that is present and
    /// will not build stopped the load on every backend before this fork; an earlier version
    /// of this branch made it fall back everywhere, which turned a corrupt file on CUDA from a
    /// failed start into throughput lost in silence. Only OpenVINO, where the file is derived
    /// by the consumer rather than shipped with the weights, may fall back.
    #[test]
    fn only_openvino_survives_a_batched_model_that_will_not_build() {
        // The derived model is written by the consumer and only the GPU plugin refuses it, so
        // these are the devices where an unbuildable file is a bad derivation rather than a bad
        // download. Bare AUTO is here because it takes the stock export and may still land on a
        // GPU: leaving it out would stop it starting on a machine with an Intel GPU.
        for mode in [
            ExecutionMode::OpenVino { device_type: "GPU" },
            ExecutionMode::OpenVino {
                device_type: "GPU.1",
            },
            ExecutionMode::OpenVino {
                device_type: "HETERO:NPU,GPU",
            },
            ExecutionMode::OpenVino {
                device_type: "BATCH:GPU(4)",
            },
            ExecutionMode::OpenVino {
                device_type: "AUTO",
            },
        ] {
            assert!(
                tolerates_unbuildable_batched(mode),
                "{mode:?} must degrade rather than fail: its batched model is derived, not shipped",
            );
        }

        // 🔴 OpenVINO on the processor and the NPU load the stock export, which ships with the
        // weights. Present and unbuildable means a damaged download there, exactly as it does
        // on CUDA, and the load must stop on it.
        for mode in [
            ExecutionMode::OpenVino { device_type: "CPU" },
            ExecutionMode::OpenVino { device_type: "NPU" },
            ExecutionMode::OpenVino {
                device_type: "MULTI:CPU,NPU",
            },
        ] {
            assert!(
                !tolerates_unbuildable_batched(mode),
                "{mode:?} is handed the export that ships with the weights; it must not degrade",
            );
        }

        for mode in [
            ExecutionMode::Cpu,
            ExecutionMode::Cuda,
            ExecutionMode::CudaFast,
            ExecutionMode::MiGraphX,
            ExecutionMode::CoreMl,
            ExecutionMode::CoreMlFast,
        ] {
            assert!(
                !tolerates_unbuildable_batched(mode),
                "{mode:?} kept this behaviour before this fork and must keep it: a batched \
                 export that ships with the weights and will not build is a damaged download",
            );
        }
    }

    #[test]
    fn openvino_never_takes_the_stock_batched_model() {
        let dir = models_dir_with_batched("openvino");
        let model = dir.join("segmentation-3.0.onnx");

        // The control: every other accelerated mode takes it, so a None below is the mode
        // talking and not a missing file or a mangled name.
        assert!(primary_batched_path(&model, ExecutionMode::Cuda).is_some());
        assert!(primary_batched_path(&model, ExecutionMode::MiGraphX).is_some());

        // The stock file is present and must still be refused on the GPU plugin: it is the one
        // that cannot be compiled there, so picking it up would be the crash this path avoids.
        for device in ["GPU", "GPU.0", "GPU.1", "HETERO:NPU,GPU", "AUTO:GPU,CPU"] {
            assert_eq!(
                primary_batched_path(
                    &model,
                    ExecutionMode::OpenVino {
                        device_type: device
                    }
                ),
                None,
                "{device} reaches the GPU plugin and must not take the stock export",
            );
        }

        // 🔴 And the devices that do NOT reach that plugin take it, which is the case this
        // asserted the other way round. The kernel that fails is the GPU plugin's; OpenVINO on
        // the processor compiles the stock export and was measured running it batched at 62.1 ms
        // per window, against 529.5 one window at a time. Refusing it there traded a 8.5x for a
        // file somebody else has to write.
        for device in ["CPU", "NPU"] {
            let taken = primary_batched_path(
                &model,
                ExecutionMode::OpenVino {
                    device_type: device,
                },
            );
            assert_eq!(
                taken,
                Some(dir.join(format!("segmentation-3.0-b{PRIMARY_BATCH_SIZE}.onnx"))),
                "{device} does not reach the GPU plugin and should take the stock export",
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn openvino_takes_the_prepared_model_when_it_is_there() {
        let dir = models_dir_with_batched("openvino-dynseq");
        let model = dir.join("segmentation-3.0.onnx");
        let prepared = dir.join(format!(
            "segmentation-3.0-b{PRIMARY_BATCH_SIZE}-dynseq.onnx"
        ));
        std::fs::write(&prepared, b"").unwrap();

        assert_eq!(
            primary_batched_path(&model, ExecutionMode::OpenVino { device_type: "GPU" }),
            Some(prepared),
        );

        // And nobody else goes looking for it: the stock export is what they can compile.
        let cuda = primary_batched_path(&model, ExecutionMode::Cuda).unwrap();
        assert!(
            cuda.to_str()
                .unwrap()
                .ends_with(&format!("-b{PRIMARY_BATCH_SIZE}.onnx")),
            "cuda took {cuda:?}"
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
