use std::path::Path;

use ort::session::Session;

use crate::inference::{
    may_reach_openvino_gpu, with_execution_mode, with_execution_mode_precision,
};

use super::{EmbeddingModel, ExecutionMode};

#[cfg(test)]
#[path = "session_openvino_tests.rs"]
mod openvino_tests;

/// FP32 for the embedding models on the OpenVINO devices that reach the GPU plugin, nothing
/// for anyone else.
///
/// The measurement is a discrete Arc Pro B70, where these models return `CL_OUT_OF_RESOURCES`
/// out of `clFinish` at the default precision and run correctly at FP32.
///
/// It reaches every GPU, not only the discrete one, and that is a limit rather than a
/// choice: `GPU`, `GPU.0` and `GPU.1` are positions in a list, not kinds of hardware, so
/// nothing here can tell an integrated part from a card. Narrowing further needs a caller who
/// says which, and none does. What the published integrated-GPU figures were taken with is
/// FP32, so they still describe what it does.
///
/// What changed: OpenVINO on the processor and the NPU no longer get it. Neither was
/// measured to need it, and neither generates the kernel the discrete card failed in -- they
/// were being handed a switch that answers a question about a different plugin.
fn openvino_precision(mode: ExecutionMode) -> Option<&'static str> {
    may_reach_openvino_gpu(mode).then_some("FP32")
}

impl EmbeddingModel {
    pub(super) fn build_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session_with_graph(model_path, mode, false)
    }

    pub(super) fn build_session_with_graph(
        model_path: &Path,
        mode: ExecutionMode,
        cuda_graph: bool,
    ) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder =
            if cuda_graph && matches!(mode, ExecutionMode::Cuda | ExecutionMode::CudaFast) {
                Self::with_cuda_graph_mode(builder)?
            } else {
                // FP32 here and nowhere else. At the default precision these models fail on
                // a discrete Intel GPU -- CL_OUT_OF_RESOURCES out of clFinish, measured on Arc
                // Pro B70 -- and they run correctly at FP32. Segmentation must NOT be given the
                // same treatment: FP32 makes it slower there and stops its batched graph
                // compiling. Every other mode ignores the argument.
                with_execution_mode_precision(builder, mode, openvino_precision(mode))?
            };
        builder.commit_from_file(model_path)
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        use ort::ep;

        Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .with_prefer_nhwc(true)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])?)
    }

    #[cfg(not(feature = "cuda"))]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        with_execution_mode(builder, ExecutionMode::Cpu)
    }

    pub(super) fn build_fbank_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        let threads = std::thread::available_parallelism()
            .map(|count| count.get().min(4))
            .unwrap_or(1);
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    pub(super) fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => ExecutionMode::Cpu,
            _ => mode,
        }
    }

    pub(super) fn build_batched_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session(model_path, Self::single_execution_mode(mode))
    }
}
