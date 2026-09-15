use std::path::Path;

use ort::session::Session;

use crate::inference::{
    may_reach_openvino_gpu, with_execution_mode, with_execution_mode_precision,
};

use super::{EmbeddingModel, ExecutionMode};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_openvino_asks_for_a_precision() {
        // The whole point is that this is not the pipeline's precision: segmentation goes
        // through the same provider on the same device at the default, and breaks at FP32.
        //
        // Narrowed here, which is where the previous version of this test asked for the
        // decision to be made. FP32 answers a failure in the GPU plugin, so it goes to the
        // devices that reach it and to nobody else.
        for device in ["GPU", "GPU.1", "HETERO:NPU,GPU", "BATCH:GPU(4)"] {
            assert_eq!(
                openvino_precision(ExecutionMode::OpenVino {
                    device_type: device
                }),
                Some("FP32"),
                "{device:?} reaches the plugin the measurement came from",
            );
        }

        // Bare AUTO is in, and the file selection excludes it: the two guesses go opposite
        // ways on purpose. OpenVINO may resolve AUTO onto a discrete card, and a card at the
        // default precision is the CL_OUT_OF_RESOURCES this switch answers, so guessing wrong
        // here costs the session. Guessing wrong about the file only costs batching.
        assert_eq!(
            openvino_precision(ExecutionMode::OpenVino {
                device_type: "AUTO"
            }),
            Some("FP32"),
        );

        // Still every GPU rather than the discrete one: GPU, GPU.0 and GPU.1 are positions,
        // not kinds of card, and nothing here can tell them apart.
        for device in ["CPU", "NPU", "MULTI:CPU,NPU"] {
            assert_eq!(
                openvino_precision(ExecutionMode::OpenVino {
                    device_type: device
                }),
                None,
                "{device:?} was never measured to need FP32 and does not use that plugin",
            );
        }

        for mode in [
            ExecutionMode::Cpu,
            ExecutionMode::Cuda,
            ExecutionMode::CudaFast,
            ExecutionMode::MiGraphX,
        ] {
            assert_eq!(openvino_precision(mode), None, "{mode:?} must be untouched");
        }
    }
}
