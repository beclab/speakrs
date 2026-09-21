//! Which execution modes ask the embedding session for a precision override.
//!
//! Kept out of the shared `mod tests` block so that upstream's test additions and ours
//! never land on the same lines. `FORK.md` says why, once.

use super::*;

#[test]
fn only_openvino_asks_for_a_precision() {
    // The whole point is that this is not the pipeline's precision: segmentation goes
    // through the same provider on the same device at the default, and breaks at FP32.
    //
    // Narrowed here, which is where the previous version of this test asked for the
    // decision to be made. FP32 answers a failure in the GPU plugin, so it goes to the
    // devices that reach it and to nobody else.
    //
    // Every GPU rather than the discrete one, which is the limit and not the intent:
    // GPU, GPU.0 and GPU.1 are positions in a list, not kinds of card, and nothing here
    // can tell them apart.
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
