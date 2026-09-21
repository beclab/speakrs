//! Which segmentation export each execution mode loads, and who may fall back without it.
//!
//! Kept out of the shared `mod tests` block so that upstream's test additions and ours
//! never land on the same lines. `FORK.md` says why, once.

use super::*;
use crate::inference::{OvTarget, openvino_gpu_plugin, ov_target};

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

/// The name a provisioning step has to produce for the current export. It is built from
/// two constants rather than spelled out, so this pins what that name is today: change
/// either constant and it shows up here, rather than as batching silently off.
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
        let looked_for = primary_batched_path(&model, mode)
            .unwrap_or_else(|| panic!("{mode:?} found no batched model with both files present"));
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
/// The rows that `contains("GPU")` got wrong are the point of the table. `BATCH:GPU(4)`
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
        // A list carrying a name this crate cannot read is not a list without a GPU.
        ("AUTO:-CPU", OvTarget::GpuComposite),
        ("MULTI:-CPU", OvTarget::GpuComposite),
        ("AUTO:BATCH:GPU(4)", OvTarget::GpuComposite),
    ] {
        assert_eq!(ov_target(device), want, "{device:?}");
    }
}

/// `NonGpuComposite` is the only answer that both takes the stock export and declines to
/// survive its refusal, so it is claimed only when every name in the list was read. Reaching
/// it by finding no GPU cannot tell "this list names no GPU" from "this list holds a name I
/// cannot read" -- and `AUTO:-CPU`, OpenVINO's documented way to ask for everything except
/// the processor, is the second. On an Intel machine that is how you ask for the card, so
/// the direction that fold takes is: the string asking for the GPU most plainly is the one
/// that fails to load.
///
/// The two readable rows are here to hold the other edge. Widening this to "assume GPU
/// whenever no GPU was named" would send `MULTI:CPU,NPU` to a derived model nobody
/// provisions on a processor, and cost it the batching it has today.
#[test]
fn a_device_list_this_cannot_read_is_not_a_list_without_a_gpu() {
    for device in ["AUTO:-CPU", "MULTI:-CPU", "AUTO:BATCH:GPU(4)", "AUTO:GPU("] {
        let mode = ExecutionMode::OpenVino {
            device_type: device,
        };
        assert!(
            openvino_gpu_plugin(mode),
            "{device:?} holds a name this crate cannot read, so it must be assumed to \
             reach the plugin",
        );
        assert!(
            tolerates_unbuildable_batched(mode),
            "{device:?} must survive a batched model the plugin refuses, not fail the load",
        );
    }

    for device in ["MULTI:CPU,NPU", "BATCH:CPU(16)"] {
        let mode = ExecutionMode::OpenVino {
            device_type: device,
        };
        assert!(
            !openvino_gpu_plugin(mode),
            "{device:?} names only devices this crate read, and none is a GPU",
        );
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

/// Bare `AUTO` keeps the behaviour it has today, deliberately: the stock export, and a
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

/// Adding a backend must not change the others. A batched export that is present and
/// will not build stopped the load on every backend before this one was added; an earlier
/// version of this work made it fall back everywhere, which turned a corrupt file on CUDA
/// from a failed start into throughput lost in silence. Only OpenVINO, where the file is
/// derived by whoever provisions the models rather than shipped with the weights, may
/// fall back.
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

    // OpenVINO on the processor and the NPU load the stock export, which ships with the
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
            "{mode:?} had this behaviour before this backend existed and must keep it: a \
             batched export that ships with the weights and will not build is a damaged \
             download",
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

    // And the devices that do NOT reach that plugin take it, which is the case this
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
