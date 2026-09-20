//! Types for the score-aware exclusive timeline: the support this fork keeps alongside the
//! binary mask, and the per-frame evidence for how each speaker was chosen.
//!
//! Their own file, not a block inside `data.rs`, so that upstream's edits to that file and
//! ours never land on the same lines.

use std::ops::Deref;

use ndarray::Array2;

use super::data::{DiarizationResult, DiscreteDiarization};

/// Overlap-added support for each global speaker cluster, shape (frames, speakers).
///
/// Values are sums of hard segmentation decisions from overlapping inference windows. They are
/// useful for comparing speakers within one frame, but are not calibrated probabilities.
#[derive(Debug, Clone)]
pub struct FrameSpeakerSupport(pub Array2<f32>);

impl Deref for FrameSpeakerSupport {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// How an exclusive speaker was selected for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusiveFrameDecision {
    /// Exactly one active speaker had the greatest support.
    UniqueSupport,
    /// Equal-support speakers were resolved by an unambiguous surrounding speaker.
    TieContinuity,
    /// Support and surrounding speakers did not determine one path; stable cluster order won.
    DeterministicFallback,
}

/// Compact evidence for one frame in a score-aware exclusive reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExclusiveFrameEvidence {
    /// Selected global speaker index, or `None` for silence.
    pub speaker_idx: Option<usize>,
    /// Selected speaker support for this frame.
    pub support: f32,
    /// Strongest other active speaker, if one exists.
    pub runner_up_speaker_idx: Option<usize>,
    /// Support for `runner_up_speaker_idx`, or zero when there is no runner-up.
    pub runner_up_support: f32,
    /// Selection reason, or `None` for silence.
    pub decision: Option<ExclusiveFrameDecision>,
}

/// Exclusive diarization and the frame evidence used to produce it.
#[derive(Debug, Clone)]
pub struct ScoreAwareExclusiveDiarization {
    /// Binary, single-speaker-per-frame diarization.
    pub diarization: DiscreteDiarization,
    /// Evidence aligned one-to-one with diarization frames.
    pub frames: Vec<ExclusiveFrameEvidence>,
}

impl DiarizationResult {
    /// Build an exclusive timeline from frame support while preserving the ordinary activity mask.
    pub fn score_aware_exclusive(&self) -> ScoreAwareExclusiveDiarization {
        crate::reconstruct::score_aware_exclusive(
            &self.frame_speaker_support.0,
            &self.discrete_diarization.0,
        )
    }
}
