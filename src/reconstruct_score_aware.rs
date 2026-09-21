//! Exclusive reconstruction that reads frame support instead of the binary mask.
//!
//! Its own file, not a block inside `reconstruct.rs`, so that upstream's edits to that
//! file and ours never land on the same lines. Upstream answers the same question with
//! `ExclusiveDiarization::from_scored`, one frame at a time; this one resolves a run of
//! frames together so that equal support does not make the speaker flicker.

use ndarray::Array2;

use crate::pipeline::{
    DiscreteDiarization, ExclusiveFrameDecision, ExclusiveFrameEvidence,
    ScoreAwareExclusiveDiarization,
};

pub(crate) fn score_aware_exclusive(
    support: &Array2<f32>,
    ordinary: &Array2<f32>,
) -> ScoreAwareExclusiveDiarization {
    assert_eq!(support.raw_dim(), ordinary.raw_dim());
    let (num_frames, num_speakers) = ordinary.dim();
    let mut output = Array2::<f32>::zeros((num_frames, num_speakers));
    let mut evidence = vec![
        ExclusiveFrameEvidence {
            speaker_idx: None,
            support: 0.0,
            runner_up_speaker_idx: None,
            runner_up_support: 0.0,
            decision: None,
        };
        num_frames
    ];

    let mut component_start = 0;
    while component_start < num_frames {
        while component_start < num_frames && !row_has_activity(ordinary, component_start) {
            component_start += 1;
        }
        if component_start == num_frames {
            break;
        }
        let mut component_end = component_start + 1;
        while component_end < num_frames && row_has_activity(ordinary, component_end) {
            component_end += 1;
        }

        reconstruct_component(
            support,
            ordinary,
            component_start,
            component_end,
            &mut output,
            &mut evidence,
        );
        component_start = component_end;
    }

    ScoreAwareExclusiveDiarization {
        diarization: DiscreteDiarization(output),
        frames: evidence,
    }
}

fn row_has_activity(ordinary: &Array2<f32>, frame_idx: usize) -> bool {
    ordinary.row(frame_idx).iter().any(|value| *value > 0.5)
}

fn frame_maxima(support: &Array2<f32>, ordinary: &Array2<f32>, frame_idx: usize) -> Vec<usize> {
    let active: Vec<usize> = ordinary
        .row(frame_idx)
        .iter()
        .enumerate()
        .filter_map(|(speaker_idx, value)| (*value > 0.5).then_some(speaker_idx))
        .collect();
    let max_support = active
        .iter()
        .map(|speaker_idx| support[[frame_idx, *speaker_idx]])
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    active
        .into_iter()
        .filter(|speaker_idx| support[[frame_idx, *speaker_idx]] == max_support)
        .collect()
}

fn reconstruct_component(
    support: &Array2<f32>,
    ordinary: &Array2<f32>,
    start: usize,
    end: usize,
    output: &mut Array2<f32>,
    evidence: &mut [ExclusiveFrameEvidence],
) {
    let num_speakers = ordinary.ncols();
    let frames = end - start;
    let maxima: Vec<Vec<usize>> = (start..end)
        .map(|frame_idx| frame_maxima(support, ordinary, frame_idx))
        .collect();
    let unreachable = usize::MAX / 4;
    let mut costs = vec![vec![unreachable; num_speakers]; frames];
    let mut parents = vec![vec![usize::MAX; num_speakers]; frames];

    for &speaker_idx in &maxima[0] {
        costs[0][speaker_idx] = 0;
    }
    for offset in 1..frames {
        for &speaker_idx in &maxima[offset] {
            let mut best = (unreachable, usize::MAX);
            for &previous_idx in &maxima[offset - 1] {
                let candidate = (
                    costs[offset - 1][previous_idx] + usize::from(previous_idx != speaker_idx),
                    previous_idx,
                );
                if candidate < best {
                    best = candidate;
                }
            }
            costs[offset][speaker_idx] = best.0;
            parents[offset][speaker_idx] = best.1;
        }
    }

    let mut selected = vec![0; frames];
    selected[frames - 1] = maxima[frames - 1]
        .iter()
        .copied()
        .min_by_key(|speaker_idx| (costs[frames - 1][*speaker_idx], *speaker_idx))
        .unwrap_or(0);
    for offset in (1..frames).rev() {
        selected[offset - 1] = parents[offset][selected[offset]];
    }

    let mut decisions = vec![ExclusiveFrameDecision::UniqueSupport; frames];
    let mut offset = 0;
    while offset < frames {
        if maxima[offset].len() == 1 {
            offset += 1;
            continue;
        }
        let run_start = offset;
        while offset < frames && maxima[offset].len() > 1 {
            offset += 1;
        }
        let run_end = offset;
        let left = (0..run_start)
            .rev()
            .find(|idx| maxima[*idx].len() == 1)
            .map(|idx| maxima[idx][0]);
        let right = (run_end..frames)
            .find(|idx| maxima[*idx].len() == 1)
            .map(|idx| maxima[idx][0]);
        let anchor = match (left, right) {
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            _ => None,
        };
        let continuity_resolved = anchor.is_some_and(|speaker_idx| {
            (run_start..run_end)
                .all(|idx| maxima[idx].contains(&speaker_idx) && selected[idx] == speaker_idx)
        });
        decisions[run_start..run_end].fill(if continuity_resolved {
            ExclusiveFrameDecision::TieContinuity
        } else {
            ExclusiveFrameDecision::DeterministicFallback
        });
    }

    for offset in 0..frames {
        let frame_idx = start + offset;
        let speaker_idx = selected[offset];
        output[[frame_idx, speaker_idx]] = 1.0;
        let runner_up = ordinary
            .row(frame_idx)
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (*value > 0.5 && idx != speaker_idx).then_some(idx))
            .max_by(|left, right| {
                support[[frame_idx, *left]]
                    .total_cmp(&support[[frame_idx, *right]])
                    .then_with(|| right.cmp(left))
            });
        evidence[frame_idx] = ExclusiveFrameEvidence {
            speaker_idx: Some(speaker_idx),
            support: support[[frame_idx, speaker_idx]],
            runner_up_speaker_idx: runner_up,
            runner_up_support: runner_up.map_or(0.0, |idx| support[[frame_idx, idx]]),
            decision: Some(decisions[offset]),
        };
    }
}
