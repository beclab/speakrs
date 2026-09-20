use ndarray::{Array2, s};

use crate::pipeline::{
    ChunkSpeakerClusters, DecodedSegmentations, DiscreteDiarization, ExclusiveFrameDecision,
    ExclusiveFrameEvidence, FrameActivations, ScoreAwareExclusiveDiarization, SpeakerCountTrack,
};

pub struct Reconstructor<'a> {
    segmentations: &'a DecodedSegmentations,
    hard_clusters: Option<&'a ChunkSpeakerClusters>,
    start_frames: &'a [usize],
    warmup_frames: usize,
}

impl<'a> Reconstructor<'a> {
    pub fn new(
        segmentations: &'a DecodedSegmentations,
        start_frames: &'a [usize],
        warmup_frames: usize,
    ) -> Self {
        Self {
            segmentations,
            hard_clusters: None,
            start_frames,
            warmup_frames,
        }
    }

    pub fn with_clusters(
        segmentations: &'a DecodedSegmentations,
        hard_clusters: &'a ChunkSpeakerClusters,
        start_frames: &'a [usize],
        warmup_frames: usize,
    ) -> Self {
        Self {
            segmentations,
            hard_clusters: Some(hard_clusters),
            start_frames,
            warmup_frames,
        }
    }

    pub fn speaker_count(&self, output_frames: usize) -> SpeakerCountTrack {
        let num_chunks = self.segmentations.shape()[0];
        if num_chunks == 0 {
            return SpeakerCountTrack(Vec::new());
        }

        let num_frames = self.segmentations.shape()[1];
        let warmup_end = num_frames.saturating_sub(self.warmup_frames);
        let mut numerator = vec![0.0f32; output_frames];
        let mut denominator = vec![0.0f32; output_frames];

        for (chunk_idx, &start_frame) in self.start_frames.iter().enumerate().take(num_chunks) {
            for frame_idx in self.warmup_frames..warmup_end {
                let out_frame = start_frame + frame_idx;
                if out_frame >= output_frames {
                    continue;
                }

                numerator[out_frame] += self
                    .segmentations
                    .slice(s![chunk_idx, frame_idx, ..])
                    .iter()
                    .sum::<f32>();
                denominator[out_frame] += 1.0;
            }
        }

        SpeakerCountTrack(
            numerator
                .into_iter()
                .zip(denominator)
                .map(|(sum, weight)| {
                    if weight == 0.0 {
                        0
                    } else {
                        round_ties_even(sum / weight).max(0.0) as usize
                    }
                })
                .collect(),
        )
    }

    pub(crate) fn frame_activations(&self, speaker_count: &SpeakerCountTrack) -> FrameActivations {
        let Some(hard_clusters) = self.hard_clusters else {
            return FrameActivations(Array2::zeros((speaker_count.len(), 0)));
        };
        let num_chunks = self.segmentations.shape()[0];
        let num_frames = self.segmentations.shape()[1];
        let num_clusters = hard_clusters
            .iter()
            .copied()
            .filter(|cluster| *cluster >= 0)
            .max()
            .map_or(0, |cluster| cluster as usize + 1);
        let warmup_end = num_frames.saturating_sub(self.warmup_frames);
        let mut activations = Array2::<f32>::zeros((speaker_count.len(), num_clusters));

        for (chunk_idx, &start_frame) in self.start_frames.iter().enumerate().take(num_chunks) {
            let chunk_labels = hard_clusters.row(chunk_idx);
            let chunk_segmentations = self.segmentations.slice(s![chunk_idx, .., ..]);
            let local_cluster_mapping = build_cluster_mapping(&chunk_labels, num_clusters);

            for (cluster_idx, local_indices) in local_cluster_mapping.iter().enumerate() {
                if local_indices.is_empty() {
                    continue;
                }

                for frame_idx in self.warmup_frames..warmup_end {
                    let out_frame = start_frame + frame_idx;
                    if out_frame >= speaker_count.len() {
                        continue;
                    }

                    let mut score = 0.0f32;
                    for &local_idx in local_indices {
                        score = score.max(chunk_segmentations[[frame_idx, local_idx]]);
                    }
                    activations[[out_frame, cluster_idx]] += score;
                }
            }
        }

        let max_speakers_per_frame = speaker_count.iter().copied().max().unwrap_or(0);
        if activations.ncols() < max_speakers_per_frame {
            let mut padded = Array2::<f32>::zeros((activations.nrows(), max_speakers_per_frame));
            padded
                .slice_mut(s![.., ..activations.ncols()])
                .assign(&activations);
            activations = padded;
        }

        FrameActivations(activations)
    }

    pub fn reconstruct(&self, speaker_count: &SpeakerCountTrack) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            for speaker_idx in top_k_indices(&activations, frame_idx, count) {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
        }
        DiscreteDiarization(discrete)
    }

    pub fn reconstruct_smoothed(
        &self,
        speaker_count: &SpeakerCountTrack,
        epsilon: f32,
    ) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        let mut previous_speakers: Vec<usize> = Vec::new();

        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            let current_speakers =
                top_k_indices_smoothed(&activations, frame_idx, count, &previous_speakers, epsilon);
            for &speaker_idx in &current_speakers {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
            previous_speakers = current_speakers;
        }

        DiscreteDiarization(discrete)
    }
}

/// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
pub fn make_exclusive(activations: &mut Array2<f32>) {
    for mut row in activations.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            continue;
        }

        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        for (column_idx, value) in row.iter_mut().enumerate() {
            if column_idx != argmax {
                *value = 0.0;
            }
        }
    }
}

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

fn build_cluster_mapping(
    chunk_labels: &ndarray::ArrayView1<i32>,
    num_clusters: usize,
) -> Vec<Vec<usize>> {
    let mut mapping = vec![Vec::new(); num_clusters];
    for (local_idx, &label) in chunk_labels.iter().enumerate() {
        if label >= 0 {
            mapping[label as usize].push(local_idx);
        }
    }
    mapping
}

fn top_k_indices(matrix: &Array2<f32>, frame_idx: usize, k: usize) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn top_k_indices_smoothed(
    matrix: &Array2<f32>,
    frame_idx: usize,
    k: usize,
    previous_speakers: &[usize],
    epsilon: f32,
) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();

    indexed.sort_by(|left, right| {
        let score_diff = right.1 - left.1;
        if score_diff.abs() < epsilon {
            let left_was_active = previous_speakers.contains(&left.0);
            let right_was_active = previous_speakers.contains(&right.0);
            right_was_active.cmp(&left_was_active)
        } else {
            right.1.total_cmp(&left.1)
        }
    });

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn round_ties_even(value: f32) -> f32 {
    let lower = value.floor();
    let fraction = value - lower;
    let epsilon = 1e-6;

    if fraction < 0.5 - epsilon {
        return lower;
    }

    if fraction > 0.5 + epsilon {
        return value.ceil();
    }

    if lower as i64 % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::pipeline::{ChunkSpeakerClusters, DecodedSegmentations};

    #[test]
    fn speaker_count_rounds_overlap_added_sum() {
        let segmentations = DecodedSegmentations(array![
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ]);
        let reconstructor = Reconstructor::new(&segmentations, &[0, 1], 0);

        let count = reconstructor.speaker_count(4);

        assert_eq!(&*count, &[1, 1, 1, 1]);
    }

    #[test]
    fn reconstruct_selects_top_k_per_frame() {
        let segmentations =
            DecodedSegmentations(array![[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.2, 0.8]]]);
        let hard_clusters = ChunkSpeakerClusters(array![[0, 1], [0, 1]]);
        let reconstructor =
            Reconstructor::with_clusters(&segmentations, &hard_clusters, &[0, 1], 0);
        let speaker_count = SpeakerCountTrack(vec![1, 1, 1]);

        let result = reconstructor.reconstruct(&speaker_count);

        let expected: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(&*result, &expected);
    }
}

#[cfg(test)]
#[path = "reconstruct_score_aware_tests.rs"]
mod score_aware_tests;
