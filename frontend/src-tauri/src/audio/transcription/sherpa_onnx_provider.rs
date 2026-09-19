use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::{info, warn};
use std::sync::Arc;

const SAMPLE_RATE: u32 = 16000;
/// Local Sherpa-ONNX SenseVoice/Whisper models are trained on short audio windows.
/// Feeding them minutes of audio at once causes encoder attention to allocate
/// O(n^2) memory and crash.  Split long audio into 30s windows with a 1s
/// overlap and remove duplicated text at the seams.
const MAX_CHUNK_DURATION_SECS: f64 = 30.0;
const OVERLAP_DURATION_SECS: f64 = 1.0;
const STRIDE_DURATION_SECS: f64 = MAX_CHUNK_DURATION_SECS - OVERLAP_DURATION_SECS; // 29s
const MAX_CHUNK_SAMPLES: usize = (MAX_CHUNK_DURATION_SECS * SAMPLE_RATE as f64) as usize;
const OVERLAP_SAMPLES: usize = (OVERLAP_DURATION_SECS * SAMPLE_RATE as f64) as usize;
const SPLIT_SEARCH_RADIUS_SECS: f64 = 1.0;
const SILENCE_WINDOW_SECS: f64 = 0.2;
const MIN_LOGICAL_CHUNK_DURATION_SECS: f64 = 0.5;
const MIN_OVERLAP_CHARS: usize = 2;
const MAX_OVERLAP_CHARS: usize = 30;

pub struct SherpaOnnxProvider {
    engine: Arc<crate::sherpa_onnx_engine::SherpaOnnxEngine>,
}

impl SherpaOnnxProvider {
    pub fn new(engine: Arc<crate::sherpa_onnx_engine::SherpaOnnxEngine>) -> Self {
        Self { engine }
    }

    async fn transcribe_chunk(&self, samples: &[f32]) -> Result<String, TranscriptionError> {
        let rec = self.engine.recognizer.lock().await;

        // sherpa-onnx OfflineRecognizer::decode is synchronous and CPU-heavy.
        // Running it directly inside this async function can monopolize a Tokio
        // worker thread for several seconds, starving the realtime translation
        // worker even though Hy-MT2 itself is running on the GPU. block_in_place
        // tells the multi-thread runtime to hand other async work to replacement
        // workers while the current thread performs the native decode.
        let result = tokio::task::block_in_place(|| {
            let stream = rec.create_stream();
            stream.accept_waveform(SAMPLE_RATE as i32, samples);
            rec.decode(&stream);
            stream.get_result()
        })
        .ok_or_else(|| TranscriptionError::EngineFailed("No result".into()))?;

        let text = result.text.trim().to_string();
        if self.engine.get_model_type() == crate::sherpa_onnx_engine::AsrModelType::Whisper {
            Ok(truncate_whisper_repetition_loop(&text))
        } else {
            Ok(text)
        }
    }
}

/// Whisper occasionally falls into a token/short-phrase repetition loop on
/// difficult audio (e.g. "non non non..." or "in forma in forma...").
/// Detect conservative, clearly pathological runs and keep only the first
/// occurrence. This happens before transcript emission, so UI/export/translation
/// all receive the same cleaned text.
fn truncate_whisper_repetition_loop(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 8 {
        return text.trim().to_string();
    }

    let normalized: Vec<String> = tokens
        .iter()
        .map(|token| {
            token
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '’')
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .collect();

    let mut best: Option<(usize, usize, usize)> = None;

    for start in 0..tokens.len() {
        let max_phrase = 5usize.min((tokens.len() - start) / 2);
        for phrase_len in 1..=max_phrase {
            let phrase = &normalized[start..start + phrase_len];
            if phrase.iter().any(|t| t.is_empty()) {
                continue;
            }

            let min_repeats = match phrase_len {
                1 => 8,
                2 => 6,
                3 => 5,
                _ => 4,
            };

            let mut repeats = 1usize;
            while start + (repeats + 1) * phrase_len <= normalized.len() {
                let next_start = start + repeats * phrase_len;
                let next_end = next_start + phrase_len;
                if normalized[next_start..next_end] == *phrase {
                    repeats += 1;
                } else {
                    break;
                }
            }

            if repeats >= min_repeats {
                let candidate = (start, phrase_len, repeats);
                best = match best {
                    None => Some(candidate),
                    Some(current) => {
                        // Prefer the earliest loop. At the same position prefer
                        // the run covering more repeated tokens.
                        let current_coverage = current.1 * current.2;
                        let candidate_coverage = phrase_len * repeats;
                        if start < current.0
                            || (start == current.0 && candidate_coverage > current_coverage)
                        {
                            Some(candidate)
                        } else {
                            Some(current)
                        }
                    }
                };
            }
        }
    }

    let Some((start, phrase_len, repeats)) = best else {
        return text.trim().to_string();
    };

    let keep_end = start + phrase_len;
    let cleaned = tokens[..keep_end].join(" ");
    warn!(
        "Whisper repetition loop detected: start_token={}, phrase_len={}, repeats={}, original_tokens={}, cleaned_tokens={}",
        start,
        phrase_len,
        repeats,
        tokens.len(),
        keep_end
    );
    cleaned
}

/// Remove the longest common suffix/prefix overlap between `prev` and `next`.
/// Returns a sub-slice of `next` with the duplicated text stripped.
fn remove_text_overlap<'a>(prev: &str, next: &'a str) -> &'a str {
    let prev_len = prev.chars().count();
    let next_len = next.chars().count();
    let max_len = MAX_OVERLAP_CHARS.min(prev_len).min(next_len);
    if max_len < MIN_OVERLAP_CHARS {
        return next;
    }

    // Last `max_len` chars of prev, in original order.
    let prev_tail: Vec<char> = prev
        .chars()
        .rev()
        .take(max_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let next_prefix: Vec<char> = next.chars().take(max_len).collect();

    for len in (MIN_OVERLAP_CHARS..=max_len).rev() {
        if prev_tail[prev_tail.len() - len..] == next_prefix[..len] {
            let byte_pos = next
                .char_indices()
                .nth(len)
                .map(|(i, _)| i)
                .unwrap_or(next.len());
            return &next[byte_pos..];
        }
    }

    next
}

#[async_trait]
impl TranscriptionProvider for SherpaOnnxProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        _language: Option<String>,
    ) -> Result<TranscriptResult, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::AudioTooShort {
                samples: 0,
                minimum: 1600,
            });
        }

        if audio.len() <= MAX_CHUNK_SAMPLES {
            let text = self.transcribe_chunk(&audio).await?;
            return Ok(TranscriptResult {
                text,
                confidence: Some(0.9),
                is_partial: false,
            });
        }

        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        // Logical chunks are stride-length (29s) so we can add 1s overlap and
        // still stay within the 30s model window.
        let logical_ranges = crate::audio::chunking::split_at_silence(
            &audio,
            SAMPLE_RATE,
            STRIDE_DURATION_SECS,
            SPLIT_SEARCH_RADIUS_SECS,
            SILENCE_WINDOW_SECS,
            MIN_LOGICAL_CHUNK_DURATION_SECS,
        );

        // Build overlapping windows: each window covers its logical chunk plus
        // the first 1s of the following logical chunk.
        let mut windows: Vec<(usize, usize)> = Vec::with_capacity(logical_ranges.len());
        for (i, &(log_start, log_end)) in logical_ranges.iter().enumerate() {
            let start = log_start;
            let end = if i + 1 < logical_ranges.len() {
                (log_end + OVERLAP_SAMPLES)
                    .min(audio.len())
                    .min(start + MAX_CHUNK_SAMPLES)
            } else {
                log_end
            };
            windows.push((start, end));
        }

        info!(
            "Sherpa-ONNX: splitting {:.1}s audio into {} overlapping chunks ({}s stride, {}s overlap)",
            duration_sec,
            windows.len(),
            STRIDE_DURATION_SECS,
            OVERLAP_DURATION_SECS
        );

        let mut combined_text = String::new();
        for (i, (start, end)) in windows.iter().enumerate() {
            let chunk = &audio[*start..*end];
            match self.transcribe_chunk(chunk).await {
                Ok(text) => {
                    let text = if i == 0 {
                        text.as_str()
                    } else {
                        remove_text_overlap(&combined_text, &text)
                    };
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        if !combined_text.is_empty()
                            && !combined_text.ends_with(char::is_whitespace)
                        {
                            combined_text.push(' ');
                        }
                        combined_text.push_str(trimmed);
                    }
                }
                Err(e) => {
                    warn!(
                        "Sherpa-ONNX chunk {}/{} ({:.1}s) failed: {}",
                        i + 1,
                        windows.len(),
                        chunk.len() as f64 / SAMPLE_RATE as f64,
                        e
                    );
                }
            }
        }

        Ok(TranscriptResult {
            text: combined_text,
            confidence: Some(0.85),
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        true
    }
    async fn get_current_model(&self) -> Option<String> {
        Some(self.engine.get_model_name().to_string())
    }
    fn provider_name(&self) -> &'static str {
        "Sherpa-ONNX (Native Rust)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_repetition_single_word_loop_is_truncated() {
        let input = "non non non non non non non non non non non non";
        assert_eq!(truncate_whisper_repetition_loop(input), "non");
    }

    #[test]
    fn test_whisper_repetition_phrase_loop_is_truncated() {
        let input = "Non vi ho in un'altra forma in forma in forma in forma in forma in forma in forma in forma";
        assert_eq!(
            truncate_whisper_repetition_loop(input),
            "Non vi ho in un'altra forma"
        );
    }

    #[test]
    fn test_whisper_repetition_punctuation_is_normalized() {
        let input = "inizio in forma, in forma, in forma, in forma, in forma, in forma, fine";
        assert_eq!(
            truncate_whisper_repetition_loop(input),
            "inizio in forma,"
        );
    }

    #[test]
    fn test_whisper_legitimate_small_repetition_is_kept() {
        let input = "no no no, aspetta, non e quello che intendo";
        assert_eq!(truncate_whisper_repetition_loop(input), input);
    }

    #[test]
    fn test_remove_text_overlap_basic() {
        let prev = "今天天气";
        let next = "天气很好";
        assert_eq!(remove_text_overlap(prev, next), "很好");
    }

    #[test]
    fn test_remove_text_overlap_with_punctuation() {
        let prev = "我们出发了。";
        let next = "出发了。接下来";
        assert_eq!(remove_text_overlap(prev, next), "接下来");
    }

    #[test]
    fn test_remove_text_overlap_no_match() {
        let prev = "完全无关";
        let next = "另一句话";
        assert_eq!(remove_text_overlap(prev, next), next);
    }

    #[test]
    fn test_remove_text_overlap_too_short() {
        let prev = "你好";
        let next = "好世界";
        // "好" alone is below MIN_OVERLAP_CHARS (2), so keep all.
        assert_eq!(remove_text_overlap(prev, next), next);
    }
}
