// audio/transcription/worker.rs
//
// Transcription worker that processes VAD audio chunks through Sherpa-ONNX.

use super::engine::TranscriptionEngine;
use super::provider::TranscriptionError;
use crate::audio::AudioChunk;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Runtime};

// Sequence counter for transcript updates
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

// Speech detection flag - reset per recording session
static SPEECH_DETECTED_EMITTED: AtomicBool = AtomicBool::new(false);

/// Reset the speech detected flag for a new recording session
pub fn reset_speech_detected_flag() {
    SPEECH_DETECTED_EMITTED.store(false, Ordering::SeqCst);
    info!("🔍 SPEECH_DETECTED_EMITTED reset to: {}", SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst));
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranscriptUpdate {
    pub text: String,
    pub timestamp: String,
    pub source: String,
    pub sequence_id: u64,
    pub chunk_start_time: f64,
    pub is_partial: bool,
    pub confidence: f32,
    pub audio_start_time: f64,
    pub audio_end_time: f64,
    pub duration: f64,
}

/// Transcription task ensuring ZERO chunk loss
pub fn start_transcription_task<R: Runtime>(
    app: AppHandle<R>,
    transcription_receiver: tokio::sync::mpsc::UnboundedReceiver<AudioChunk>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("🚀 Starting transcription task with Sherpa-ONNX");

        // 新录音开始：sequence 重新计数，清空待译队列与已见集合
        crate::translation::reset_translation_session();

        // Initialize transcription engine
        let transcription_engine = match super::engine::get_or_init_transcription_engine(&app).await {
            Ok(engine) => engine,
            Err(e) => {
                error!("Failed to initialize transcription engine: {}", e);
                let _ = app.emit("transcription-error", serde_json::json!({
                    "error": e,
                    "userMessage": "Recording failed: Unable to initialize speech recognition. Please check your model settings.",
                    "actionable": true
                }));
                return;
            }
        };

        let engine_name = transcription_engine.provider_name();
        info!("Using transcription engine: {}", engine_name);
        let active_model_name = transcription_engine
            .get_current_model()
            .await
            .unwrap_or_else(|| "unknown".to_string());
        let whisper_live_mode = active_model_name.starts_with("whisper-");

        // ── X-ASR streaming branch (bypasses chunk worker) ──
        if engine_name == "x-asr" {
            info!("🎙️ X-ASR detected — entering streaming mode");
            super::x_asr_provider::reset_xasr_sequence_counter();
            if let TranscriptionEngine::Provider(provider_arc) = &transcription_engine {
                if let Some(xasr) = provider_arc.as_any().downcast_ref::<super::x_asr_provider::XAsrProvider>() {
                    xasr.run_streaming(transcription_receiver, app).await;
                    info!("🎙️ X-ASR streaming task completed");
                    return;
                }
            }
            error!("X-ASR provider downcast failed — falling back to chunk mode (will fail)");
        }

        // Single worker mode for ordered emission
        const NUM_WORKERS: usize = 1;
        let (work_sender, work_receiver) = tokio::sync::mpsc::unbounded_channel::<AudioChunk>();
        let work_receiver = Arc::new(tokio::sync::Mutex::new(work_receiver));

        let chunks_queued = Arc::new(AtomicU64::new(0));
        let chunks_completed = Arc::new(AtomicU64::new(0));
        let input_finished = Arc::new(AtomicBool::new(false));

        info!("📊 Starting {} transcription worker (serial mode)", NUM_WORKERS);

        // Spawn worker tasks
        let mut worker_handles = Vec::new();
        for worker_id in 0..NUM_WORKERS {
            let engine_clone = match &transcription_engine {
                TranscriptionEngine::Provider(p) => TranscriptionEngine::Provider(p.clone()),
            };
            let app_clone = app.clone();
            let work_receiver_clone = work_receiver.clone();
            let chunks_completed_clone = chunks_completed.clone();
            let input_finished_clone = input_finished.clone();
            let chunks_queued_clone = chunks_queued.clone();
            let whisper_live_mode = whisper_live_mode;

            let worker_handle = tokio::spawn(async move {
                info!("👷 Worker {} started", worker_id);

                let initial_model_loaded = engine_clone.is_model_loaded().await;
                let current_model = engine_clone
                    .get_current_model()
                    .await
                    .unwrap_or_else(|| "unknown".to_string());

                let whisper_live_mode = current_model.starts_with("whisper-");
                let mut last_whisper_final_text = String::new();
                let mut whisper_best_partials: HashMap<u64, String> = HashMap::new();

                if initial_model_loaded {
                    info!(
                        "✅ Worker {}: model '{}' is loaded and ready",
                        worker_id, current_model
                    );
                } else {
                    warn!("⚠️ Worker {}: model not loaded - chunks may be skipped", worker_id);
                }

                loop {
                    let chunk = {
                        let mut receiver = work_receiver_clone.lock().await;
                        receiver.recv().await
                    };

                    match chunk {
                        Some(chunk) => {
                            let should_log_this_chunk = chunk.chunk_id % 10 == 0;

                            if should_log_this_chunk {
                                info!(
                                    "👷 Worker {} processing chunk {} with {} samples",
                                    worker_id,
                                    chunk.chunk_id,
                                    chunk.data.len()
                                );
                            }

                            if !engine_clone.is_model_loaded().await {
                                warn!("⚠️ Worker {}: Model unloaded, but continuing to preserve chunk {}", worker_id, chunk.chunk_id);
                                chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                continue;
                            }

                            let chunk_timestamp = chunk.timestamp;
                            let chunk_duration = chunk.data.len() as f64 / chunk.sample_rate as f64;

                            // Whisper live-preview chunks encode "partial" in the high bit
                            // and keep the lower bits stable across partial -> final updates.
                            let requested_partial = whisper_live_mode
                                && (chunk.chunk_id
                                    & super::super::pipeline::WHISPER_PARTIAL_CHUNK_FLAG)
                                    != 0;
                            let logical_chunk_id = chunk.chunk_id
                                & !super::super::pipeline::WHISPER_PARTIAL_CHUNK_FLAG;
                            let sequence_id = if whisper_live_mode {
                                logical_chunk_id
                            } else {
                                SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst)
                            };
                            let audio_start_time = chunk_timestamp;
                            let audio_end_time = chunk_timestamp + chunk_duration;

                            // Set chunk context on the engine so the streaming provider
                            // can emit partial transcript-update events with correct metadata
                            engine_clone.set_chunk_context(
                                sequence_id,
                                chunk_timestamp,
                                audio_start_time,
                                audio_end_time,
                                chunk_duration,
                            );

                            match transcribe_chunk_with_provider(&engine_clone, chunk, &app_clone).await {
                                Ok((transcript, confidence_opt, is_partial)) => {
                                    let is_partial = is_partial || requested_partial;
                                    let mut transcript = if whisper_live_mode
                                        && !last_whisper_final_text.is_empty()
                                    {
                                        strip_whisper_cross_segment_overlap(
                                            &last_whisper_final_text,
                                            &transcript,
                                        )
                                    } else {
                                        transcript
                                    };

                                    if whisper_live_mode {
                                        if is_partial {
                                            let replace_best = whisper_best_partials
                                                .get(&sequence_id)
                                                .map(|best| transcript_information_score(&transcript)
                                                    > transcript_information_score(best))
                                                .unwrap_or(true);
                                            if replace_best && !transcript.trim().is_empty() {
                                                whisper_best_partials
                                                    .insert(sequence_id, transcript.clone());
                                            }
                                        } else if let Some(best_partial) =
                                            whisper_best_partials.remove(&sequence_id)
                                        {
                                            if should_keep_partial_over_final(
                                                &best_partial,
                                                &transcript,
                                            ) {
                                                warn!(
                                                    "Whisper final regression blocked for seq={}: final='{}' best_partial='{}'",
                                                    sequence_id,
                                                    transcript,
                                                    best_partial
                                                );
                                                transcript = best_partial;
                                            }
                                        }
                                    }

                                    if !transcript.trim().is_empty() {
                                        if whisper_live_mode && !is_partial {
                                            last_whisper_final_text = transcript.clone();
                                        }
                                        info!("✅ Worker {} transcribed: {} (confidence: {:?}, partial: {})",
                                              worker_id, transcript, confidence_opt, is_partial);

                                        // Emit speech-detected event
                                        if !SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst) {
                                            SPEECH_DETECTED_EMITTED.store(true, Ordering::SeqCst);
                                            let _ = app_clone.emit("speech-detected", serde_json::json!({
                                                "message": "Speech activity detected"
                                            }));
                                        }

                                        let update = TranscriptUpdate {
                                            text: transcript,
                                            timestamp: format_current_timestamp(),
                                            source: "Audio".to_string(),
                                            sequence_id,
                                            chunk_start_time: chunk_timestamp,
                                            is_partial,
                                            confidence: confidence_opt.unwrap_or(0.85),
                                            audio_start_time,
                                            audio_end_time,
                                            duration: chunk_duration,
                                        };

                                        match app_clone.emit("transcript-update", &update) {
                                            Ok(_) => info!(
                                                "Worker {}: ✅ Emitted transcript-update seq={}",
                                                worker_id, sequence_id
                                            ),
                                            Err(e) => error!(
                                                "Worker {}: Failed to emit transcript update: {}",
                                                worker_id, e
                                            ),
                                        }

                                        // Partial Whisper snapshots are translated too, but stay
                                        // replaceable. Only the natural/safety final becomes a
                                        // committed translation.
                                        if is_partial {
                                            crate::translation::queue_partial_translation(
                                                &app_clone,
                                                &update.text,
                                                sequence_id,
                                            );
                                        } else {
                                            crate::translation::queue_translation(
                                                &app_clone,
                                                &update.text,
                                                sequence_id,
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    match e {
                                        TranscriptionError::AudioTooShort { .. } => {
                                            info!("Worker {}: {}", worker_id, e);
                                            chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                            continue;
                                        }
                                        TranscriptionError::ModelNotLoaded => {
                                            warn!("Worker {}: Model unloaded during transcription", worker_id);
                                            chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                            continue;
                                        }
                                        _ => {
                                            warn!("Worker {}: Transcription failed: {}", worker_id, e);
                                            let _ = app_clone.emit("transcription-warning", e.to_string());
                                        }
                                    }
                                }
                            }

                            let completed = chunks_completed_clone.fetch_add(1, Ordering::SeqCst) + 1;
                            let queued = chunks_queued_clone.load(Ordering::SeqCst);

                            if completed % 5 == 0 || should_log_this_chunk {
                                info!(
                                    "Worker {}: Progress {}/{} chunks ({:.1}%)",
                                    worker_id,
                                    completed,
                                    queued,
                                    (completed as f64 / queued.max(1) as f64 * 100.0)
                                );
                            }

                            let progress_percentage = if queued > 0 {
                                (completed as f64 / queued as f64 * 100.0) as u32
                            } else {
                                100
                            };

                            let _ = app_clone.emit("transcription-progress", serde_json::json!({
                                "worker_id": worker_id,
                                "chunks_completed": completed,
                                "chunks_queued": queued,
                                "progress_percentage": progress_percentage,
                                "message": format!("Worker {} processing... ({}/{})", worker_id, completed, queued)
                            }));
                        }
                        None => {
                            if input_finished_clone.load(Ordering::SeqCst) {
                                let final_queued = chunks_queued_clone.load(Ordering::SeqCst);
                                let final_completed = chunks_completed_clone.load(Ordering::SeqCst);

                                if final_completed >= final_queued {
                                    info!(
                                        "👷 Worker {} finishing - all {}/{} chunks processed",
                                        worker_id, final_completed, final_queued
                                    );
                                    break;
                                } else {
                                    warn!("👷 Worker {} detected potential chunk loss: {}/{} completed, waiting...", worker_id, final_completed, final_queued);
                                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                                }
                            } else {
                                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            }
                        }
                    }
                }

                info!("👷 Worker {} completed", worker_id);
            });

            worker_handles.push(worker_handle);
        }

        // Main dispatcher: receive chunks and distribute to workers
        let mut receiver = transcription_receiver;
        'dispatch: while let Some(chunk) = receiver.recv().await {
            // Split over-long VAD segments so SenseVoice commits text
            // incrementally instead of as one huge block (see split_long_chunk)
            for chunk in split_long_chunk(chunk, whisper_live_mode) {
                let queued = chunks_queued.fetch_add(1, Ordering::SeqCst) + 1;
                info!(
                    "📥 Dispatching chunk {} to workers (total queued: {})",
                    chunk.chunk_id, queued
                );

                if let Err(_) = work_sender.send(chunk) {
                    error!("❌ Failed to send chunk to workers - this should not happen!");
                    break 'dispatch;
                }
            }
        }

        // Signal that input is finished
        input_finished.store(true, Ordering::SeqCst);
        drop(work_sender);

        let total_chunks_queued = chunks_queued.load(Ordering::SeqCst);
        info!("📭 Input finished with {} total chunks queued.", total_chunks_queued);

        let _ = app.emit("transcription-queue-complete", serde_json::json!({
            "total_chunks": total_chunks_queued,
            "message": format!("{} chunks queued for processing - waiting for completion", total_chunks_queued)
        }));

        // Wait for all workers to complete
        for (worker_id, handle) in worker_handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                error!("❌ Worker {} panicked: {:?}", worker_id, e);
            } else {
                info!("✅ Worker {} completed successfully", worker_id);
            }
        }

        // Final verification
        let final_queued = chunks_queued.load(Ordering::SeqCst);
        let final_completed = chunks_completed.load(Ordering::SeqCst);

        if final_queued == final_completed {
            info!(
                "🎉 ALL {} chunks processed successfully - ZERO chunks lost!",
                final_completed
            );
        } else {
            error!(
                "❌ Chunk loss detected: {} queued, {} completed",
                final_queued, final_completed
            );
            let _ = app.emit("transcript-chunk-loss-detected", serde_json::json!({
                "chunks_queued": final_queued,
                "chunks_completed": final_completed,
                "chunks_lost": final_queued - final_completed,
                "message": "Some transcript chunks may have been lost during shutdown"
            }));
        }

        info!("✅ Transcription task completed");
    })
}

/// Maximum length of a VAD segment sent to transcription in one shot.
/// SenseVoice is offline (pseudo-streaming): text is committed only when a whole
/// VAD segment finishes, so very long segments (e.g. continuous news broadcast
/// with BGM, where silero can emit a single 60s+ segment) would surface as one
/// huge block of text and translation. Split anything longer at low-energy
/// points so results stream out incrementally.
const MAX_TRANSCRIPTION_SEGMENT_SAMPLES: usize = 15 * 16000; // SenseVoice: 15s
const MAX_WHISPER_TRANSCRIPTION_SEGMENT_SAMPLES: usize = 29 * 16000; // below Whisper 30s window

/// Split an over-long VAD chunk at low-energy points. SenseVoice keeps the old
/// 15s limit. Whisper live snapshots/finals are allowed up to 29s; otherwise a
/// 16s snapshot would be split into 15s + 1s with the same sequence_id and the
/// tiny tail would overwrite the complete partial subtitle.
fn split_long_chunk(chunk: AudioChunk, whisper_live_mode: bool) -> Vec<AudioChunk> {
    let data = if chunk.sample_rate != 16000 {
        crate::audio::audio_processing::resample_audio(&chunk.data, chunk.sample_rate, 16000)
    } else {
        chunk.data
    };

    let max_segment_samples = if whisper_live_mode {
        MAX_WHISPER_TRANSCRIPTION_SEGMENT_SAMPLES
    } else {
        MAX_TRANSCRIPTION_SEGMENT_SAMPLES
    };

    if data.len() <= max_segment_samples {
        return vec![AudioChunk {
            data,
            sample_rate: 16000,
            ..chunk
        }];
    }

    let duration_s = data.len() as f64 / 16000.0;
    let segment = crate::audio::vad::SpeechSegment {
        samples: data,
        start_timestamp_ms: chunk.timestamp * 1000.0,
        end_timestamp_ms: chunk.timestamp * 1000.0 + duration_s * 1000.0,
        confidence: 1.0,
    };
    let parts =
        crate::audio::common::split_segment_at_silence(&segment, max_segment_samples);
    info!(
        "✂️ Split long VAD segment {:.1}s → {} parts",
        duration_s,
        parts.len()
    );

    parts
        .into_iter()
        .map(|part| AudioChunk {
            data: part.samples,
            sample_rate: 16000,
            timestamp: part.start_timestamp_ms / 1000.0,
            chunk_id: chunk.chunk_id,
            device_type: chunk.device_type.clone(),
        })
        .collect()
}

/// Rough information score used to remember the most complete partial snapshot.
fn transcript_information_score(text: &str) -> usize {
    let words = text.split_whitespace().filter(|w| !w.trim().is_empty()).count();
    let chars = text.chars().filter(|c| !c.is_whitespace()).count();
    words * 8 + chars
}

/// A final Whisper decode can occasionally regress to only the last few words of
/// a segment even though an earlier sliding-window partial contained the full
/// utterance. Keep the partial only when the regression is obvious; otherwise
/// trust the final decode so normal Whisper corrections still work.
fn should_keep_partial_over_final(partial: &str, final_text: &str) -> bool {
    let partial = partial.trim();
    let final_text = final_text.trim();
    if partial.is_empty() || final_text.is_empty() {
        return !partial.is_empty() && final_text.is_empty();
    }

    let partial_words = partial.split_whitespace().count();
    let final_words = final_text.split_whitespace().count();
    let partial_chars = partial.chars().filter(|c| !c.is_whitespace()).count();
    let final_chars = final_text.chars().filter(|c| !c.is_whitespace()).count();

    // Require a reasonably informative partial before protecting it.
    if partial_words < 4 && partial_chars < 24 {
        return false;
    }

    // Clear shrinkage: final retained less than ~60% of both word and character
    // content, or collapsed to <=2 words while the partial had a full phrase.
    (final_words * 100 < partial_words * 60
        && final_chars * 100 < partial_chars * 65)
        || (final_words <= 2 && partial_words >= 6)
}

/// Remove audio-overlap text between two finalized Whisper live windows.
/// We require at least two matching words, or one long word, to avoid deleting
/// legitimate short repetitions such as "no, no".
fn strip_whisper_cross_segment_overlap(prev: &str, next: &str) -> String {
    fn normalize_word(word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '’')
            .flat_map(|c| c.to_lowercase())
            .collect()
    }

    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let next_words: Vec<&str> = next.split_whitespace().collect();
    if prev_words.is_empty() || next_words.is_empty() {
        return next.trim().to_string();
    }

    let max_overlap = 12usize.min(prev_words.len()).min(next_words.len());
    for len in (1..=max_overlap).rev() {
        let prev_slice = &prev_words[prev_words.len() - len..];
        let next_slice = &next_words[..len];
        let matches = prev_slice
            .iter()
            .zip(next_slice.iter())
            .all(|(a, b)| normalize_word(a) == normalize_word(b));
        if !matches {
            continue;
        }

        let safe_single_word = len == 1 && normalize_word(next_slice[0]).chars().count() >= 6;
        if len >= 2 || safe_single_word {
            return next_words[len..].join(" ").trim().to_string();
        }
    }

    next.trim().to_string()
}

/// Transcribe audio chunk using the Sherpa-ONNX provider
async fn transcribe_chunk_with_provider<R: Runtime>(
    engine: &TranscriptionEngine,
    chunk: AudioChunk,
    app: &AppHandle<R>,
) -> std::result::Result<(String, Option<f32>, bool), TranscriptionError> {
    let transcription_data = if chunk.sample_rate != 16000 {
        crate::audio::audio_processing::resample_audio(&chunk.data, chunk.sample_rate, 16000)
    } else {
        chunk.data
    };

    let speech_samples = transcription_data;

    if speech_samples.is_empty() {
        warn!("Audio chunk {} is empty, skipping transcription", chunk.chunk_id);
        return Err(TranscriptionError::AudioTooShort {
            samples: 0,
            minimum: 1600,
        });
    }

    let energy: f32 =
        speech_samples.iter().map(|&x| x * x).sum::<f32>() / speech_samples.len() as f32;
    info!(
        "Processing speech audio chunk {} with {} samples (energy: {:.6})",
        chunk.chunk_id,
        speech_samples.len(),
        energy
    );

    match engine {
        TranscriptionEngine::Provider(provider) => {
            let language = crate::get_language_preference_internal();

            match provider.transcribe(speech_samples, language).await {
                Ok(result) => {
                    let cleaned_text = result.text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok((String::new(), result.confidence, result.is_partial));
                    }

                    info!(
                        "{} transcription complete for chunk {}: '{}'",
                        provider.provider_name(),
                        chunk.chunk_id,
                        cleaned_text
                    );

                    Ok((cleaned_text, result.confidence, result.is_partial))
                }
                Err(e) => {
                    error!("Transcription failed for chunk {}: {}", chunk.chunk_id, e);
                    let _ = app.emit("transcription-error", &serde_json::json!({
                        "error": e.to_string(),
                        "userMessage": format!("Transcription failed: {}", e),
                        "actionable": false
                    }));
                    Err(e)
                }
            }
        }
    }
}

#[cfg(test)]
mod live_whisper_tests {
    use super::{should_keep_partial_over_final, strip_whisper_cross_segment_overlap};

    #[test]
    fn blocks_obvious_final_regression() {
        assert!(should_keep_partial_over_final(
            "This is a very important opportunity for you to learn the important things.",
            "obviously"
        ));
        assert!(should_keep_partial_over_final(
            "E viene usato per salutare una persona o piu persone.",
            "una persona"
        ));
    }

    #[test]
    fn accepts_normal_final_correction() {
        assert!(!should_keep_partial_over_final(
            "Vorrei sapere se possiamo modificare questa machina",
            "Vorrei sapere se possiamo modificare questa macchina"
        ));
        assert!(!should_keep_partial_over_final(
            "Ciao come stai",
            "Ciao, come stai?"
        ));
    }

    #[test]
    fn strips_two_word_overlap() {
        assert_eq!(
            strip_whisper_cross_segment_overlap(
                "Vorrei sapere se possiamo modificare questa macchina",
                "questa macchina per migliorare la produzione"
            ),
            "per migliorare la produzione"
        );
    }

    #[test]
    fn keeps_short_legitimate_repeat() {
        assert_eq!(
            strip_whisper_cross_segment_overlap("No", "No non e corretto"),
            "No non e corretto"
        );
    }

    #[test]
    fn strips_long_single_word_overlap() {
        assert_eq!(
            strip_whisper_cross_segment_overlap(
                "dobbiamo controllare alimentazione",
                "alimentazione della macchina"
            ),
            "della macchina"
        );
    }
}

/// Format current timestamp (wall-clock time)
fn format_current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();

    let hours = (now.as_secs() / 3600) % 24;
    let minutes = (now.as_secs() / 60) % 60;
    let seconds = now.as_secs() % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}
