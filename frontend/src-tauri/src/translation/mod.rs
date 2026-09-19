// translation/mod.rs
//
// Local translation module (OPUS-MT, zh ⇄ en).
// Provides:
// - lazy-loaded per-direction engines (`get_engine`)
// - the `translate_text` command for the translation page
// - realtime hooks: queue_translation + process_pending_translations, which
//   emit `translate-update` events consumed by the frontend transcript view.

pub mod commands;
pub mod engine;
pub mod llm;

use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use tauri::{AppHandle, Emitter, Runtime};

use engine::OpusMtEngine;

pub const MODEL_DIR_ZH_EN: &str = "opus-mt-zh-en";
pub const MODEL_DIR_EN_ZH: &str = "opus-mt-en-zh";

/// Realtime inline translation master switch (default off).
pub static TRANSLATION_ENABLED: AtomicBool = AtomicBool::new(false);

/// Realtime translation target language: 13 种语言代码之一（default "en"，
/// 对应默认 home "zh" 的默认目标）。允许等于 HOME_LANG（此时源语言==目标语言
/// 的段落直接跳过/返回原文）。
pub(crate) static TARGET_LANG: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new("en".to_string()));

/// Home 语言（用户的母语/主要工作语言），取值限 en/zh/ko/ja（default "zh"）。
/// 仅用于推导默认目标语言（home != "en" → "en"，home == "en" → "zh"），
/// 不参与实时方向解析。
pub(crate) static HOME_LANG: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new("zh".to_string()));

/// 翻译引擎选择："opus"（中英快速）| "hymt2"（高质量多语言 LLM）。
pub(crate) static TRANSLATION_ENGINE: LazyLock<Mutex<String>> =
    LazyLock::new(|| Mutex::new("opus".to_string()));

/// 当前翻译引擎 id（"opus" | "hymt2"）。
pub fn current_engine() -> String {
    TRANSLATION_ENGINE
        .lock()
        .map(|e| e.clone())
        .unwrap_or_else(|_| "opus".to_string())
}

/// 当前目标语言设置（13 种语言代码之一）。
pub fn target_lang() -> String {
    TARGET_LANG
        .lock()
        .map(|t| t.clone())
        .unwrap_or_else(|_| "en".to_string())
}

/// 当前 home 语言（en/zh/ko/ja）。
pub fn home_lang() -> String {
    HOME_LANG
        .lock()
        .map(|h| h.clone())
        .unwrap_or_else(|_| "zh".to_string())
}

/// home 语言对应的默认目标语言：home != "en" → "en"，home == "en" → "zh"。
pub fn default_target_for_home(home: &str) -> String {
    if home == "en" { "zh".to_string() } else { "en".to_string() }
}

// ── Engines (lazy-loaded) ─────────────────────────────────────────────────────

static ZH_EN_ENGINE: LazyLock<Mutex<Option<Arc<OpusMtEngine>>>> = LazyLock::new(|| Mutex::new(None));
static EN_ZH_ENGINE: LazyLock<Mutex<Option<Arc<OpusMtEngine>>>> = LazyLock::new(|| Mutex::new(None));

fn model_dir(name: &str) -> PathBuf {
    crate::sherpa_onnx_engine::commands::resolved_models_dir().join(name)
}

pub fn get_engine(direction: &str) -> Result<Arc<OpusMtEngine>, String> {
    let (slot, dir_name) = match direction {
        "zh-en" => (&ZH_EN_ENGINE, MODEL_DIR_ZH_EN),
        "en-zh" => (&EN_ZH_ENGINE, MODEL_DIR_EN_ZH),
        other => return Err(format!("不支持的翻译方向: {}", other)),
    };

    let mut guard = slot.lock().map_err(|e| e.to_string())?;
    if let Some(engine) = guard.as_ref() {
        return Ok(engine.clone());
    }

    let dir = model_dir(dir_name);
    crate::llama_sidecar::emit_model_loading(dir_name, "start", None, None);
    let start = std::time::Instant::now();
    let engine = match OpusMtEngine::load(&dir) {
        Ok(engine) => engine,
        Err(e) => {
            let msg = e.to_string();
            crate::llama_sidecar::emit_model_loading(dir_name, "error", None, Some(msg.clone()));
            return Err(msg);
        }
    };
    crate::llama_sidecar::emit_model_loading(
        dir_name,
        "done",
        Some(start.elapsed().as_millis() as u64),
        None,
    );
    let engine = Arc::new(engine);
    *guard = Some(engine.clone());
    log::info!("Translation engine ready: {} ({})", direction, dir.display());
    Ok(engine)
}

/// Unload both OPUS-MT direction engines, freeing their memory (called when
/// switching to a different translation engine).
pub fn unload_opus_engines() {
    for (slot, direction) in [
        (&ZH_EN_ENGINE, "zh-en"),
        (&EN_ZH_ENGINE, "en-zh"),
    ] {
        if let Ok(mut guard) = slot.lock() {
            if guard.take().is_some() {
                log::info!("OPUS-MT 引擎已卸载 ({})，内存已释放", direction);
            }
        }
    }
}

/// Whether the given direction's model files are present on disk.
pub fn is_model_installed(direction: &str) -> bool {
    let dir_name = match direction {
        "zh-en" => MODEL_DIR_ZH_EN,
        "en-zh" => MODEL_DIR_EN_ZH,
        _ => return false,
    };
    let dir = model_dir(dir_name);
    engine::REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

// ── Language detection (CJK share heuristic) ──────────────────────────────────

/// Rough zh/en detection: true when CJK characters make up > 30% of the text.
pub fn is_chinese_dominant(text: &str) -> bool {
    let total = text.chars().filter(|c| !c.is_whitespace()).count();
    if total == 0 {
        return false;
    }
    let cjk = text
        .chars()
        .filter(|c| {
            let u = *c as u32;
            (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u)
        })
        .count();
    cjk * 10 > total * 3
}

/// 粗略源语言检测：含 hangul（0xAC00-0xD7AF、0x1100-0x11FF）→ "ko"，
/// 含假名（0x3040-0x30FF）→ "ja"，汉字占比 > 30% → "zh"，否则 → "en"。
pub fn detect_source_lang(text: &str) -> &'static str {
    let mut has_hangul = false;
    let mut has_kana = false;
    for c in text.chars() {
        let u = c as u32;
        if (0xAC00..=0xD7AF).contains(&u) || (0x1100..=0x11FF).contains(&u) {
            has_hangul = true;
        } else if (0x3040..=0x30FF).contains(&u) {
            has_kana = true;
        }
        if has_hangul && has_kana {
            break;
        }
    }
    if has_hangul {
        "ko"
    } else if has_kana {
        "ja"
    } else if is_chinese_dominant(text) {
        "zh"
    } else {
        "en"
    }
}

fn looks_like_italian(text: &str) -> bool {
    let lower = text.to_lowercase();
    let padded = format!(" {} ", lower.replace(|c: char| !c.is_alphabetic() && c != 'à' && c != 'è' && c != 'é' && c != 'ì' && c != 'ò' && c != 'ù', " "));
    const MARKERS: &[&str] = &[
        " ciao ", " sono ", " non ", " una ", " oggi ", " quindi ", " adesso ",
        " molto ", " giornata ", " lavorare ", " rilassarmi ", " fatto ", " ancora ",
        " questo ", " questa ", " momento ", " fuori ", " perché ", " però ", " così ",
    ];
    let score = MARKERS.iter().filter(|m| padded.contains(**m)).count();
    score >= 3 || (score >= 2 && lower.contains('è'))
}

fn hymt2_source_lang(text: &str, asr_hint: Option<&str>) -> String {
    // Explicit script evidence wins over a fixed ASR hint. This matters in
    // mixed meetings: Whisper may be configured as Italian while a Chinese
    // participant speaks for one segment.
    let detected = detect_source_lang(text);
    if matches!(detected, "zh" | "ja" | "ko") {
        return detected.to_string();
    }

    // A fixed ASR language remains authoritative for Latin-script languages.
    if let Some(lang @ ("it" | "fr" | "de" | "es" | "pt" | "ru" | "en")) = asr_hint {
        return lang.to_string();
    }

    // Whisper Auto has no useful Latin-script hint. The generic detector falls
    // back to English for Latin text, which was causing Italian speech to be
    // logged and prompted as en->zh. Use a conservative Italian heuristic so
    // Auto mode still sends clear Italian speech to Hy-MT2 as it->target.
    if looks_like_italian(text) {
        return "it".to_string();
    }

    detected.to_string()
}

// ── Realtime translation queue ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TranslateTask {
    text: String,
    sequence_id: u64,
    is_partial: bool,
    context_before: Option<String>,
}

static TRANSLATE_QUEUE: LazyLock<Mutex<VecDeque<TranslateTask>>> =
    LazyLock::new(|| Mutex::new(VecDeque::new()));

/// 已入队翻译的段落 sequence_id 集合：开启翻译时据此补译未入队的已提交段落。
static TRANSLATE_SEEN: LazyLock<Mutex<HashSet<u64>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// 上一个 committed final 原文。仅给 1-2 词的 Hy-MT2 final 做词义消歧，
/// 不改变字幕时间轴，也不把上下文内容并入当前译文。
static LAST_FINAL_SOURCE: LazyLock<Mutex<Option<String>>> =
    LazyLock::new(|| Mutex::new(None));

fn is_short_context_fragment(text: &str) -> bool {
    let words = text.split_whitespace().filter(|w| !w.trim().is_empty()).count();
    words > 0 && words <= 2 && text.chars().filter(|c| c.is_alphanumeric()).count() <= 24
}

/// 新录音开始（sequence 重置）：清空待译队列与已见集合。
pub fn reset_translation_session() {
    if let Ok(mut q) = TRANSLATE_QUEUE.lock() {
        q.clear();
    }
    if let Ok(mut seen) = TRANSLATE_SEEN.lock() {
        seen.clear();
    }
    if let Ok(mut last) = LAST_FINAL_SOURCE.lock() {
        *last = None;
    }
}

/// 该 sequence_id 是否已入队过翻译（用于补译去重）。
pub fn translation_seen(sequence_id: u64) -> bool {
    TRANSLATE_SEEN
        .lock()
        .map(|s| s.contains(&sequence_id))
        .unwrap_or(false)
}

#[derive(Debug, Clone, Serialize)]
pub struct TranslateUpdate {
    pub sequence_id: u64,
    pub original_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub is_partial: bool,
}

/// 根据目标语言设置与当前引擎解析翻译方向。
/// hymt2 路径规则：源语言==目标语言 → 返回 None（跳过，无需翻译）；
/// 否则译成目标语言（home 语言不参与方向解析）。
/// 返回 (direction, source_lang, effective_target)；目标语言不被当前引擎
/// 支持时也返回 None（跳过）。
fn resolve_direction(text: &str, target: &str) -> Option<(String, String, String)> {
    let engine = current_engine();
    if engine == "hymt2" {
        // Hy-MT2: prefer the explicit ASR language hint.
        // 拉丁字母语言仅靠字符特征无法可靠区分（例如意大利语会被误判为英语），
        // 因此录音场景里 language=it 应直接驱动 it -> target 翻译。
        let asr_hint = crate::get_language_preference_internal()
            .filter(|lang| lang != "auto" && !lang.is_empty());
        let source_lang = hymt2_source_lang(text, asr_hint.as_deref());
        // 兼容存量 "auto"：按 home 的默认目标解析
        let effective_target: String = if target == "auto" {
            default_target_for_home(&home_lang())
        } else {
            target.to_string()
        };
        let target_supported = llm::SUPPORTED_TARGET_LANGS.contains(&effective_target.as_str());
        if source_lang == effective_target || !target_supported {
            return None;
        }
        Some((
            format!("{}-{}", source_lang, effective_target),
            source_lang,
            effective_target,
        ))
    } else {
        // OPUS realtime engine. For Italian we use a stable two-stage path:
        // it -> en, then (when Chinese is requested) en -> zh.
        if !matches!(target, "auto" | "zh" | "en") {
            log::warn!("OPUS-MT 引擎不支持目标语言 {}，跳过翻译", target);
            return None;
        }

        let asr_hint = crate::get_language_preference_internal()
            .filter(|lang| lang != "auto" && !lang.is_empty());
        if asr_hint.as_deref() == Some("it") && !is_chinese_dominant(text) {
            log::warn!("Italian realtime translation requires Hy-MT2; OPUS Italian chain is disabled");
            return None;
        }

        let source_is_zh = is_chinese_dominant(text);
        let (direction, source_lang) =
            if source_is_zh { ("zh-en", "zh") } else { ("en-zh", "en") };
        let effective_target: &str = if target == "auto" {
            if source_is_zh { "en" } else { "zh" }
        } else {
            target
        };
        if source_lang == effective_target {
            None
        } else {
            Some((
                direction.to_string(),
                source_lang.to_string(),
                effective_target.to_string(),
            ))
        }
    }
}

/// Queue a committed transcript segment for translation (no-op when disabled).
/// 方向按 resolve_direction 的 home 规则解析；解析为 None 时跳过。
/// 入队后自动触发单例 worker（不阻塞调用方）。
pub fn queue_translation<R: Runtime>(app: &AppHandle<R>, text: &str, sequence_id: u64) {
    if !TRANSLATION_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    let text = text.trim().to_string();
    if text.is_empty() {
        return;
    }
    let target = TARGET_LANG.lock().map(|t| t.clone()).unwrap_or_else(|_| "en".to_string());
    if resolve_direction(&text, &target).is_none() {
        return;
    }

    let engine_kind = current_engine();
    let context_before = if engine_kind == "hymt2" && is_short_context_fragment(&text) {
        LAST_FINAL_SOURCE
            .lock()
            .ok()
            .and_then(|last| last.clone())
    } else {
        None
    };
    if let Ok(mut last) = LAST_FINAL_SOURCE.lock() {
        *last = Some(text.clone());
    }

    if let Ok(mut q) = TRANSLATE_QUEUE.lock() {
        // A committed final is authoritative. Drop stale previews for the same
        // row and prioritize finals ahead of all queued partial previews while
        // preserving FIFO order among finals.
        q.retain(|task| !(task.sequence_id == sequence_id && task.is_partial));
        let insert_at = q.iter().position(|task| task.is_partial).unwrap_or(q.len());
        q.insert(
            insert_at,
            TranslateTask {
                text,
                sequence_id,
                is_partial: false,
                context_before,
            },
        );
        log::info!(
            "Translation queued: seq={} queue_len={} engine={} target={} contextual={}",
            sequence_id,
            q.len(),
            engine_kind,
            target,
            q.get(insert_at).and_then(|task| task.context_before.as_ref()).is_some()
        );
    }
    if let Ok(mut seen) = TRANSLATE_SEEN.lock() {
        seen.insert(sequence_id);
    }
    kick_translation_worker(app);
}

/// Queue a replaceable realtime preview translation. Multiple pending previews
/// for the same sequence are coalesced so Hy-MT2 cannot build an ever-growing
/// backlog while somebody speaks continuously.
pub fn queue_partial_translation<R: Runtime>(
    app: &AppHandle<R>,
    text: &str,
    sequence_id: u64,
) {
    if !TRANSLATION_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    // Hy-MT2 generation is serialized through a single llama-helper process.
    // A Whisper partial is replaceable, but once generation has started it
    // cannot be preempted by a later committed final. On longer Italian
    // utterances that can block the translation worker for tens of seconds
    // (or until max_tokens), starving all following final segments.
    //
    // Therefore Hy-MT2 realtime translation is final-only. Whisper still
    // updates the source transcript live; committed segments are translated
    // immediately by queue_translation(). OPUS previews remain enabled because
    // that engine is much cheaper and does not occupy the llama sidecar.
    if current_engine() == "hymt2" {
        return;
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        return;
    }
    let target = TARGET_LANG
        .lock()
        .map(|t| t.clone())
        .unwrap_or_else(|_| "en".to_string());
    let Some((direction, _, _)) = resolve_direction(&text, &target) else {
        return;
    };
    if let Ok(mut q) = TRANSLATE_QUEUE.lock() {
        // Never let preview work delay committed translations.
        if q.iter().any(|task| !task.is_partial) {
            return;
        }

        // Realtime preview is best-effort: keep only the newest pending partial
        // across the whole meeting. Older previews are obsolete once newer ASR
        // text exists and would only create translation backlog.
        q.retain(|task| !task.is_partial);
        q.push_back(TranslateTask {
            text,
            sequence_id,
            is_partial: true,
            context_before: None,
        });
    }
    kick_translation_worker(app);
}

/// 最终译文单例 worker：队列串行消费，fire-and-forget，不阻塞调用方
/// （修复长段 beam 翻译串行阻塞 X-ASR 轮询循环的问题）。
static FINAL_WORKER_RUNNING: AtomicBool = AtomicBool::new(false);

/// 触发最终译文 worker（幂等）：已有 worker 在跑时直接返回。
pub fn kick_translation_worker<R: Runtime>(app: &AppHandle<R>) {
    if !TRANSLATION_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    if FINAL_WORKER_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            process_pending_translations(app.clone()).await;

            let empty = TRANSLATE_QUEUE.lock().map(|q| q.is_empty()).unwrap_or(true);
            if !empty {
                continue;
            }

            // Release ownership before exiting, then re-check the queue.
            //
            // Lost-wakeup race fixed here:
            // 1) worker observes queue empty while RUNNING=true
            // 2) producer enqueues and kick_translation_worker sees true, so it returns
            // 3) worker stores false and exits
            // The queued task would then sit idle until some later event kicked the
            // worker again. This matched the realtime logs exactly.
            FINAL_WORKER_RUNNING.store(false, Ordering::SeqCst);

            let has_work = TRANSLATE_QUEUE
                .lock()
                .map(|q| !q.is_empty())
                .unwrap_or(false);
            if !has_work {
                return;
            }

            // If no producer already claimed the worker after we released it,
            // reclaim it ourselves and continue draining. If another producer
            // did claim it, that producer has spawned a replacement worker.
            if FINAL_WORKER_RUNNING
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                log::debug!("Translation worker recovered work queued during shutdown race");
                continue;
            }
            return;
        }
    });
}

/// Drain the queue, translate each task and emit `translate-update` events.
/// Blocking inference runs inside; call from async contexts (it spawns the
/// blocking work on the tokio blocking pool).
pub async fn process_pending_translations<R: Runtime>(app: AppHandle<R>) {
    if !TRANSLATION_ENABLED.load(Ordering::SeqCst) {
        if let Ok(mut q) = TRANSLATE_QUEUE.lock() {
            q.clear();
        }
        return;
    }

    let target = TARGET_LANG.lock().map(|t| t.clone()).unwrap_or_else(|_| "en".to_string());

    loop {
        let task = match TRANSLATE_QUEUE.lock() {
            Ok(mut q) => q.pop_front(),
            Err(_) => None,
        };
        let Some(task) = task else { break };

        let Some((direction, source_lang, effective_target)) = resolve_direction(&task.text, &target) else {
            continue;
        };

        let text = task.text.clone();
        let seq = task.sequence_id;
        let task_is_partial = task.is_partial;
        let context_before = task.context_before.clone();
        let engine_kind = current_engine();
        let direction_for_task = direction.clone();
        let app_for_stream = app.clone();
        let original_for_stream = task.text.clone();
        let source_lang_stream = source_lang.clone();
        let target_lang_stream = effective_target.clone();
        let started_at = std::time::Instant::now();
        log::info!(
            "Translation start: seq={} engine={} direction={} chars={}",
            seq,
            engine_kind,
            direction,
            task.text.chars().count()
        );

        let result = tokio::task::spawn_blocking(move || {
            if engine_kind == "hymt2" {
                // Hy-MT2 LLM 引擎：走 llama-helper sidecar，ASR 模式指令；
                // 流式生成，节流 emit 部分译文（原始输出快照，未清洗）
                let mut partial = String::new();
                let mut tokens_since_emit = 0usize;
                let mut last_emit = std::time::Instant::now();
                let mut on_delta = |delta: &str| {
                        partial.push_str(delta);
                        tokens_since_emit += 1;
                        if tokens_since_emit >= 8
                            || last_emit.elapsed() >= std::time::Duration::from_millis(250)
                        {
                            tokens_since_emit = 0;
                            last_emit = std::time::Instant::now();
                            // If a committed final for this sequence arrived
                            // while this preview was generating, the preview is
                            // stale and must not overwrite the final UI state.
                            if task_is_partial && translation_seen(seq) {
                                return;
                            }
                            let update = TranslateUpdate {
                                sequence_id: seq,
                                original_text: original_for_stream.clone(),
                                translated_text: partial.clone(),
                                source_lang: source_lang_stream.clone(),
                                target_lang: target_lang_stream.clone(),
                                is_partial: true,
                            };
                            if let Err(e) = app_for_stream.emit("translate-update", &update) {
                                log::warn!("translate-update (partial) emit failed: {}", e);
                            }
                        }
                    };

                if let Some(context) = context_before.as_deref() {
                    llm::translate_with_context(
                        &text,
                        context,
                        &direction_for_task,
                        Some(&mut on_delta),
                    )
                } else {
                    llm::translate(
                        &text,
                        &direction_for_task,
                        true,
                        Some(&mut on_delta),
                    )
                }
            } else {
                get_engine(&direction_for_task).and_then(|engine| {
                    engine.translate_greedy(&text).map_err(|e| e.to_string())
                })
            }
        })
        .await;

        match result {
            Ok(Ok(translated)) => {
                log::info!(
                    "Translation done: seq={} elapsed_ms={} output_chars={}",
                    seq,
                    started_at.elapsed().as_millis(),
                    translated.chars().count()
                );
                if task_is_partial && translation_seen(seq) {
                    log::debug!(
                        "Dropping stale partial translation for seq={} because final is queued",
                        seq
                    );
                    continue;
                }
                let update = TranslateUpdate {
                    sequence_id: seq,
                    original_text: task.text.clone(),
                    translated_text: translated,
                    source_lang,
                    target_lang: effective_target,
                    is_partial: task_is_partial,
                };
                if let Err(e) = app.emit("translate-update", &update) {
                    log::warn!("translate-update emit failed: {}", e);
                }
            }
            Ok(Err(e)) => {
                log::warn!(
                    "Translation failed: seq={} elapsed_ms={} error={}",
                    seq,
                    started_at.elapsed().as_millis(),
                    e
                );
            }
            Err(e) => {
                log::warn!(
                    "Translation task join error: seq={} elapsed_ms={} error={}",
                    seq,
                    started_at.elapsed().as_millis(),
                    e
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_context_fragment_detection_is_conservative() {
        assert!(is_short_context_fragment("palestra."));
        assert!(is_short_context_fragment("Ho fatto"));
        assert!(!is_short_context_fragment("forse potrei andare in palestra"));
        assert!(!is_short_context_fragment(""));
    }

    #[test]
    fn final_tasks_are_prioritized_ahead_of_partials() {
        let mut q = VecDeque::new();
        q.push_back(TranslateTask {
            text: "preview".into(),
            sequence_id: 1,
            is_partial: true,
            context_before: None,
        });
        let insert_at = q.iter().position(|task| task.is_partial).unwrap_or(q.len());
        q.insert(
            insert_at,
            TranslateTask {
                text: "final".into(),
                sequence_id: 1,
                is_partial: false,
                context_before: None,
            },
        );
        assert!(!q[0].is_partial);
        assert!(q[1].is_partial);
    }

    #[test]
    fn newest_partial_replaces_older_pending_preview() {
        let mut q = VecDeque::new();
        q.push_back(TranslateTask {
            text: "old preview".into(),
            sequence_id: 1,
            is_partial: true,
            context_before: None,
        });
        q.retain(|task| !task.is_partial);
        q.push_back(TranslateTask {
            text: "new preview".into(),
            sequence_id: 2,
            is_partial: true,
            context_before: None,
        });
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].sequence_id, 2);
    }

    #[test]
    fn detect_source_lang_hangul_is_ko() {
        assert_eq!(detect_source_lang("안녕하세요, hello"), "ko");
        // 韩文音节块之外的 hangul jamo 也算
        assert_eq!(detect_source_lang("한국어"), "ko");
    }

    #[test]
    fn detect_source_lang_kana_is_ja() {
        assert_eq!(detect_source_lang("これはテストです"), "ja");
        assert_eq!(detect_source_lang("カタカナ"), "ja");
    }

    #[test]
    fn detect_source_lang_hanzi_dominant_is_zh() {
        assert_eq!(detect_source_lang("今天天气真不错，我们出去走走"), "zh");
    }

    #[test]
    fn detect_source_lang_latin_is_en() {
        assert_eq!(detect_source_lang("Hello, this is a test."), "en");
        assert_eq!(detect_source_lang("Bonjour le monde"), "en");
        // 汉字占比不足 30% 时回落 en
        assert_eq!(detect_source_lang("abcdefgh 中"), "en");
    }

    #[test]
    fn hymt2_source_lang_script_overrides_fixed_italian_hint() {
        assert_eq!(hymt2_source_lang("这是中国工程师的回答", Some("it")), "zh");
        assert_eq!(hymt2_source_lang("これは日本語です", Some("it")), "ja");
        assert_eq!(hymt2_source_lang("안녕하세요", Some("it")), "ko");
    }

    #[test]
    fn hymt2_source_lang_uses_italian_hint_for_latin_text() {
        assert_eq!(
            hymt2_source_lang("Questa e una prova di traduzione.", Some("it")),
            "it"
        );
        assert_eq!(
            hymt2_source_lang("Questa e una prova di traduzione.", None),
            "en"
        );
    }

    // resolve_direction 测试需要改全局静态（引擎/home），用互斥锁串行化，
    // 避免用例间互相干扰。
    static RESOLVE_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_engine_home<T>(engine: &str, home: &str, f: impl FnOnce() -> T) -> T {
        let _guard = RESOLVE_TEST_LOCK.lock().unwrap();
        let saved_engine = current_engine();
        let saved_home = home_lang();
        *TRANSLATION_ENGINE.lock().unwrap() = engine.to_string();
        *HOME_LANG.lock().unwrap() = home.to_string();
        let result = f();
        *TRANSLATION_ENGINE.lock().unwrap() = saved_engine;
        *HOME_LANG.lock().unwrap() = saved_home;
        result
    }

    #[test]
    fn resolve_direction_source_equals_target_skips() {
        // target=en，英文输入：源==目标 → None（跳过，无需翻译）
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("Hello, this is a test.", "en")
        });
        assert_eq!(r, None);
        // target=zh，中文输入：同样跳过
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("今天天气真不错，我们出去走走", "zh")
        });
        assert_eq!(r, None);
    }

    #[test]
    fn resolve_direction_translates_to_target() {
        // target=en，中文/日语输入 → 译成目标语言（home 不参与）
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("今天天气真不错，我们出去走走", "en")
        });
        assert_eq!(r, Some(("zh-en".to_string(), "zh".to_string(), "en".to_string())));
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("これはテストです", "en")
        });
        assert_eq!(r, Some(("ja-en".to_string(), "ja".to_string(), "en".to_string())));
        // home=en 时规则不变：target=zh，英文输入 → en-zh
        let r = with_engine_home("hymt2", "en", || {
            resolve_direction("Hello, this is a test.", "zh")
        });
        assert_eq!(r, Some(("en-zh".to_string(), "en".to_string(), "zh".to_string())));
    }

    #[test]
    fn resolve_direction_legacy_auto_uses_default_target() {
        // 存量 "auto" 按 home 的默认目标解析（home=zh → 默认目标 en）
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("Hello, this is a test.", "auto")
        });
        assert_eq!(r, None); // 源 en == 默认目标 en → 跳过
        let r = with_engine_home("hymt2", "zh", || {
            resolve_direction("今天天气真不错，我们出去走走", "auto")
        });
        assert_eq!(r, Some(("zh-en".to_string(), "zh".to_string(), "en".to_string())));
        // home=en → 默认目标 zh；中文输入源==目标 → 跳过，英文输入 → en-zh
        let r = with_engine_home("hymt2", "en", || {
            resolve_direction("今天天气真不错，我们出去走走", "auto")
        });
        assert_eq!(r, None);
        let r = with_engine_home("hymt2", "en", || {
            resolve_direction("Hello, this is a test.", "auto")
        });
        assert_eq!(r, Some(("en-zh".to_string(), "en".to_string(), "zh".to_string())));
    }

    #[test]
    fn default_target_for_home_rules() {
        assert_eq!(default_target_for_home("zh"), "en");
        assert_eq!(default_target_for_home("ja"), "en");
        assert_eq!(default_target_for_home("ko"), "en");
        assert_eq!(default_target_for_home("en"), "zh");
    }
}
