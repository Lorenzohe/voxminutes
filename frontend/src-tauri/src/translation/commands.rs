// translation/commands.rs
//
// Tauri commands for translation.

use std::sync::atomic::Ordering;

use crate::database::repositories::setting::SettingsRepository;
use crate::state::AppState;

use super::{current_engine, get_engine, is_hymt2_engine, is_model_installed, llm, HOME_LANG, TARGET_LANG, TRANSLATION_ENABLED, TRANSLATION_ENGINE};

/// Translate a block of text. direction: "auto" | "zh-en" | "en-zh"（hymt2 引擎
/// 额外支持任意 "{src}-{tgt}" 语言对，如 "en-ja"、"zh-Hant-en"）。
/// target 为可选的目标语言覆盖（仅 hymt2 引擎生效；不传或 "auto" 时回退到
/// 全局 TARGET_LANG 设置解析）。opus 引擎忽略 target，仅支持 zh ⇄ en。
/// hymt2 引擎下源语言与解析出的目标语言相同时，无需翻译，直接返回原文。
/// request_id 提供时（仅 hymt2 引擎），生成过程中的增量文本以
/// `translate-text-stream` 事件（payload `{request_id, delta}`）推给前端；
/// 返回值始终是完整译文。opus 引擎忽略 request_id。
/// Blocking inference runs on the blocking pool.
#[tauri::command]
pub async fn translate_text(
    app: tauri::AppHandle,
    text: String,
    direction: String,
    target: Option<String>,
    request_id: Option<String>,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }

    let selected_engine = current_engine();
    if is_hymt2_engine(&selected_engine) {
        // Hy-MT2 Q4/Q6 共用同一套最终版翻译逻辑，只切换模型文件。
        if !crate::model_download::hy_mt2_installed_for_engine(&selected_engine) {
            return Err(format!(
                "{} 翻译模型未安装，请先到设置页下载。",
                if selected_engine == "hymt2-q4" { "HY-MT2-Q4" } else { "HY-MT2-Q6" }
            ));
        }
        let explicit = llm::parse_direction(&direction);
        // 源语言：显式方向优先，否则按文本特征检测
        let src = explicit
            .map(|(s, _)| s.to_string())
            .unwrap_or_else(|| super::detect_source_lang(&text).to_string());
        // 目标语言：target 参数 > 显式方向 > 全局设置（兼容存量 "auto"：按 home 默认目标）
        let tgt = match target.filter(|t| t != "auto" && !t.trim().is_empty()) {
            Some(t) => t,
            None => match explicit {
                Some((_, t)) => t.to_string(),
                None => {
                    let global = super::target_lang();
                    if global == "auto" {
                        super::default_target_for_home(&super::home_lang())
                    } else {
                        global
                    }
                }
            },
        };
        if !llm::SUPPORTED_TARGET_LANGS.contains(&tgt.as_str()) {
            return Err(format!("不支持的目标语言: {}", tgt));
        }
        if src == tgt {
            // 源语言与目标语言相同：无需翻译，直接返回原文（与实时路径的跳过语义一致）
            return Ok(text);
        }
        let resolved = format!("{}-{}", src, tgt);
        return tokio::task::spawn_blocking(move || {
            use tauri::Emitter;
            if let Some(request_id) = request_id {
                // 流式：增量文本以 translate-text-stream 事件推给前端
                let app = app.clone();
                llm::translate_for_engine(&text, &resolved, false, &selected_engine, Some(&mut |delta: &str| {
                    let _ = app.emit(
                        "translate-text-stream",
                        serde_json::json!({ "request_id": request_id, "delta": delta }),
                    );
                }))
            } else {
                llm::translate_for_engine(&text, &resolved, false, &selected_engine, None)
            }
        })
        .await
        .map_err(|e| format!("翻译任务失败: {}", e))?;
    }

    // OPUS-MT 引擎：仅 zh ⇄ en，忽略 target 参数
    let resolved: &'static str = match direction.as_str() {
        "auto" => {
            if super::is_chinese_dominant(&text) {
                "zh-en"
            } else {
                "en-zh"
            }
        }
        "zh-en" => "zh-en",
        "en-zh" => "en-zh",
        other => return Err(format!("不支持的翻译方向: {}", other)),
    };

    if !is_model_installed(resolved) {
        return Err("翻译模型未安装，请先到设置页下载 OPUS-MT 翻译模型。".to_string());
    }

    tokio::task::spawn_blocking(move || {
        let engine = get_engine(resolved)?;
        engine.translate(&text).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("翻译任务失败: {}", e))?
}

#[tauri::command]
pub async fn set_translation_engine(
    state: tauri::State<'_, AppState>,
    engine: String,
) -> Result<(), String> {
    // Backward compatibility: the old generic "hymt2" setting was the final
    // Q6 route, so migrate it to the explicit Q6 id.
    let engine = match engine.as_str() {
        "hymt2" => "hymt2-q6".to_string(),
        "opus" | "hymt2-q4" | "hymt2-q6" => engine,
        _ => return Err(format!("不支持的翻译引擎: {}", engine)),
    };

    log::info!("Translation engine: {}", engine);
    {
        let mut guard = TRANSLATION_ENGINE.lock().map_err(|e| e.to_string())?;
        *guard = engine.clone();
    }
    SettingsRepository::set(state.db_manager.pool(), "translation.engine", &engine)
        .await
        .map_err(|e| format!("保存翻译引擎设置失败: {}", e))?;

    // Q4/Q6 share the same final translation pipeline. Switching between them
    // only changes the GGUF path; llama-helper reloads the selected model.
    tauri::async_runtime::spawn(async move {
        let _ = tokio::task::spawn_blocking(move || {
            if is_hymt2_engine(&engine) {
                super::unload_opus_engines();
                if crate::model_download::hy_mt2_installed_for_engine(&engine) {
                    if let Err(e) = llm::warmup_for_engine(&engine) {
                        log::warn!("{} 翻译引擎预热失败: {}", engine, e);
                    }
                }
            } else {
                for direction in ["zh-en", "en-zh"] {
                    if is_model_installed(direction) {
                        if let Err(e) = get_engine(direction) {
                            log::warn!("OPUS-MT 翻译引擎预热失败 ({}): {}", direction, e);
                        }
                    }
                }
            }
        })
        .await;
    });
    Ok(())
}

#[tauri::command]
pub fn get_translation_engine() -> String {
    current_engine()
}

#[tauri::command]
pub async fn set_translation_enabled(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    enabled: bool,
) -> Result<(), String> {
    log::info!("Translation enabled: {}", enabled);
    TRANSLATION_ENABLED.store(enabled, Ordering::SeqCst);

    // 持久化实时翻译开关，避免应用重启后静默恢复为关闭。
    SettingsRepository::set(
        state.db_manager.pool(),
        "translation.enabled",
        if enabled { "true" } else { "false" },
    )
    .await
    .map_err(|e| format!("保存实时翻译开关失败: {}", e))?;

    if enabled {
        // 补译：开启翻译时，把当前录音中已提交但未入队过的段落补进翻译队列
        // （关闭期间提交的段落此前被 queue_translation 直接丢弃）。
        let mut requeued = 0usize;
        for (sequence_id, text) in crate::audio::recording_commands::committed_segment_texts() {
            if !super::translation_seen(sequence_id) {
                super::queue_translation(&app, &text, sequence_id);
                requeued += 1;
            }
        }
        if requeued > 0 {
            log::info!("开启翻译：补译 {} 条已提交段落", requeued);
        }
    }

    Ok(())
}

#[tauri::command]
pub fn get_translation_enabled() -> bool {
    TRANSLATION_ENABLED.load(Ordering::SeqCst)
}

#[tauri::command]
pub async fn set_translation_target_lang(
    state: tauri::State<'_, AppState>,
    lang: String,
) -> Result<(), String> {
    if !llm::SUPPORTED_TARGET_LANGS.contains(&lang.as_str()) {
        return Err(format!("不支持的目标语言: {}", lang));
    }
    log::info!("Translation target language: {}", lang);
    {
        let mut guard = TARGET_LANG.lock().map_err(|e| e.to_string())?;
        *guard = lang.clone();
    }
    // 持久化到设置表，下次启动时读回
    SettingsRepository::set(state.db_manager.pool(), "translation.target_lang", &lang)
        .await
        .map_err(|e| format!("保存目标语言设置失败: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn get_translation_target_lang() -> String {
    super::target_lang()
}

#[tauri::command]
pub async fn set_translation_home_lang(
    state: tauri::State<'_, AppState>,
    lang: String,
) -> Result<(), String> {
    if !matches!(lang.as_str(), "en" | "zh" | "ko" | "ja") {
        return Err(format!("不支持的 home 语言: {}", lang));
    }
    log::info!("Translation home language: {}", lang);
    {
        let mut guard = HOME_LANG.lock().map_err(|e| e.to_string())?;
        *guard = lang.clone();
    }
    // 持久化到设置表，下次启动时读回
    SettingsRepository::set(state.db_manager.pool(), "translation.home_lang", &lang)
        .await
        .map_err(|e| format!("保存 home 语言设置失败: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn get_translation_home_lang() -> String {
    super::home_lang()
}
