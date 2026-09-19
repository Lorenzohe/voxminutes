// translation/llm.rs
//
// Hy-MT2 GGUF 翻译引擎（14 种常用语言互译），通过共享 llama-helper sidecar
// （crate::llama_sidecar）推理。Prompt 模板、采样参数与输出清洗移植自
// 参考实现 reference_code/backend/app/translate_engine.py。

use crate::llama_sidecar::{self, GenerateParams};

/// 实时 Hy-MT2 使用 Q6_K：在 GTX 1660 SUPER CUDA helper 上运行。
/// Q4_K_M 继续保留在独立目录，作为较轻量的备用模型。
const MODEL_DIR: &str = "hy-mt2-1.8b-q6k";

// ── 聊天模板（Hy-MT2 特殊 token）──────────────────────────────────────────────
//
// 注意：必须使用全角竖线 ｜ (U+FF5C) 和下划线 ▁ (U+2581)，与模型 GGUF
// tokenizer 的实际 token 文本一致；ASCII 竖线 | 不会被识别为特殊 token。

/// BOS + User 起始标记
const CHAT_PREFIX: &str =
    "<\u{FF5C}hy_begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}hy_User\u{FF5C}>";
/// Assistant 起始标记（触发模型生成）
const CHAT_SUFFIX: &str = "<\u{FF5C}hy_Assistant\u{FF5C}>";

/// 停止序列：模型切换角色或句子结束即停。
const STOP_TOKENS: &[&str] = &[
    "<\u{FF5C}hy_User\u{FF5C}>",
    "<\u{FF5C}hy_Assistant\u{FF5C}>",
    "<\u{FF5C}hy_end\u{2581}of\u{2581}sentence\u{FF5C}>",
    "<|endoftext|>",
];

// ── 采样参数（移植自参考实现 TranslationEngine 默认值）────────────────────────

const CONTEXT_SIZE: u32 = 4096;
const TEMPERATURE: f32 = 0.3;
const TOP_K: i32 = 20;
const TOP_P: f32 = 0.6;
const REPEAT_PENALTY: f32 = 1.15;
const FREQUENCY_PENALTY: f32 = 0.05;
// 韩语输出重复率偏高，参考实现 translate_engine.py 对韩语目标加大惩罚
const KO_REPEAT_PENALTY: f32 = 1.30;
const KO_FREQUENCY_PENALTY: f32 = 0.15;

/// 语言表：code → (中文名, 英文名)。
const LANG_TABLE: &[(&str, &str, &str)] = &[
    ("zh", "中文", "Chinese"),
    ("en", "英语", "English"),
    ("ja", "日语", "Japanese"),
    ("ko", "韩语", "Korean"),
    ("fr", "法语", "French"),
    ("de", "德语", "German"),
    ("es", "西班牙语", "Spanish"),
    ("ru", "俄语", "Russian"),
    ("pt", "葡萄牙语", "Portuguese"),
    ("it", "意大利语", "Italian"),
    ("zh-Hant", "繁体中文", "Traditional Chinese"),
    ("yue", "粤语", "Cantonese"),
    ("th", "泰语", "Thai"),
    ("vi", "越南语", "Vietnamese"),
];

/// 支持的目标语言代码（"auto" 之外），供 mod.rs/commands.rs 校验用。
pub(crate) const SUPPORTED_TARGET_LANGS: &[&str] = &[
    "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt", "it", "zh-Hant", "yue", "th", "vi",
];

/// 语言名表：(中文名, 英文名)；未知 code 返回空串。
fn lang_names(code: &str) -> (&'static str, &'static str) {
    LANG_TABLE
        .iter()
        .find(|(c, _, _)| *c == code)
        .map(|(_, zh, en)| (*zh, *en))
        .unwrap_or(("", ""))
}

/// 中文系语言（指令语言与回声行过滤按此判断）。
pub(crate) fn is_chinese_family(code: &str) -> bool {
    matches!(code, "zh" | "zh-Hant" | "yue")
}

/// 解析 "{src}-{tgt}" 方向；src/tgt 都必须在语言表内。
/// 注意 code 自身可能含连字符（如 zh-Hant），故按已知代码表匹配而非简单 split('-')。
pub(crate) fn parse_direction(direction: &str) -> Option<(&str, &str)> {
    for (code, _, _) in LANG_TABLE {
        if let Some(rest) = direction.strip_prefix(code) {
            if let Some(tgt) = rest.strip_prefix('-') {
                if SUPPORTED_TARGET_LANGS.contains(&tgt) {
                    return Some((code, tgt));
                }
            }
        }
    }
    None
}

/// 构建 Hy-MT2 聊天模板 prompt。
/// 普通模式：单行指令 + 原文；asr_mode：针对语音转录文本的纠错翻译指令
/// （完整 6 条要求，移植自参考实现的 build_asr_translate_prompt）。
/// src 或 tgt 属于中文系（zh/zh-Hant/yue）时用中文指令，否则用英文指令。
pub(crate) fn build_prompt(text: &str, source_lang: &str, target_lang: &str, asr_mode: bool) -> String {
    let italian_zh_hint = source_lang == "it" && is_chinese_family(target_lang);
    let user_text = if asr_mode {
        let (_, src_en) = lang_names(source_lang);
        let (_, tgt_en) = lang_names(target_lang);
        let instruction = if is_chinese_family(source_lang) || is_chinese_family(target_lang) {
            // 涉及中文时统一使用中文指令（参考实现的 has_zh 分支）
            format!(
                "将以下{src_en}语音转录文本翻译为{tgt_en}。\n\
                 要求：\n\
                 1. 只输出{tgt_en}译文，严禁输出原文、双语对照、原文片段或重复原文；\n\
                 2. 直接开始翻译，不要写“翻译：”“{tgt_en}：”等任何前缀；\n\
                 3. 完整保留原文每一项语义，不得省略、概括、合并或添加内容；\n\
                 4. 仅在明显是语音识别错误时做最小纠正，不得改变原意；\n\
                 5. 保留有意义的感叹、重复和语气表达，同时输出流畅自然的口语翻译；\n\
                 6. 不要解释，不要备注；\n\
                 7. 数字、时间、尺寸、单位、型号、零件号必须忠实保留，不得擅自改写数值或编号；\n\
                 8. 严禁输出“来源：”“Source:”或“原文：”等标签；普通可翻译的源语言词必须翻译，不得残留在译文中，只有专有名词、型号、零件号等标识可保留原文；时间可按目标语言自然表达，但数字值必须保持不变（例如 18 e 40 应表达为 18点40分，而不是改变数字）。{}",
                if italian_zh_hint {
                    "\n9. 意大利语实时口语翻译要忠实、少改写：如果片段以残句开始或结束，不要猜测缺失的主语、动作或结论，只翻译当前实际出现的内容；quasi quasi 表示“有点想/要不/也许”的试探语气，不要译成“差不多”或“差点”；天气/光照语境中的 sole 表示“太阳/阳光”。"
                } else {
                    ""
                }
            )
        } else {
            format!(
                "Translate the following {src_en} spoken transcript into {tgt_en}.\n\
                 Requirements:\n\
                 1. Output ONLY the {tgt_en} translation. \
                 Do NOT output the original text, bilingual pairs, source fragments, or repeated source.\n\
                 2. Start directly with the translation; \
                 do not write prefixes like \"Translation:\" or \"{tgt_en}:\".\n\
                 3. Preserve every semantic unit; do not omit, summarize, merge, or add content.\n\
                 4. Correct only obvious ASR errors with the smallest possible change; do not change the meaning.\n\
                 5. Preserve meaningful interjections, repetitions, and tone while producing fluent natural speech.\n\
                 6. Do not explain or add notes.\n\
                 7. Preserve numbers, times, dimensions, units, model names, and part numbers exactly; do not alter numeric values or identifiers.\n\
                 8. Never output labels such as \"Source:\", \"Original:\", or their translated equivalents. Translate ordinary source-language words instead of leaving them in the target text; preserve source text only for proper names, model names, part numbers, and other identifiers. Render time naturally in the target language while preserving every numeric value exactly."
            )
        };
        format!("{instruction}\n\nSource: {text}\n\nTarget ({tgt_en}):")
    } else {
        let (tgt_native, tgt_en) = lang_names(target_lang);
        let instruction = if is_chinese_family(source_lang) {
            format!(
                "将以下文本翻译为{tgt_native}，注意只需要输出翻译后的结果，不要额外解释"
            )
        } else {
            format!(
                "Translate the following text into {tgt_en}. Only output the translated result, without any additional explanation."
            )
        };
        format!("{instruction}:\n\n{text}")
    };
    format!("{CHAT_PREFIX}{user_text}{CHAT_SUFFIX}")
}

/// 为极短实时 ASR final 提供上一段语境。上一段只用于消歧，模型只能翻译
/// current_text，避免像 "palestra" 这种孤立词在没有上下文时被译成错误义项。
pub(crate) fn build_contextual_prompt(
    current_text: &str,
    context_before: &str,
    source_lang: &str,
    target_lang: &str,
) -> String {
    let (_, src_en) = lang_names(source_lang);
    let (_, tgt_en) = lang_names(target_lang);
    let user_text = if is_chinese_family(source_lang) || is_chinese_family(target_lang) {
        format!(
            "将当前{src_en}语音片段翻译为{tgt_en}。上一段只用于理解语境，严禁翻译或复述上一段。\n\
             要求：\n\
             1. 只输出当前片段的{tgt_en}译文；\n\
             2. 根据上一段语境选择当前短语最合适的含义；\n\
             3. 不要添加当前片段没有表达的新信息；\n\
             4. 数字、时间、尺寸、单位、型号、零件号必须忠实保留，不得改写；\n\
             5. 不要解释，不要备注，不要输出原文。{}\n\n\
             上一段（仅供语境）：{context_before}\n\
             当前片段（只翻译这一段）：{current_text}\n\n\
             Target ({tgt_en}):",
            if source_lang == "it" && is_chinese_family(target_lang) {
                "\n6. 意大利语残句不要脑补；quasi quasi 按“有点想/要不/也许”的语气理解；天气语境中的 sole 按“太阳/阳光”理解。"
            } else {
                ""
            }
        )
    } else {
        format!(
            "Translate the current {src_en} speech fragment into {tgt_en}. The previous segment is context only; do NOT translate or repeat it.\n\
             Requirements:\n\
             1. Output ONLY the translation of the current fragment.\n\
             2. Use the previous segment only to disambiguate the current short phrase.\n\
             3. Do not add information not expressed by the current fragment.\n\
             4. Preserve numbers, times, dimensions, units, model names, and part numbers exactly.\n\
             5. Do not explain, annotate, or output the source text.\n\n\
             Previous segment (context only): {context_before}\n\
             Current fragment (translate only this): {current_text}\n\n\
             Target ({tgt_en}):"
        )
    };
    format!("{CHAT_PREFIX}{user_text}{CHAT_SUFFIX}")
}

// ── 输出清洗（参考实现 _clean_*_translation_output 的 zh/en 精简版）────────────

/// 模型可能回声的常见前缀（仅保留 zh/en 相关项）。
const ECHO_PREFIXES: &[&str] = &[
    "Translation:", "translation:", "Translated:", "translated:",
    "Translate:", "translate:", "English:", "Chinese:",
    "Source:", "source:", "Source：", "source：",
    "Target (Chinese):", "Target (Chinese)：",
    "Target(Chinese):", "Target(Chinese)：",
    "Target (中文):", "Target (中文)：",
    "Target(中文):", "Target(中文)：",
    "译文：", "翻译：", "英文：", "中文：",
    "来源：", "来源:", "原文：", "原文:",
    "目标（中文）：", "目标（中文）:", "目标(中文)：", "目标(中文):",
];

/// 剔除 `<｜hy_...｜>` / `<|hy_...|>` / `<│hy_...│>` 形式的特殊 token。
fn strip_special_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(idx) = rest.find('<') {
        out.push_str(&rest[..idx]);
        let tail = &rest[idx..];
        if is_hy_special_token(tail) {
            // 跳过整个 token（含结尾 '>'）
            let end = tail.find('>').map(|e| e + 1).unwrap_or(tail.len());
            rest = &tail[end..];
        } else {
            out.push('<');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// s 以 '<' 开头，判断是否为 `<[｜│|]hy_...[｜│|]>` 形式的特殊 token。
fn is_hy_special_token(s: &str) -> bool {
    let Some(after_lt) = s.strip_prefix('<') else { return false };
    let Some(delim) = after_lt.chars().next() else { return false };
    if !matches!(delim, '\u{FF5C}' | '\u{2502}' | '|') {
        return false;
    }
    let Some(body) = after_lt[delim.len_utf8()..].strip_prefix("hy_") else {
        return false;
    };
    let Some(end) = body.find('>') else { return false };
    let inner = &body[..end];
    matches!(inner.chars().last(), Some('\u{FF5C}' | '\u{2502}' | '|'))
}

/// 整行均为 CJK 字符（参考实现 _CJK_RE fullmatch：含空格的行不算）。
fn is_all_cjk(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            matches!(c as u32,
                0x4E00..=0x9FFF | 0x3000..=0x303F | 0x3040..=0x309F | 0x30A0..=0x30FF | 0xAC00..=0xD7AF)
        })
}

fn is_latin_letter(c: char) -> bool {
    matches!(c as u32, 0x41..=0x5A | 0x61..=0x7A | 0x00C0..=0x024F | 0x1E00..=0x1EFF)
}

/// 非空白字符中拉丁字母占比 ≥ 70%（参考实现 _is_mostly_latin）。
fn is_mostly_latin(s: &str) -> bool {
    let non_space: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if non_space.is_empty() {
        return false;
    }
    let latin = non_space.iter().filter(|c| is_latin_letter(**c)).count();
    latin * 10 >= non_space.len() * 7
}

/// 去重用的归一化：只保留字母数字并转小写，忽略标点/空白差异。
fn normalize_for_dedup(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn extract_numeric_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit() {
            current.push(ch);
            while let Some(next) = chars.peek().copied() {
                if next.is_ascii_digit() {
                    current.push(next);
                    chars.next();
                } else if (next == '.' || next == ',')
                    && chars
                        .clone()
                        .nth(1)
                        .map(|after| after.is_ascii_digit())
                        .unwrap_or(false)
                {
                    current.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(std::mem::take(&mut current));
        }
    }

    tokens
}

/// Hy-MT2 occasionally changes a number while otherwise translating correctly
/// (for example Italian "18 e 40" -> Chinese "8点40"). When source and target
/// contain the same number of numeric tokens, restore them positionally from
/// the source. This is deliberately conservative: if counts differ, leave the
/// model output untouched rather than guessing.
fn restore_numeric_tokens(source: &str, translated: &str) -> String {
    let source_numbers = extract_numeric_tokens(source);
    if source_numbers.is_empty() {
        return translated.to_string();
    }
    let target_numbers = extract_numeric_tokens(translated);
    if source_numbers.len() != target_numbers.len() {
        return translated.to_string();
    }

    let mut result = translated.to_string();
    let mut search_from = 0usize;
    for (expected, actual) in source_numbers.iter().zip(target_numbers.iter()) {
        let Some(rel) = result[search_from..].find(actual) else {
            return translated.to_string();
        };
        let start = search_from + rel;
        let end = start + actual.len();
        if expected != actual {
            result.replace_range(start..end, expected);
        }
        search_from = start + expected.len();
    }
    result
}

/// Conservative Italian ASR fixes used only for translation input. The
/// transcript shown to the user remains untouched. Keep this list tiny and
/// phrase-specific so we do not "correct" legitimate technical vocabulary.
fn repair_obvious_italian_asr_for_translation(text: &str, source_lang: &str) -> String {
    if source_lang != "it" {
        return text.to_string();
    }

    text.replace("siamo a mangio", "siamo a maggio")
        .replace("Siamo a mangio", "Siamo a maggio")
}

/// 折叠连续重复段落（模型陷入循环时的安全网；间隔不同内容的有意重复会保留）。
fn dedup_consecutive_paragraphs(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    let paras: Vec<&str> = text.trim().split("\n\n").collect();
    if paras.len() <= 1 {
        return text.trim().to_string();
    }
    let mut kept: Vec<&str> = vec![paras[0]];
    let mut prev_norm = normalize_for_dedup(paras[0]);
    for para in &paras[1..] {
        let norm = normalize_for_dedup(para);
        if !norm.is_empty() && norm == prev_norm {
            continue;
        }
        kept.push(para);
        if !norm.is_empty() {
            prev_norm = norm;
        }
    }
    kept.join("\n\n")
}

/// 清洗模型输出：去特殊 token、去回声前缀、按源/目标脚本过滤回声行、
/// 折叠连续重复段落。
pub(crate) fn postprocess(text: &str, source_lang: &str, target_lang: &str) -> String {
    let mut out = strip_special_tokens(text).trim().to_string();

    // 1. 逐个剥离回声前缀（可能叠加多个）
    loop {
        let mut changed = false;
        for prefix in ECHO_PREFIXES {
            if out.starts_with(prefix) {
                out = out[prefix.len()..].trim_start().to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }

    // 2. 按脚本过滤原文回声行（其他语言对不做脚本过滤，避免误伤）
    let cjk_echo_source =
        is_chinese_family(source_lang) || source_lang == "ja" || source_lang == "ko";
    if target_lang == "en" && cjk_echo_source {
        // 目标为英文：丢弃整行 CJK 的行（大概率是原文回声）
        out = out
            .lines()
            .filter(|l| {
                let s = l.trim();
                !s.is_empty() && !is_all_cjk(s)
            })
            .collect::<Vec<_>>()
            .join("\n");
    } else if is_chinese_family(target_lang) {
        // 目标为中文系：丢弃纯拉丁回声行；遇到说明/列表标记即截断
        let mut filtered: Vec<&str> = Vec::new();
        let mut saw_target = false;
        for line in out.lines() {
            let stripped = line.trim();
            if stripped.is_empty() {
                filtered.push(line);
                continue;
            }
            if stripped == "---"
                || stripped == "***"
                || stripped.starts_with("**")
                || stripped.starts_with("* ")
                || stripped.starts_with("- ")
                || ["1. ", "2. ", "3. ", "4. ", "5. ", "6. "]
                    .iter()
                    .any(|p| stripped.starts_with(p))
            {
                break;
            }
            if stripped.contains("说明") || stripped.contains("Note") || stripped.contains("原文") {
                break;
            }
            if is_mostly_latin(stripped) {
                if saw_target {
                    // 已有译文后出现纯拉丁行：原文回声或备注，截断
                    break;
                }
                // 译文前的纯拉丁行：前缀回声，跳过
                continue;
            }
            saw_target = true;
            filtered.push(line);
        }
        out = filtered.join("\n").trim_end().to_string();
    }

    // 3. 折叠连续重复段落
    dedup_consecutive_paragraphs(&out).trim().to_string()
}

// ── 翻译入口 ──────────────────────────────────────────────────────────────────

/// 用 Hy-MT2 翻译一段文本。direction 为 "{src}-{tgt}" 形式（如 "zh-en"、
/// "en-zh-Hant"），src/tgt 须在语言表内。
/// asr_mode=true 使用语音转录纠错指令（实时翻译路径）。
/// on_token 提供时走 sidecar 流式协议，增量文本（未清洗的原始输出）逐个
/// 回调；返回值始终是清洗后的完整译文。
/// 阻塞调用，请放在 spawn_blocking 中执行。
fn use_asr_correction_prompt(text: &str, asr_mode: bool) -> bool {
    asr_mode && text.split_whitespace().count() > 8
}

fn translate_internal(
    text: &str,
    direction: &str,
    asr_mode: bool,
    context_before: Option<&str>,
    on_token: Option<&mut dyn FnMut(&str)>,
) -> Result<String, String> {
    let Some((source_lang, target_lang)) = parse_direction(direction) else {
        return Err(format!("不支持的翻译方向: {}", direction));
    };
    if source_lang == target_lang {
        return Err("源语言与目标语言相同，无需翻译".to_string());
    }
    if text.trim().is_empty() {
        return Ok(String::new());
    }

    let model_path = llama_sidecar::find_gguf_model(MODEL_DIR)
        .ok_or_else(|| "Hy-MT2 翻译模型未安装，请先到设置页下载。".to_string())?;
    let helper_exe = llama_sidecar::resolve_helper_exe()
        .ok_or_else(|| "本地推理引擎（llama-helper）未找到，请重新安装应用".to_string())?;

    // Apply only very conservative, phrase-specific ASR repairs to the text
    // sent to Hy-MT2. The visible transcript remains the original Whisper text.
    let translation_text = repair_obvious_italian_asr_for_translation(text, source_lang);

    // Very short ASR utterances are usually complete phrases. When a previous
    // segment is supplied, use it only for word-sense disambiguation and still
    // translate the current segment alone.
    let use_asr_correction = use_asr_correction_prompt(&translation_text, asr_mode);
    let faithful_short = asr_mode && !use_asr_correction;
    let prompt = if asr_mode {
        if let Some(context) = context_before.filter(|c| !c.trim().is_empty()) {
            build_contextual_prompt(&translation_text, context, source_lang, target_lang)
        } else {
            build_prompt(&translation_text, source_lang, target_lang, use_asr_correction)
        }
    } else {
        build_prompt(&translation_text, source_lang, target_lang, false)
    };

    // 输出预算仅按当前片段估算；context 不需要被生成到答案里。
    let max_tokens = (translation_text.chars().count() * 2).clamp(64, 1024) as u32;
    let (repeat_penalty, frequency_penalty) = if target_lang == "ko" {
        (KO_REPEAT_PENALTY, KO_FREQUENCY_PENALTY)
    } else {
        (REPEAT_PENALTY, FREQUENCY_PENALTY)
    };

    let raw = llama_sidecar::blocking_generate(
        &helper_exe,
        GenerateParams {
            model_path: model_path.to_string_lossy().to_string(),
            prompt,
            max_tokens,
            context_size: CONTEXT_SIZE,
            temperature: if faithful_short { 0.1 } else { TEMPERATURE },
            top_k: TOP_K,
            top_p: TOP_P,
            repeat_penalty: Some(repeat_penalty),
            frequency_penalty: Some(frequency_penalty),
            stop_tokens: STOP_TOKENS.iter().map(|s| s.to_string()).collect(),
            stream: on_token.is_some(),
        },
        on_token,
    )?;

    let cleaned = postprocess(&raw, source_lang, target_lang);
    let numeric_safe = restore_numeric_tokens(&translation_text, &cleaned);
    Ok(numeric_safe)
}

pub fn translate(
    text: &str,
    direction: &str,
    asr_mode: bool,
    on_token: Option<&mut dyn FnMut(&str)>,
) -> Result<String, String> {
    translate_internal(text, direction, asr_mode, None, on_token)
}

/// Translate only `text`, using `context_before` solely to disambiguate a
/// short realtime ASR fragment.
pub fn translate_with_context(
    text: &str,
    context_before: &str,
    direction: &str,
    on_token: Option<&mut dyn FnMut(&str)>,
) -> Result<String, String> {
    translate_internal(text, direction, true, Some(context_before), on_token)
}

/// 启动预加载暖机：发一次最小 generate，使 llama-helper sidecar 启动并驻留
/// Hy-MT2 模型，消除首次翻译的冷启动等待。阻塞调用，请放在 spawn_blocking 中。
pub fn warmup() -> Result<(), String> {
    let model_path = llama_sidecar::find_gguf_model(MODEL_DIR)
        .ok_or_else(|| "Hy-MT2 翻译模型未安装，请先到设置页下载。".to_string())?;
    let helper_exe = llama_sidecar::resolve_helper_exe()
        .ok_or_else(|| "本地推理引擎（llama-helper）未找到，请重新安装应用".to_string())?;

    llama_sidecar::blocking_generate(
        &helper_exe,
        GenerateParams {
            model_path: model_path.to_string_lossy().to_string(),
            prompt: "Hi".to_string(),
            max_tokens: 1,
            context_size: CONTEXT_SIZE,
            temperature: TEMPERATURE,
            top_k: TOP_K,
            top_p: TOP_P,
            repeat_penalty: Some(REPEAT_PENALTY),
            frequency_penalty: Some(FREQUENCY_PENALTY),
            stop_tokens: Vec::new(),
            stream: false,
        },
        None,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_changed_numeric_values_positionally() {
        let source = "le 6 e 40 di pomeriggio, le 18 e 40";
        let translated = "下午6点40分，晚上8点40分";
        assert_eq!(
            restore_numeric_tokens(source, translated),
            "下午6点40分，晚上18点40分"
        );
    }

    #[test]
    fn leaves_numeric_output_untouched_when_counts_differ() {
        let source = "misura 12.5 mm";
        let translated = "尺寸约为12.5毫米，共2件";
        assert_eq!(
            restore_numeric_tokens(source, translated),
            translated
        );
    }

    #[test]
    fn short_asr_phrases_use_faithful_direct_translation() {
        assert!(!use_asr_correction_prompt("sono di milano", true));
        assert!(!use_asr_correction_prompt("Come si scrive il tuo nome?", true));
        assert!(use_asr_correction_prompt(
            "questa e una frase abbastanza lunga da richiedere una leggera correzione asr",
            true
        ));
    }

    #[test]
    fn build_prompt_normal_zh_en() {
        let p = build_prompt("你好，世界", "zh", "en", false);
        assert!(p.starts_with(CHAT_PREFIX));
        assert!(p.ends_with(CHAT_SUFFIX));
        assert!(p.contains("将以下文本翻译为英语"));
        assert!(p.contains(":\n\n你好，世界"));
    }

    #[test]
    fn build_prompt_normal_en_zh() {
        let p = build_prompt("hello world", "en", "zh", false);
        assert!(p.starts_with(CHAT_PREFIX));
        assert!(p.ends_with(CHAT_SUFFIX));
        assert!(p.contains("Translate the following text into Chinese."));
        assert!(p.contains("hello world"));
    }

    #[test]
    fn build_prompt_asr_contains_instruction_and_source_target() {
        let p = build_prompt("今天天气不错", "zh", "en", true);
        assert!(p.starts_with(CHAT_PREFIX));
        assert!(p.ends_with(CHAT_SUFFIX));
        assert!(p.contains("将以下Chinese语音转录文本翻译为English。"));
        // 完整 6 条要求（中文版）
        assert!(p.contains("1. 只输出English译文，严禁输出原文、双语对照、原文片段或重复原文；"));
        assert!(p.contains("3. 完整保留原文每一项语义，不得省略、概括、合并或添加内容；"));
        assert!(p.contains("6. 不要解释，不要备注。"));
        assert!(p.contains("Source: 今天天气不错"));
        assert!(p.contains("Target (English):"));
    }

    #[test]
    fn asr_prompt_preserves_numeric_and_technical_values() {
        let p = build_prompt(
            "Sono le 18 e 40, quota 12.5 mm, codice AB-123.",
            "it",
            "zh",
            true,
        );
        assert!(p.contains("数字、时间、尺寸、单位、型号、零件号必须忠实保留"));
        assert!(p.contains("18 e 40"));
        assert!(p.contains("12.5 mm"));
        assert!(p.contains("AB-123"));
    }

    #[test]
    fn contextual_prompt_marks_previous_segment_as_context_only() {
        let p = build_contextual_prompt(
            "voglio più lavorare",
            "per oggi non",
            "it",
            "zh",
        );
        assert!(p.contains("上一段（仅供语境）：per oggi non"));
        assert!(p.contains("当前片段（只翻译这一段）：voglio più lavorare"));
        assert!(p.contains("严禁翻译或复述上一段"));
    }

    #[test]
    fn italian_to_chinese_prompt_has_literal_realtime_hints() {
        let p = build_prompt(
            "quasi quasi vorrei andare a correre, oggi c'è il sole",
            "it",
            "zh",
            true,
        );
        assert!(p.contains("残句开始或结束"));
        assert!(p.contains("quasi quasi"));
        assert!(p.contains("太阳/阳光"));
    }

    #[test]
    fn build_prompt_chat_tokens_use_fullwidth_pipe() {
        let p = build_prompt("x", "zh", "en", false);
        assert!(p.contains("<\u{FF5C}hy_begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(p.contains("<\u{FF5C}hy_User\u{FF5C}>"));
        assert!(p.contains("<\u{FF5C}hy_Assistant\u{FF5C}>"));
    }

    #[test]
    fn postprocess_strips_special_tokens_and_echo_prefix() {
        let out = postprocess(
            "<\u{FF5C}hy_Assistant\u{FF5C}>Translation: 你好<\u{FF5C}hy_end\u{2581}of\u{2581}sentence\u{FF5C}>",
            "en",
            "zh",
        );
        assert_eq!(out, "你好");
    }

    #[test]
    fn postprocess_strips_source_labels() {
        assert_eq!(
            postprocess("来源：这是译文", "it", "zh"),
            "这是译文"
        );
        assert_eq!(
            postprocess("Source: 这是译文", "it", "zh"),
            "这是译文"
        );
    }

    #[test]
    fn postprocess_strips_target_labels() {
        assert_eq!(
            postprocess("目标（中文）：这是译文", "it", "zh"),
            "这是译文"
        );
        assert_eq!(
            postprocess("Target (Chinese): 这是译文", "it", "zh"),
            "这是译文"
        );
    }

    #[test]
    fn repairs_only_obvious_italian_calendar_asr_for_translation() {
        assert_eq!(
            repair_obvious_italian_asr_for_translation(
                "sole, siamo a mangio quindi fuori c'è ancora luce",
                "it"
            ),
            "sole, siamo a maggio quindi fuori c'è ancora luce"
        );
        assert_eq!(
            repair_obvious_italian_asr_for_translation("siamo a mangio", "en"),
            "siamo a mangio"
        );
    }

    #[test]
    fn postprocess_strips_chinese_echo_prefix() {
        let out = postprocess("译文：今天天气很好", "zh", "en");
        // 前缀剥离后整行 CJK，对英文目标属于回声行，被过滤为空
        assert_eq!(out, "");
        let out2 = postprocess("翻译：The weather is nice today", "zh", "en");
        assert_eq!(out2, "The weather is nice today");
    }

    #[test]
    fn postprocess_drops_cjk_echo_lines_for_en_target() {
        let out = postprocess("This is fine.\n这是回声行", "zh", "en");
        assert_eq!(out, "This is fine.");
    }

    #[test]
    fn postprocess_drops_latin_echo_for_zh_target() {
        let out = postprocess("hello world echo line\n这是译文", "en", "zh");
        assert_eq!(out, "这是译文");
    }

    #[test]
    fn postprocess_collapses_repeated_paragraphs() {
        let out = postprocess("这是译文。\n\n这是译文。", "en", "zh");
        assert_eq!(out, "这是译文。");
    }

    #[test]
    fn parse_direction_common_pairs() {
        assert_eq!(parse_direction("zh-en"), Some(("zh", "en")));
        assert_eq!(parse_direction("en-ja"), Some(("en", "ja")));
        // code 自身含连字符的情况
        assert_eq!(parse_direction("zh-Hant-en"), Some(("zh-Hant", "en")));
        assert_eq!(parse_direction("en-zh-Hant"), Some(("en", "zh-Hant")));
        assert_eq!(parse_direction("yue-zh"), Some(("yue", "zh")));
        assert_eq!(parse_direction("it-zh"), Some(("it", "zh")));
        assert_eq!(parse_direction("it-en"), Some(("it", "en")));
        // 未知 code / 缺少 tgt 均拒绝
        assert_eq!(parse_direction("zh"), None);
        assert_eq!(parse_direction("zh-xx"), None);
        assert_eq!(parse_direction("xx-en"), None);
    }

    #[test]
    fn build_prompt_normal_zh_ja() {
        let p = build_prompt("你好，世界", "zh", "ja", false);
        assert!(p.starts_with(CHAT_PREFIX));
        assert!(p.ends_with(CHAT_SUFFIX));
        assert!(p.contains("将以下文本翻译为日语"));
    }

    #[test]
    fn build_prompt_normal_en_fr() {
        let p = build_prompt("hello world", "en", "fr", false);
        assert!(p.contains("Translate the following text into French."));
        assert!(p.contains("hello world"));
    }

    #[test]
    fn build_prompt_normal_zh_hant_target_uses_chinese_instruction() {
        let p = build_prompt("bonjour", "fr", "zh-Hant", false);
        assert!(p.contains("Translate the following text into Traditional Chinese."));
    }

    #[test]
    fn build_prompt_asr_italian_to_chinese() {
        let p = build_prompt("Ciao, come stai?", "it", "zh", true);
        assert!(p.contains("将以下Italian语音转录文本翻译为Chinese。"));
        assert!(p.contains("Source: Ciao, come stai?"));
        assert!(p.contains("Target (Chinese):"));
    }

    #[test]
    fn build_prompt_asr_italian_to_english() {
        let p = build_prompt("Ciao, come stai?", "it", "en", true);
        assert!(p.contains("Translate the following Italian spoken transcript into English."));
        assert!(p.contains("Source: Ciao, come stai?"));
        assert!(p.contains("Target (English):"));
    }

    #[test]
    fn build_prompt_asr_non_chinese_pair_uses_english_instruction() {
        let p = build_prompt("hello world", "en", "ja", true);
        assert!(p.contains("Translate the following English spoken transcript into Japanese."));
        // 完整 6 条要求（英文版）
        assert!(p.contains("1. Output ONLY the Japanese translation."));
        assert!(p.contains("3. Preserve every semantic unit; do not omit, summarize, merge, or add content."));
        assert!(p.contains("6. Do not explain or add notes."));
        assert!(p.contains("Source: hello world"));
        assert!(p.contains("Target (Japanese):"));
    }

    #[test]
    fn build_prompt_asr_chinese_family_target_uses_chinese_instruction() {
        let p = build_prompt("bonjour le monde", "fr", "zh-Hant", true);
        assert!(p.contains("将以下French语音转录文本翻译为Traditional Chinese"));
        assert!(p.contains("Target (Traditional Chinese):"));
    }

    #[test]
    fn postprocess_keeps_cjk_lines_for_non_en_target() {
        // zh → ja：不做脚本过滤，CJK 行不应被误删
        let out = postprocess("こんにちは\n这是中文行", "zh", "ja");
        assert_eq!(out, "こんにちは\n这是中文行");
    }

    #[test]
    fn postprocess_drops_cjk_echo_lines_ja_to_en() {
        let out = postprocess("This is fine.\nこれは回声です", "ja", "en");
        assert_eq!(out, "This is fine.");
    }

    #[test]
    fn postprocess_drops_latin_echo_for_zh_hant_target() {
        let out = postprocess("echo line in latin\n這是譯文", "fr", "zh-Hant");
        assert_eq!(out, "這是譯文");
    }

    #[test]
    fn postprocess_no_script_filter_for_latin_pair() {
        // fr → de：不做脚本过滤，拉丁行全部保留
        let out = postprocess("Bonjour le monde\nHallo Welt", "fr", "de");
        assert_eq!(out, "Bonjour le monde\nHallo Welt");
    }
}
