// translation/m2m100.rs
//
// Fast multilingual machine translation for realtime subtitles.
// Model: Meta M2M100 418M, INT8 ONNX (Xenova export).
//
// M2M100 source format: [src_lang_token] + text + [eos].
// Generation starts from eos (decoder_start_token_id=2) and forces the
// target language token as the first generated token.

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub const ENCODER_FILE: &str = "encoder_model_quantized.onnx";
pub const DECODER_FILE: &str = "decoder_model_merged_quantized.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const REQUIRED_FILES: &[&str] = &[ENCODER_FILE, DECODER_FILE, TOKENIZER_FILE];

pub const SUPPORTED_TARGET_LANGS: &[&str] = &[
    "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt", "it", "th", "vi",
];

const EOS_TOKEN_ID: i64 = 2;
const DECODER_LAYERS: usize = 12;
const ATTENTION_HEADS: i64 = 16;
const HEAD_DIM: i64 = 64;
const MAX_NEW_TOKENS: usize = 192;

fn sanitize_bpe_merges(value: &mut serde_json::Value) -> usize {
    let vocab: HashSet<String> = value
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .map(|v| v.keys().cloned().collect())
        .unwrap_or_default();

    if vocab.is_empty() {
        return 0;
    }

    let Some(merges) = value
        .get_mut("model")
        .and_then(|m| m.get_mut("merges"))
        .and_then(|m| m.as_array_mut())
    else {
        return 0;
    };

    let before = merges.len();
    merges.retain(|entry| {
        let pair: Option<(&str, &str)> = match entry {
            serde_json::Value::Array(parts) if parts.len() == 2 => {
                match (parts[0].as_str(), parts[1].as_str()) {
                    (Some(left), Some(right)) => Some((left, right)),
                    _ => None,
                }
            }
            serde_json::Value::String(rule) => rule
                .split_once(' ')
                .map(|(left, right)| (left, right)),
            _ => None,
        };

        let Some((left, right)) = pair else {
            return false;
        };

        // Hugging Face tokenizers requires both sides of every BPE merge,
        // and the token produced by that merge, to exist in model.vocab.
        // Some Transformers.js fast-tokenizer exports contain stale merge
        // rules (e.g. a literal "8") that violate this invariant.
        let merged = format!("{}{}", left, right);
        vocab.contains(left) && vocab.contains(right) && vocab.contains(&merged)
    });
    before - merges.len()
}

fn load_tokenizer_compat(path: &Path) -> Result<Tokenizer> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("读取 M2M100 tokenizer.json 失败: {}", e))?;

    // Prefer the file unchanged. This keeps us compatible if the upstream
    // tokenizer is fixed in a future revision.
    match Tokenizer::from_bytes(raw.as_bytes()) {
        Ok(tokenizer) => return Ok(tokenizer),
        Err(original_error) => {
            log::warn!(
                "M2M100 tokenizer direct load failed: {}. Trying BPE merge compatibility cleanup.",
                original_error
            );
        }
    }

    let mut value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow!("解析 M2M100 tokenizer.json 失败: {}", e))?;
    let removed = sanitize_bpe_merges(&mut value);
    if removed == 0 {
        return Err(anyhow!(
            "加载 M2M100 tokenizer 失败，且未发现可修复的 BPE merge 规则"
        ));
    }

    log::warn!(
        "M2M100 tokenizer compatibility cleanup removed {} invalid BPE merge rule(s)",
        removed
    );
    let patched = serde_json::to_vec(&value)
        .map_err(|e| anyhow!("重写 M2M100 tokenizer.json 失败: {}", e))?;

    Tokenizer::from_bytes(&patched)
        .map_err(|e| anyhow!("加载兼容处理后的 M2M100 tokenizer 失败: {}", e))
}

pub struct M2M100Engine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl M2M100Engine {
    pub fn load(dir: &Path) -> Result<Self> {
        let encoder_path = dir.join(ENCODER_FILE);
        let decoder_path = dir.join(DECODER_FILE);
        let tokenizer_path = dir.join(TOKENIZER_FILE);
        for p in [&encoder_path, &decoder_path, &tokenizer_path] {
            if !p.exists() {
                return Err(anyhow!("M2M100 模型文件缺失: {}", p.display()));
            }
        }

        let encoder = Session::builder()
            .map_err(|e| anyhow!("初始化 M2M100 encoder 失败: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("配置 M2M100 encoder 线程失败: {}", e))?
            .commit_from_file(&encoder_path)
            .map_err(|e| anyhow!("加载 M2M100 encoder 失败: {}", e))?;
        let decoder = Session::builder()
            .map_err(|e| anyhow!("初始化 M2M100 decoder 失败: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("配置 M2M100 decoder 线程失败: {}", e))?
            .commit_from_file(&decoder_path)
            .map_err(|e| anyhow!("加载 M2M100 decoder 失败: {}", e))?;
        let tokenizer = load_tokenizer_compat(&tokenizer_path)?;

        let enc_inputs: Vec<String> = encoder.inputs.iter().map(|i| i.name.clone()).collect();
        let dec_inputs: Vec<String> = decoder.inputs.iter().map(|i| i.name.clone()).collect();
        let dec_outputs: Vec<String> = decoder.outputs.iter().map(|o| o.name.clone()).collect();
        log::info!(
            "M2M100 ready from {} (encoder inputs: {:?}, decoder inputs: {:?}, decoder outputs: {:?})",
            dir.display(),
            enc_inputs,
            dec_inputs,
            dec_outputs
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
        })
    }

    fn lang_token_id(&self, lang: &str) -> Result<i64> {
        if !SUPPORTED_TARGET_LANGS.contains(&lang) {
            return Err(anyhow!("M2M100 不支持语言: {}", lang));
        }
        self.tokenizer
            .token_to_id(&format!("__{}__", lang))
            .map(|id| id as i64)
            .ok_or_else(|| anyhow!("M2M100 tokenizer 缺少语言 token: __{}__", lang))
    }

    pub fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        if source_lang == target_lang {
            return Ok(text.to_string());
        }

        let src_lang_id = self.lang_token_id(source_lang)?;
        let tgt_lang_id = self.lang_token_id(target_lang)?;
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("M2M100 分词失败: {}", e))?;

        let mut ids: Vec<i64> = Vec::with_capacity(encoding.len() + 2);
        ids.push(src_lang_id);
        ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
        ids.push(EOS_TOKEN_ID);
        let mask = vec![1i64; ids.len()];
        let seq_len = ids.len() as i64;

        let input_ids = Tensor::from_array((vec![1i64, seq_len], ids))
            .map_err(|e| anyhow!("创建 M2M100 input_ids 失败: {}", e))?;
        let attention = Tensor::from_array((vec![1i64, seq_len], mask.clone()))
            .map_err(|e| anyhow!("创建 M2M100 attention_mask 失败: {}", e))?;

        let mut encoder = self.encoder.lock().unwrap();
        let enc_outputs = encoder
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention
            ])
            .map_err(|e| anyhow!("M2M100 encoder 推理失败: {}", e))?;
        let (hs_shape, hs_data) = enc_outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("读取 M2M100 encoder 输出失败: {}", e))?;
        let hidden_shape: Vec<i64> = hs_shape.iter().copied().collect();
        let hidden_vec: Vec<f32> = hs_data.to_vec();
        drop(enc_outputs);
        drop(encoder);

        let mut decoder = self.decoder.lock().unwrap();
        let ctx = DecodeCtx {
            hidden_shape,
            hidden_vec,
            mask,
            seq_len,
        };
        let generated = self.greedy_decode(&mut decoder, &ctx, tgt_lang_id)?;
        drop(decoder);

        let token_ids: Vec<u32> = generated.into_iter().map(|x| x as u32).collect();
        let out = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow!("M2M100 解码失败: {}", e))?;
        Ok(out.trim().to_string())
    }

    fn greedy_decode(
        &self,
        decoder: &mut Session,
        ctx: &DecodeCtx,
        target_lang_id: i64,
    ) -> Result<Vec<i64>> {
        let empty: Vec<(Vec<i64>, Vec<f32>)> = (0..DECODER_LAYERS * 2)
            .map(|_| (vec![1, ATTENTION_HEADS, 0, HEAD_DIM], Vec::new()))
            .collect();
        let allocator = ort::memory::Allocator::default();
        let has_enc_mask = decoder
            .inputs
            .iter()
            .any(|i| i.name == "encoder_attention_mask");

        let mut dec_past = empty.clone();
        let mut enc_past = empty;

        let start = self.decoder_step(
            decoder,
            ctx,
            EOS_TOKEN_ID,
            &dec_past,
            &enc_past,
            true,
            has_enc_mask,
            &allocator,
        )?;
        dec_past = start.decoder_past;
        if let Some(ep) = start.encoder_past {
            enc_past = ep;
        }

        let mut prev = target_lang_id;
        let mut out = Vec::new();

        for _ in 0..MAX_NEW_TOKENS {
            let so = self.decoder_step(
                decoder,
                ctx,
                prev,
                &dec_past,
                &enc_past,
                false,
                has_enc_mask,
                &allocator,
            )?;
            dec_past = so.decoder_past;
            if let Some(ep) = so.encoder_past {
                enc_past = ep;
            }

            let next = so
                .last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(EOS_TOKEN_ID);
            if next == EOS_TOKEN_ID {
                break;
            }
            out.push(next);
            prev = next;
        }

        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn decoder_step(
        &self,
        decoder: &mut Session,
        ctx: &DecodeCtx,
        prev_token: i64,
        decoder_past: &[(Vec<i64>, Vec<f32>)],
        encoder_past: &[(Vec<i64>, Vec<f32>)],
        first_step: bool,
        has_enc_mask: bool,
        allocator: &ort::memory::Allocator,
    ) -> Result<StepOutput> {
        let dec_ids = Tensor::from_array((vec![1i64, 1i64], vec![prev_token]))
            .map_err(|e| anyhow!("创建 M2M100 decoder input 失败: {}", e))?;
        let enc_hs = Tensor::from_array((ctx.hidden_shape.clone(), ctx.hidden_vec.clone()))
            .map_err(|e| anyhow!("创建 M2M100 encoder_hidden_states 失败: {}", e))?;
        let use_cache = Tensor::from_array((vec![1i64], vec![!first_step]))
            .map_err(|e| anyhow!("创建 M2M100 use_cache_branch 失败: {}", e))?;

        let mut inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = vec![
            ("input_ids".into(), dec_ids.into()),
            ("encoder_hidden_states".into(), enc_hs.into()),
            ("use_cache_branch".into(), use_cache.into()),
        ];

        let enc_mask_tensor;
        if has_enc_mask {
            enc_mask_tensor =
                Tensor::from_array((vec![1i64, ctx.seq_len], ctx.mask.clone()))
                    .map_err(|e| anyhow!("创建 M2M100 encoder_attention_mask 失败: {}", e))?;
            inputs.push(("encoder_attention_mask".into(), enc_mask_tensor.into()));
        }

        for i in 0..DECODER_LAYERS {
            for (side_past, side_name) in [
                (decoder_past, "decoder"),
                (encoder_past, "encoder"),
            ] {
                for (idx, kv) in ["key", "value"].iter().enumerate() {
                    let (shape, data) = &side_past[i * 2 + idx];
                    let name = format!("past_key_values.{}.{}.{}", i, side_name, kv);
                    let t = if data.is_empty() {
                        Tensor::<f32>::new(allocator, shape.clone())
                            .map_err(|e| anyhow!("创建空 {} 失败: {}", name, e))?
                    } else {
                        Tensor::from_array((shape.clone(), data.clone()))
                            .map_err(|e| anyhow!("创建 {} 失败: {}", name, e))?
                    };
                    inputs.push((name.into(), t.into()));
                }
            }
        }

        let outs = decoder
            .run(inputs)
            .map_err(|e| anyhow!("M2M100 decoder 推理失败: {}", e))?;

        let (logits_shape, logits) = outs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("读取 M2M100 logits 失败: {}", e))?;
        let dims: Vec<i64> = logits_shape.iter().copied().collect();
        if dims.len() != 3 || dims[1] == 0 {
            return Err(anyhow!("M2M100 decoder logits 形状异常: {:?}", dims));
        }
        let vocab = dims[2] as usize;
        let offset = (dims[1] as usize - 1) * vocab;
        let last_logits = logits[offset..offset + vocab].to_vec();

        let mut new_decoder_past = Vec::with_capacity(DECODER_LAYERS * 2);
        let mut new_encoder_past: Option<Vec<(Vec<i64>, Vec<f32>)>> = None;
        for i in 0..DECODER_LAYERS {
            for kv in ["key", "value"] {
                for (side, is_dec) in [("decoder", true), ("encoder", false)] {
                    let name = format!("present.{}.{}.{}", i, side, kv);
                    if !is_dec && !first_step {
                        continue;
                    }
                    let (shape, data) = outs[name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| anyhow!("读取 {} 失败: {}", name, e))?;
                    let entry = (shape.iter().copied().collect(), data.to_vec());
                    if is_dec {
                        new_decoder_past.push(entry);
                    } else {
                        new_encoder_past
                            .get_or_insert_with(Vec::new)
                            .push(entry);
                    }
                }
            }
        }

        Ok(StepOutput {
            last_logits,
            decoder_past: new_decoder_past,
            encoder_past: new_encoder_past,
        })
    }
}

struct DecodeCtx {
    hidden_shape: Vec<i64>,
    hidden_vec: Vec<f32>,
    mask: Vec<i64>,
    seq_len: i64,
}

struct StepOutput {
    last_logits: Vec<f32>,
    decoder_past: Vec<(Vec<i64>, Vec<f32>)>,
    encoder_past: Option<Vec<(Vec<i64>, Vec<f32>)>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_bpe_merges_are_filtered_for_transformers_js_compatibility() {
        let mut value = serde_json::json!({
            "model": {
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "ab": 2
                },
                "merges": [
                    ["a", "b"],
                    ["8", "a"],
                    ["a", "missing"]
                ]
            }
        });
        let removed = sanitize_bpe_merges(&mut value);
        assert_eq!(removed, 2);
        assert_eq!(value["model"]["merges"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn realtime_languages_cover_italian_chinese() {
        assert!(SUPPORTED_TARGET_LANGS.contains(&"it"));
        assert!(SUPPORTED_TARGET_LANGS.contains(&"zh"));
    }
}
