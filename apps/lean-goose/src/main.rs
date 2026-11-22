use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure, Context, Result};
use async_trait::async_trait;
use goose_agent::inference::{InferenceEffect, InferenceRunner};
use goose_agent::machine::{EffectHandler, MachineSession, SessionLoader, StateMachine, Step};
use goose_agent::operation::{Emitter, MachineEffect};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::ProviderUsage;
use goose_provider_types::conversation::Conversation;
use goose_provider_types::formats::openai::create_request;
use goose_provider_types::images::ImageFormat;
use goose_provider_types::model::ModelConfig;
use goose_providers::api_client::{ApiClient, ApiResponse, AuthMethod};
use goose_providers::openai_compatible::OpenAiCompatibleProvider;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const APP: &str = "lean-goose/goose-agent-0.1.0-alpha.14";
const TOTAL_TOKEN_LIMIT: usize = 15_999;

#[derive(Clone)]
struct Session(Conversation);

impl MachineSession for Session {
    fn id(&self) -> &str {
        "lean-rollout"
    }

    fn conversation(&self) -> Option<&Conversation> {
        Some(&self.0)
    }
}

enum Effect {
    Append(Message),
    Usage(ProviderUsage),
}

impl From<Message> for Effect {
    fn from(message: Message) -> Self {
        Self::Append(message)
    }
}

impl MachineEffect for Effect {
    fn ensure_message_ids(&mut self) {
        if let Self::Append(message) = self {
            message.id.get_or_insert_with(|| "lean-assistant".into());
        }
    }
}

impl InferenceEffect for Effect {
    fn record_usage(usage: ProviderUsage) -> Self {
        Self::Usage(usage)
    }
}

struct Runtime {
    session: Mutex<Session>,
    usage: Mutex<Vec<ProviderUsage>>,
}

#[async_trait]
impl SessionLoader<Session> for Runtime {
    async fn load(&self, _: &str) -> Result<Session> {
        Ok(self.session.lock().unwrap().clone())
    }
}

#[async_trait]
impl EffectHandler<Session, Effect> for Runtime {
    async fn apply_effects(&self, _: &Session, effects: &mut [Effect], _: &Emitter) -> Result<()> {
        for effect in effects {
            match effect {
                Effect::Append(message) => self.session.lock().unwrap().0.push(message.clone()),
                Effect::Usage(usage) => self.usage.lock().unwrap().push(usage.clone()),
            }
        }
        Ok(())
    }
}

fn payload(response: ApiResponse, endpoint: &str) -> Result<Value> {
    ensure!(
        response.status.is_success(),
        "{endpoint}: HTTP {}",
        response.status
    );
    response
        .payload
        .with_context(|| format!("{endpoint}: missing JSON"))
}

async fn preflight(api: &ApiClient, config: &ModelConfig, message: &Message) -> Result<usize> {
    let mut request = create_request(
        config,
        "",
        std::slice::from_ref(message),
        &[],
        &ImageFormat::OpenAi,
        false,
    )?;
    request["add_generation_prompt"] = json!(true);
    let template = payload(
        api.api_post("apply-template", &request).await?,
        "apply-template",
    )?;
    let framed = template["prompt"]
        .as_str()
        .context("template omitted prompt")?;
    let tokenized = payload(
        api.api_post(
            "tokenize",
            &json!({
                "content": framed, "add_special": false, "parse_special": true
            }),
        )
        .await?,
        "tokenize",
    )?;
    let tokens = tokenized["tokens"]
        .as_array()
        .context("tokenizer omitted tokens")?
        .len();
    ensure!(
        tokens + config.max_tokens.context("missing generation cap")? as usize <= TOTAL_TOKEN_LIMIT,
        "prompt exceeds token budget"
    );
    Ok(tokens)
}

async fn infer(
    api: &ApiClient,
    provider: Arc<OpenAiCompatibleProvider>,
    model: &str,
    request: &Value,
) -> Result<Value> {
    let prompt = request["prompt"]
        .as_str()
        .context("prompt must be a string")?;
    ensure!(
        !prompt.is_empty() && prompt.len() <= 64_000 && !prompt.contains("<|"),
        "invalid prompt"
    );
    let max_tokens = request["max_tokens"]
        .as_i64()
        .filter(|n| (1..=192).contains(n))
        .context("max_tokens must be between 1 and 192")? as i32;
    let model_config = ModelConfig::new(model)
        .with_context_limit(Some(TOTAL_TOKEN_LIMIT))
        .with_temperature(Some(0.0))
        .with_max_tokens(Some(max_tokens));
    let message = Message::user().with_text(prompt);
    let preflight_started = Instant::now();
    let prompt_tokens = preflight(api, &model_config, &message).await?;
    let preflight_seconds = preflight_started.elapsed().as_secs_f64();
    let cancel = CancellationToken::new();
    let machine = StateMachine::new(
        vec![Step::Inference(Arc::new(InferenceRunner::new(
            provider,
            model_config,
        )))],
        cancel.clone(),
    );
    let runtime = Runtime {
        session: Mutex::new(Session(Conversation::new_unvalidated([message]))),
        usage: Mutex::new(Vec::new()),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let emitter = Emitter::new(tx, cancel);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let started = Instant::now();
    let result = machine.run(&runtime, "lean-rollout", &emitter).await;
    let request_seconds = started.elapsed().as_secs_f64();
    drop(emitter);
    drain.abort();
    let session = result?;
    let answer = session
        .0
        .last()
        .filter(|_| session.0.len() > 1)
        .context("GDK returned no assistant message")?
        .as_concat_text();
    let usage = runtime.usage.lock().unwrap();
    let [reported] = usage.as_slice() else {
        return Err(anyhow!("expected one GDK usage record"));
    };
    let parity = reported.usage.input_tokens == Some(prompt_tokens as i64);
    Ok(json!({
        "app": APP, "model": model, "answer": answer, "provider": reported,
        "preflight_prompt_tokens": prompt_tokens, "prompt_token_parity": parity,
        "preflight_seconds": preflight_seconds, "request_seconds": request_seconds,
        "max_tokens": max_tokens, "context_limit": TOTAL_TOKEN_LIMIT,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [port, model] = args.as_slice() else {
        return Err(anyhow!("usage: lean-goose PORT MODEL"));
    };
    let port: u16 = port.parse().context("invalid llama.cpp port")?;
    ensure!(port > 0 && !model.is_empty(), "port and model are required");
    let base = format!("http://127.0.0.1:{port}/");
    let client = |url| -> Result<ApiClient> {
        ApiClient::with_timeout_and_tls(url, AuthMethod::NoAuth, Duration::from_secs(90), None)?
            .with_loopback_http_only()
    };
    let api = client(base.clone())?;
    let provider = Arc::new(
        OpenAiCompatibleProvider::new(
            "lean-llamacpp".into(),
            client(format!("{base}v1/"))?,
            String::new(),
        )
        .with_supports_streaming(false),
    );
    let advertised = payload(api.api_get("v1/models").await?, "v1/models")?;
    let listed = advertised["data"]
        .as_array()
        .context("model list omitted data")?;
    let identity = listed
        .iter()
        .find(|entry| entry["id"].as_str() == Some(model))
        .context("llama.cpp does not advertise the requested model")?;
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "{}",
        json!({"app": APP, "model": identity, "resident": true})
    )?;
    stdout.flush()?;
    for line in io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line?).context("invalid JSONL request")?;
        let result = infer(&api, provider.clone(), model, &request).await;
        let receipt =
            result.unwrap_or_else(|error| json!({"app": APP, "error": format!("{error:#}")}));
        writeln!(stdout, "{receipt}")?;
        stdout.flush()?;
    }
    Ok(())
}
