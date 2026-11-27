//! data-goblin: the Data Goblin GDK app.
//!
//!   data-goblin run (--prompt TEXT | --skill NAME [--args TEXT]) [options]
//!   data-goblin climb [--lane LANE] [--moves N] [options]
//!   data-goblin skills [--root DIR] [--skills-dir DIR]...
//!   data-goblin tools [options]
//!
//! `run` and every `climb` move print one JSON receipt and append it to `--receipts`
//! (default `<state dir>/receipts.jsonl`).

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use datagoblin_gdk::agent::{Agent, AgentSetup, ApprovePolicy, ToolMode};
use datagoblin_gdk::mcp::{HillclimbTools, ServerSpec};
use datagoblin_gdk::skills::SkillSet;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const HILLCLIMB_SKILL: &str = "datagoblin-hillclimb";

const USAGE: &str = "usage: data-goblin (run (--prompt TEXT | --skill NAME [--args TEXT]) | climb [--lane LANE] [--moves N] \
| skills | tools) [--root DIR] [--python PY] [--base-url URL] [--model ID] [--teacher URL]... [--tool-mode native|text] \
[--approve read-only|ledger|local|all] [--max-turns N] [--max-tokens N] [--context N] [--temperature F] \
[--timeout-secs N] [--state-dir DIR] [--receipts FILE] [--skills-dir DIR]...";

struct Args {
    command: String,
    root: PathBuf,
    python: String,
    base_url: String,
    model: String,
    teachers: Vec<String>,
    tool_mode: ToolMode,
    approve: ApprovePolicy,
    max_turns: u32,
    max_tokens: i32,
    context: usize,
    temperature: f32,
    timeout_secs: u64,
    state_dir: Option<PathBuf>,
    receipts: Option<PathBuf>,
    skills_dirs: Vec<PathBuf>,
    prompt: Option<String>,
    skill: Option<String>,
    skill_args: String,
    lane: Option<String>,
    moves: u32,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn loopback(url: &str) -> bool {
    ["http://127.0.0.1:", "http://localhost:", "http://[::1]:"].iter().any(|prefix| url.starts_with(prefix))
}

fn parse_args() -> Result<Args> {
    let mut raw = std::env::args().skip(1).peekable();
    let command = match raw.peek().map(String::as_str) {
        Some("run" | "climb" | "skills" | "tools") => raw.next().unwrap(),
        Some("--help" | "-h") | None => bail!("{USAGE}"),
        Some(_) => "run".to_string(),
    };
    let mut args = Args {
        command,
        root: PathBuf::from(env_or("DATAGOBLIN_ROOT", ".")),
        python: env_or("DATAGOBLIN_PYTHON", "python"),
        base_url: env_or("DATAGOBLIN_BASE_URL", "http://127.0.0.1:18090/v1"),
        model: env_or("DATAGOBLIN_MODEL", "inkling-2-mini"),
        teachers: Vec::new(),
        tool_mode: ToolMode::Native,
        approve: ApprovePolicy::Ledger,
        max_turns: 40,
        max_tokens: 2048,
        context: 32768,
        temperature: 0.2,
        timeout_secs: 900,
        state_dir: None,
        receipts: None,
        skills_dirs: Vec::new(),
        prompt: None,
        skill: None,
        skill_args: String::new(),
        lane: None,
        moves: 1,
    };
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| anyhow!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => args.root = PathBuf::from(value()?),
            "--python" => args.python = value()?,
            "--base-url" => args.base_url = value()?,
            "--model" => args.model = value()?,
            "--teacher" => args.teachers.push(value()?),
            "--tool-mode" => {
                args.tool_mode = match value()?.as_str() {
                    "native" => ToolMode::Native,
                    "text" => ToolMode::Text,
                    other => bail!("--tool-mode {other}: native or text"),
                }
            }
            "--approve" => {
                let raw = value()?;
                args.approve =
                    ApprovePolicy::parse(&raw).ok_or_else(|| anyhow!("--approve {raw}: read-only, ledger, local or all"))?;
            }
            "--max-turns" => args.max_turns = value()?.parse()?,
            "--max-tokens" => args.max_tokens = value()?.parse()?,
            "--context" => args.context = value()?.parse()?,
            "--temperature" => args.temperature = value()?.parse()?,
            "--timeout-secs" => args.timeout_secs = value()?.parse()?,
            "--state-dir" => args.state_dir = Some(PathBuf::from(value()?)),
            "--receipts" => args.receipts = Some(PathBuf::from(value()?)),
            "--skills-dir" => args.skills_dirs.push(PathBuf::from(value()?)),
            "--prompt" => args.prompt = Some(value()?),
            "--skill" => args.skill = Some(value()?),
            "--args" => args.skill_args = value()?,
            "--lane" => args.lane = Some(value()?),
            "--moves" => args.moves = value()?.parse()?,
            "--help" | "-h" => bail!("{USAGE}"),
            other => bail!("unknown flag {other}\n{USAGE}"),
        }
    }
    args.root = std::fs::canonicalize(&args.root).map_err(|e| anyhow!("--root {}: {e}", args.root.display()))?;
    if args.skills_dirs.is_empty() {
        args.skills_dirs.push(args.root.join("gdk").join("skills"));
    }
    for url in std::iter::once(&args.base_url).chain(args.teachers.iter()) {
        if !loopback(url) {
            bail!("{url}: Data Goblin talks to local model servers only (http://127.0.0.1:PORT/v1)");
        }
    }
    if !args.base_url.ends_with('/') {
        args.base_url.push('/');
    }
    if args.command == "run" && args.prompt.is_none() && args.skill.is_none() {
        bail!("run needs --prompt or --skill\n{USAGE}");
    }
    Ok(args)
}

fn state_dir(args: &Args) -> PathBuf {
    args.state_dir.clone().unwrap_or_else(|| args.root.join("runs").join("hillclimb"))
}

async fn connect(args: &Args) -> Result<Arc<HillclimbTools>> {
    let mut endpoints = vec![args.base_url.trim_end_matches('/').to_string()];
    endpoints.extend(args.teachers.iter().map(|u| u.trim_end_matches('/').to_string()));
    HillclimbTools::connect(&ServerSpec {
        python: args.python.clone(),
        root: args.root.clone(),
        state_dir: args.state_dir.clone(),
        max_tier: args.approve.max_tier().to_string(),
        read_only: args.approve == ApprovePolicy::ReadOnly,
        endpoints,
    })
    .await
}

fn setup(args: &Args) -> AgentSetup {
    AgentSetup {
        base_url: args.base_url.clone(),
        served_model: args.model.clone(),
        tool_mode: args.tool_mode,
        context_tokens: args.context,
        max_tokens: args.max_tokens,
        max_turns: args.max_turns,
        temperature: args.temperature,
        policy: args.approve,
        timeout: Duration::from_secs(args.timeout_secs),
    }
}

fn append_receipt(args: &Args, receipt: &Value) -> Result<()> {
    let path = args.receipts.clone().unwrap_or_else(|| state_dir(args).join("receipts.jsonl"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(file, "{receipt}")?;
    Ok(())
}

async fn run_once(args: &Args, tools: Arc<HillclimbTools>, skills: Arc<SkillSet>, prompt: &str) -> Result<Value> {
    let started = Instant::now();
    let agent = Agent::new(setup(args), tools, skills)?;
    agent.push_user(prompt);
    let error = agent.run(CancellationToken::new()).await.err().map(|e| format!("agent_error: {e:#}"));
    let receipt = agent.receipt(prompt, started, error);
    append_receipt(args, &receipt)?;
    Ok(receipt)
}

fn summary(receipt: &Value) -> Value {
    json!({
        "moves": receipt["moves"], "decisions": receipt["decisions"], "toolCalls": receipt["toolCalls"].as_array().map(Vec::len),
        "failedCalls": receipt["toolCalls"].as_array().map(|c| c.iter().filter(|x| x["ok"] != json!(true)).count()),
        "inferenceCalls": receipt["inferenceCalls"], "seconds": receipt["seconds"], "error": receipt["error"],
        "answer": receipt["answer"],
    })
}

async fn climb(args: &Args, tools: Arc<HillclimbTools>, skills: Arc<SkillSet>) -> Result<i32> {
    if !skills.names().iter().any(|name| name == HILLCLIMB_SKILL) {
        bail!("skill {HILLCLIMB_SKILL} not found under {:?}", args.skills_dirs);
    }
    let prompt = match &args.lane {
        Some(lane) => format!("/{HILLCLIMB_SKILL} {lane}"),
        None => format!("/{HILLCLIMB_SKILL}"),
    };
    let mut idle = 0;
    for index in 1..=args.moves {
        let receipt = run_once(args, tools.clone(), skills.clone(), &prompt).await?;
        let mut line = summary(&receipt);
        line["iteration"] = json!(index);
        println!("{line}");
        if receipt["error"].is_string() {
            return Ok(1);
        }
        idle = if receipt["decisions"].as_array().is_some_and(|d| !d.is_empty()) { 0 } else { idle + 1 };
        if idle >= 2 {
            println!("{}", json!({"stopped": "no_decision_in_two_runs"}));
            return Ok(1);
        }
        if let Some(lane) = &args.lane {
            let status = tools.call_json("ledger_status", json!({"lane": lane, "last": 1})).await?;
            let lane_status = status["lanes"][0]["status"].as_str().unwrap_or("unknown").to_string();
            if lane_status != "open" {
                println!("{}", json!({"stopped": format!("lane_{lane_status}"), "lane": status["lanes"][0]}));
                return Ok(0);
            }
        }
    }
    Ok(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let skills = Arc::new(SkillSet::discover(&args.skills_dirs));
    if args.command == "skills" {
        for skill in &skills.entries {
            println!(
                "{}",
                json!({"name": skill.name, "description": skill.description, "path": skill.path,
                       "supportingFiles": skill.supporting_files.len()})
            );
        }
        return Ok(());
    }
    let tools = connect(&args).await?;
    let code = match args.command.as_str() {
        "tools" => {
            let mut names: Vec<String> = tools.names().into_iter().collect();
            names.sort();
            let mut tiers: Vec<(&String, &String)> = tools.recipe_tiers.iter().collect();
            tiers.sort();
            println!(
                "{}",
                json!({"tools": names, "readOnly": tools.read_only, "recipes": tiers, "skills": skills.names(),
                       "approve": args.approve.as_str(), "maxTier": args.approve.max_tier(),
                       "instructions": tools.instructions})
            );
            0
        }
        "climb" => climb(&args, tools, skills).await?,
        _ => {
            let prompt = match (&args.prompt, &args.skill) {
                (Some(prompt), _) => prompt.clone(),
                (None, Some(skill)) => format!("/{skill} {}", args.skill_args).trim_end().to_string(),
                (None, None) => unreachable!(),
            };
            let receipt = run_once(&args, tools, skills, &prompt).await?;
            println!("{receipt}");
            if receipt["error"].is_string() {
                1
            } else {
                0
            }
        }
    };
    std::process::exit(code);
}
