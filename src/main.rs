use clap::Parser;
use colored::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
#[derive(Parser, Debug)]
#[command(
    name = "orca",
    about = "O.R.C.A. - Four-Model Collaborative Task Processor",
    long_about = None,
    after_help = "Configuration:\n  \
                  A config.json file in the current directory is loaded automatically.\n  \
                  Priority: command-line arguments > environment variables > config.json > defaults.\n\n\
                  Environment variables:\n  \
                  OPENAI_API_KEY         API key for OpenAI-compatible service.\n  \
                  OPENAI_BASE_URL        Base URL for the API (default: https:
                  ORCA_OPTIMIZE_MODEL    Model for Optimize stage.\n  \
                  ORCA_REFINE_MODEL      Model for Refine stage.\n  \
                  ORCA_ASSESS_MODEL      Model for Assess stage.\n\n\
                  Example config.json:\n  \
                  {\n    \"api_key\": \"your-key-here\",\n    \
                  \"base_url\": \"https:
                  \"models\": {\n      \"optimize\": \"deepseek-chat\",\n      \
                  \"refine\": \"deepseek-chat\",\n      \
                  \"assess\": \"deepseek-chat\"\n    }\n  }"
)]
struct Args {
    #[arg(required_unless_present = "create_config")]
    input: Vec<String>,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    optimize_model: Option<String>,
    #[arg(long)]
    refine_model: Option<String>,
    #[arg(long)]
    assess_model: Option<String>,
    #[arg(long)]
    create_config: bool,
}
const CONFIG_FILE: &str = "config.json";
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelsConfig {
    optimize: String,
    refine: String,
    assess: String,
}
impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            optimize: "gpt-3.5-turbo".to_string(),
            refine: "gpt-3.5-turbo".to_string(),
            assess: "gpt-3.5-turbo".to_string(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    api_key: Option<String>,
    base_url: Option<String>,
    models: ModelsConfig,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: Some("https://api.openai.com/v1".to_string()),
            models: ModelsConfig::default(),
        }
    }
}
fn load_config() -> Config {
    match fs::read_to_string(CONFIG_FILE) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            eprintln!(
                "{} Failed to parse {}: {}. Using defaults.",
                "WARNING:".yellow(),
                CONFIG_FILE,
                e
            );
            Config::default()
        }),
        Err(_) => Config::default(),
    }
}
fn create_default_config() -> anyhow::Result<()> {
    if PathBuf::from(CONFIG_FILE).exists() {
        println!(
            "{} {} already exists. Not overwriting.",
            "WARNING:".yellow(),
            CONFIG_FILE
        );
        return Ok(());
    }
    let default_config = Config::default();
    let content = serde_json::to_string_pretty(&default_config)?;
    fs::write(CONFIG_FILE, content)?;
    println!(
        "{} Created default configuration file: {}",
        "INFO:".cyan(),
        CONFIG_FILE
    );
    Ok(())
}
#[derive(Debug, Serialize)]
struct SystemInfo {
    os: String,
    os_version: String,
    current_directory: String,
    home_directory: String,
    desktop_path: String,
    shell: String,
    user: String,
    is_admin: String,
}
fn gather_system_info() -> SystemInfo {
    let os = std::env::consts::OS.to_string();
    let os_version = if os == "windows" {
        "unknown".to_string()
    } else {
        String::from_utf8(
            Command::new("uname")
                .arg("-r")
                .output()
                .map(|o| o.stdout)
                .unwrap_or_default(),
        )
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let current_directory = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let home_directory = dirs::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let desktop_path = format!("{}/Desktop", home_directory);
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let is_admin = if cfg!(windows) {
        "unknown".to_string()
    } else {
        if unsafe { libc::geteuid() } == 0 {
            "yes"
        } else {
            "no"
        }
            .to_string()
    };
    SystemInfo {
        os,
        os_version,
        current_directory,
        home_directory,
        desktop_path,
        shell,
        user,
        is_admin,
    }
}
#[derive(Clone)]
struct LLMClient {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::blocking::Client,
}
impl LLMClient {
    fn new(api_key: Option<String>, base_url: Option<String>, model: String) -> Option<Self> {
        let api_key = api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok())?;
        let base_url = base_url
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();
        Some(Self {
            api_key,
            base_url,
            model,
            client: reqwest::blocking::Client::new(),
        })
    }
    fn chat_completion(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> anyhow::Result<Option<String>> {
        let url = format!("{}/chat/completions", self.base_url);
        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            temperature,
        };
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            eprintln!("{} HTTP error {}: {}", "ERROR:".red(), status, text);
            return Ok(None);
        }
        let result: ChatCompletionResponse = response.json()?;
        Ok(result.choices.first().map(|c| c.message.content.clone()))
    }
}
#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}
#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessageResponse,
}
#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}
struct Optimizer {
    llm_client: Option<LLMClient>,
}
impl Optimizer {
    const SYSTEM_PROMPT: &'static str = "\
        You are an assistant that rewrites vague user commands into precise, context-aware instructions.\n\
        Incorporate the provided system environment information to make the command fully self-contained.\n\
        Output only the rewritten command description, no extra text.";
    fn new(llm_client: Option<LLMClient>) -> Self {
        Self { llm_client }
    }
    fn optimize(&self, user_input: &str, sys_info: &SystemInfo) -> String {
        println!("\n{}", "--- Optimize ---".bold().purple());
        println!("{} Original input: {}", "INFO:".cyan(), user_input);
        if let Some(client) = &self.llm_client {
            match self.optimize_with_llm(client, user_input, sys_info) {
                Ok(opt) => {
                    println!("{} Optimized: {}", "INFO:".cyan(), opt);
                    return opt;
                }
                Err(e) => {
                    eprintln!(
                        "{} Optimize via LLM failed: {}. Falling back to template.",
                        "WARNING:".yellow(),
                        e
                    );
                }
            }
        }
        self.optimize_with_template(user_input, sys_info)
    }
    fn optimize_with_llm(
        &self,
        client: &LLMClient,
        user_input: &str,
        sys_info: &SystemInfo,
    ) -> anyhow::Result<String> {
        let sys_info_json = serde_json::to_string_pretty(sys_info)?;
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Self::SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "System Info:\n{}\n\nUser Input: {}",
                    sys_info_json, user_input
                ),
            },
        ];
        let response = client.chat_completion(&messages, 0.3)?;
        response.ok_or_else(|| anyhow::anyhow!("Empty LLM response"))
    }
    fn optimize_with_template(&self, user_input: &str, sys_info: &SystemInfo) -> String {
        let mut optimized = user_input.to_string();
        if user_input.to_lowercase().contains("desktop") {
            optimized = optimized.replace("desktop", &sys_info.desktop_path);
        }
        println!("{} Optimized (template): {}", "INFO:".cyan(), optimized);
        optimized
    }
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ExecutionPlan {
    commands: Vec<String>,
    is_destructive: bool,
    requires_sudo: bool,
    description: String,
}
struct Refiner {
    llm_client: Option<LLMClient>,
}
impl Refiner {
    const SYSTEM_PROMPT: &'static str = "\
        You are an assistant that translates natural language instructions into a JSON object containing shell commands.\n\
        The JSON must have the following structure:\n\
        {\n\
          \"commands\": [\"list\", \"of\", \"shell\", \"commands\", \"to\", \"execute\", \"in\", \"order\"],\n\
          \"is_destructive\": true/false,\n\
          \"requires_sudo\": true/false,\n\
          \"description\": \"brief summary of what will be done\"\n\
        }\n\n\
        Rules:\n\
        - Each command must be a single string, properly escaped.\n\
        - \"is_destructive\" is true if the command permanently deletes/modifies files outside temporary directories, formats disks, etc.\n\
        - \"requires_sudo\" is true if the command likely needs administrative privileges.\n\
        - Output ONLY valid JSON, no additional text.";
    fn new(llm_client: Option<LLMClient>) -> Self {
        Self { llm_client }
    }
    fn refine(&self, optimized_prompt: &str) -> Option<ExecutionPlan> {
        println!("\n{}", "--- Refine ---".bold().purple());
        println!("{} Input prompt: {}", "INFO:".cyan(), optimized_prompt);
        let json_str = if let Some(client) = &self.llm_client {
            match self.refine_with_llm(client, optimized_prompt) {
                Ok(Some(s)) => s,
                Ok(None) => {
                    eprintln!("{} Refine via LLM returned empty.", "ERROR:".red());
                    return None;
                }
                Err(e) => {
                    eprintln!(
                        "{} Refine via LLM failed: {}. Falling back to rule-based.",
                        "WARNING:".yellow(),
                        e
                    );
                    self.refine_with_rules(optimized_prompt)
                }
            }
        } else {
            eprintln!(
                "{} Refine using rule-based fallback (LLM not configured).",
                "WARNING:".yellow()
            );
            self.refine_with_rules(optimized_prompt)
        };
        match serde_json::from_str::<ExecutionPlan>(&json_str) {
            Ok(plan) => {
                println!("{} Refine successful.", "SUCCESS:".green());
                println!(
                    "{} JSON: {}",
                    "INFO:".cyan(),
                    serde_json::to_string_pretty(&plan).unwrap()
                );
                Some(plan)
            }
            Err(e) => {
                eprintln!("{} Invalid JSON from Refine: {}", "ERROR:".red(), e);
                None
            }
        }
    }
    fn refine_with_llm(
        &self,
        client: &LLMClient,
        prompt: &str,
    ) -> anyhow::Result<Option<String>> {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Self::SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        client.chat_completion(&messages, 0.0)
    }
    fn refine_with_rules(&self, prompt: &str) -> String {
        let lower = prompt.to_lowercase();
        let mut commands = Vec::new();
        let is_destructive = false;
        let requires_sudo = prompt.contains("sudo");
        if lower.contains("mkdir") {
            if let Some(cap) = Regex::new(r#"mkdir\s+(["']?)([^"']+?)\1"#).unwrap().captures(prompt) {
                let path = &cap[2];
                commands.push(format!("mkdir -p {}", shell_escape(path)));
            }
        } else if lower.contains("touch") {
            if let Some(cap) = Regex::new(r#"touch\s+(["']?)([^"']+?)\1"#).unwrap().captures(prompt) {
                let path = &cap[2];
                commands.push(format!("touch {}", shell_escape(path)));
            }
        } else if lower.contains("echo") {
            commands.push(prompt.to_string());
        } else if lower.contains("ls") {
            commands.push("ls -la".to_string());
        }
        if commands.is_empty() {
            commands.push(format!("echo 'Command not understood: {}'", shell_escape(prompt)));
        }
        let plan = ExecutionPlan {
            commands,
            is_destructive,
            requires_sudo,
            description: format!("Fallback rule-based execution for: {}", prompt),
        };
        serde_json::to_string(&plan).unwrap()
    }
}
fn shell_escape(s: &str) -> String {
    if s.contains('\'') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        format!("'{}'", s)
    }
}
struct Executor {
    dry_run: bool,
    destructive_patterns: Vec<Regex>,
}
impl Executor {
    fn new(dry_run: bool) -> Self {
        let patterns = vec![
            r"\brm\s+-rf\b",
            r"\bdd\s+if=",
            r"\bmkfs\b",
            r"\bformat\b",
            r"\bdel\b",
            r"\bdel\s+/[fFq]\b",
            r"\bDROP\s+TABLE\b",
        ];
        Self {
            dry_run,
            destructive_patterns: patterns
                .into_iter()
                .map(|p| Regex::new(p).unwrap())
                .collect(),
        }
    }
    fn execute(&self, plan: &ExecutionPlan) -> (bool, Vec<CommandResult>) {
        println!("\n{}", "--- Carry out ---".bold().purple());
        if plan.commands.is_empty() {
            eprintln!("{} No commands to execute.", "ERROR:".red());
            return (false, vec![]);
        }
        if plan.is_destructive || self.has_destructive_command(&plan.commands) {
            println!(
                "{} This operation is marked as DESTRUCTIVE or contains dangerous patterns.",
                "WARNING:".yellow()
            );
            if !confirm("Do you really want to continue? [y/N]: ") {
                println!("{} Execution cancelled by user.", "INFO:".cyan());
                return (false, vec![]);
            }
        }
        if plan.requires_sudo {
            println!(
                "{} Some commands may require sudo. Ensure you have proper permissions.",
                "WARNING:".yellow()
            );
        }
        if self.dry_run {
            println!("{} DRY RUN - commands would be executed:", "INFO:".cyan());
            for cmd in &plan.commands {
                println!("  $ {}", cmd.cyan());
            }
            return (true, vec![]);
        }
        let mut results = Vec::new();
        let mut overall_success = true;
        for cmd in &plan.commands {
            println!("{} Executing: {}", "INFO:".cyan(), cmd);
            let output = if cfg!(windows) {
                Command::new("cmd").args(["/C", cmd]).output()
            } else {
                Command::new("sh").arg("-c").arg(cmd).output()
            };
            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    let success = out.status.success();
                    let result = CommandResult {
                        command: cmd.clone(),
                        stdout: stdout.clone(),
                        stderr: stderr.clone(),
                        returncode: out.status.code().unwrap_or(-1),
                        success,
                    };
                    results.push(result);
                    if success {
                        println!("{} Command succeeded.", "SUCCESS:".green());
                        if !stdout.is_empty() {
                            println!("Output:\n{}", stdout);
                        }
                    } else {
                        eprintln!(
                            "{} Command failed (exit {}). Error:\n{}",
                            "ERROR:".red(),
                            out.status.code().unwrap_or(-1),
                            stderr
                        );
                        overall_success = false;
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("{} Execution exception: {}", "ERROR:".red(), e);
                    results.push(CommandResult {
                        command: cmd.clone(),
                        stdout: String::new(),
                        stderr: e.to_string(),
                        returncode: -1,
                        success: false,
                    });
                    overall_success = false;
                    break;
                }
            }
        }
        (overall_success, results)
    }
    fn has_destructive_command(&self, commands: &[String]) -> bool {
        commands.iter().any(|cmd| {
            self.destructive_patterns
                .iter()
                .any(|pat| pat.is_match(cmd))
        })
    }
}
#[derive(Debug, Clone)]
struct CommandResult {
    command: String,
    stdout: String,
    stderr: String,
    returncode: i32,
    success: bool,
}
fn confirm(prompt: &str) -> bool {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim().to_lowercase();
    input == "y" || input == "yes"
}
struct Assessor {
    llm_client: Option<LLMClient>,
}
impl Assessor {
    const SYSTEM_PROMPT: &'static str = "\
        You are an assistant that diagnoses shell command failures.\n\
        Given the original plan, the failed command, and the error message (stdout/stderr),\n\
        suggest a corrected list of commands to achieve the user's intent.\n\n\
        Output only a valid JSON array of corrected command strings.\n\
        Example: [\"mkdir -p ~/Desktop/folder\", \"touch ~/Desktop/folder/file.txt\"]\n\n\
        If the error is unfixable, output an empty array [].";
    fn new(llm_client: Option<LLMClient>) -> Self {
        Self { llm_client }
    }
    fn assess(
        &self,
        original_plan: &ExecutionPlan,
        failed_result: &CommandResult,
    ) -> Option<Vec<String>> {
        println!("\n{}", "--- Assess ---".bold().purple());
        if let Some(client) = &self.llm_client {
            match self.assess_with_llm(client, original_plan, failed_result) {
                Ok(Some(cmds)) => {
                    println!(
                        "{} Assess suggests corrected commands: {:?}",
                        "INFO:".cyan(),
                        cmds
                    );
                    return Some(cmds);
                }
                Ok(None) => {
                    eprintln!("{} Assess via LLM returned empty.", "WARNING:".yellow());
                }
                Err(e) => {
                    eprintln!(
                        "{} Assess via LLM failed: {}. Falling back to rules.",
                        "WARNING:".yellow(),
                        e
                    );
                }
            }
        }
        self.assess_with_rules(original_plan, failed_result)
    }
    fn assess_with_llm(
        &self,
        client: &LLMClient,
        original_plan: &ExecutionPlan,
        failed_result: &CommandResult,
    ) -> anyhow::Result<Option<Vec<String>>> {
        #[derive(Serialize)]
        struct AssessInput {
            original_plan: ExecutionPlan,
            failed_command: String,
            stdout: String,
            stderr: String,
            returncode: i32,
        }
        let input = AssessInput {
            original_plan: original_plan.clone(),
            failed_command: failed_result.command.clone(),
            stdout: failed_result.stdout.clone(),
            stderr: failed_result.stderr.clone(),
            returncode: failed_result.returncode,
        };
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Self::SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: serde_json::to_string_pretty(&input)?,
            },
        ];
        let response = client.chat_completion(&messages, 0.1)?;
        if let Some(text) = response {
            let cmds: Vec<String> = serde_json::from_str(&text)?;
            Ok(Some(cmds))
        } else {
            Ok(None)
        }
    }
    fn assess_with_rules(
        &self,
        original_plan: &ExecutionPlan,
        failed_result: &CommandResult,
    ) -> Option<Vec<String>> {
        let stderr = failed_result.stderr.to_lowercase();
        let command = &failed_result.command;
        if stderr.contains("permission denied") && !original_plan.requires_sudo {
            println!("{} Rule-based fix: adding sudo.", "INFO:".cyan());
            let corrected = original_plan
                .commands
                .iter()
                .map(|cmd| {
                    if cmd.trim().starts_with("sudo") {
                        cmd.clone()
                    } else {
                        format!("sudo {}", cmd)
                    }
                })
                .collect();
            return Some(corrected);
        }
        if stderr.contains("no such file or directory") {
            if command.contains("mkdir") && !command.contains("-p") {
                println!("{} Rule-based fix: adding -p flag to mkdir.", "INFO:".cyan());
                let corrected = vec![command.replace("mkdir", "mkdir -p")];
                return Some(corrected);
            }
        }
        if stderr.contains("command not found") {
            println!(
                "{} Command not found. Cannot automatically fix.",
                "WARNING:".yellow()
            );
            return None;
        }
        println!("{} No rule-based fix available.", "WARNING:".yellow());
        None
    }
}
struct ORCAProcessor {
    optimizer: Optimizer,
    refiner: Refiner,
    executor: Executor,
    assessor: Assessor,
    sys_info: SystemInfo,
    max_retries: usize,
}
impl ORCAProcessor {
    fn new(args: &Args) -> Self {
        let config = load_config();
        let api_key = args
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .or(config.api_key);
        let base_url = args
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .or(config.base_url);
        let optimize_model = args
            .optimize_model
            .clone()
            .or_else(|| std::env::var("ORCA_OPTIMIZE_MODEL").ok())
            .unwrap_or(config.models.optimize);
        let refine_model = args
            .refine_model
            .clone()
            .or_else(|| std::env::var("ORCA_REFINE_MODEL").ok())
            .unwrap_or(config.models.refine);
        let assess_model = args
            .assess_model
            .clone()
            .or_else(|| std::env::var("ORCA_ASSESS_MODEL").ok())
            .unwrap_or(config.models.assess);
        let optimize_client = LLMClient::new(api_key.clone(), base_url.clone(), optimize_model);
        let refine_client = LLMClient::new(api_key.clone(), base_url.clone(), refine_model);
        let assess_client = LLMClient::new(api_key, base_url, assess_model);
        let sys_info = gather_system_info();
        Self {
            optimizer: Optimizer::new(optimize_client),
            refiner: Refiner::new(refine_client),
            executor: Executor::new(args.dry_run),
            assessor: Assessor::new(assess_client),
            sys_info,
            max_retries: 3,
        }
    }
    fn process(&self, user_input: &str) -> bool {
        println!("\n{}", "=".repeat(60).bold());
        println!("{}", "O.R.C.A. Task Processor".bold().purple());
        println!("{}", "=".repeat(60).bold());
        let optimized_prompt = self.optimizer.optimize(user_input, &self.sys_info);
        let mut current_plan = match self.refiner.refine(&optimized_prompt) {
            Some(plan) => plan,
            None => {
                eprintln!("{} Refine stage failed. Exiting.", "ERROR:".red());
                return false;
            }
        };
        for attempt in 1..=self.max_retries {
            println!(
                "{} Attempt {}/{}",
                "INFO:".cyan(),
                attempt,
                self.max_retries
            );
            let (success, results) = self.executor.execute(&current_plan);
            if success {
                println!("{} All commands executed successfully.", "SUCCESS:".green());
                return true;
            }
            if attempt == self.max_retries {
                eprintln!(
                    "{} Max retries ({}) reached. Task failed.",
                    "ERROR:".red(),
                    self.max_retries
                );
                return false;
            }
            let failed_result = results.iter().find(|r| !r.success);
            if failed_result.is_none() {
                eprintln!(
                    "{} No specific command failure detected, but overall success false.",
                    "ERROR:".red()
                );
                return false;
            }
            let failed_result = failed_result.unwrap();
            let corrected_commands =
                self.assessor
                    .assess(&current_plan, failed_result);
            match corrected_commands {
                Some(cmds) => {
                    current_plan.commands = cmds;
                    println!("{} Retrying with corrected commands...", "INFO:".cyan());
                }
                None => {
                    eprintln!(
                        "{} Assess could not provide a fix. Aborting.",
                        "WARNING:".yellow()
                    );
                    return false;
                }
            }
        }
        false
    }
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.create_config {
        create_default_config()?;
        return Ok(());
    }
    let user_input = args.input.join(" ");
    let processor = ORCAProcessor::new(&args);
    let success = processor.process(&user_input);
    if success {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
