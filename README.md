# O.R.C.A
**Optimize · Refine · Carry out · Assess**

A lightweight, cross-platform, Rust-powered intelligent task processor that translates natural language into safe, structured system commands with self-healing capabilities.

---

## Overview
O.R.C.A is a four-stage AI agent designed to understand user intent, enhance context awareness, generate structured execution plans, safely run tasks, and automatically diagnose and fix failures.

Built for security, stability, and scalability, O.R.C.A brings modern LLM-driven automation to desktop and server environments.

---

## Features
- **Four-Stage Pipeline**: Optimize → Refine → Carry out → Assess
- **Structured Command Planning**: Strict JSON-based execution plan
- **Dangerous Command Detection**: rm -rf, mkfs, dd, and other high-risk operations
- **Self-Reflection & Auto-Repair**: Automatic retry with corrected commands
- **Cross-Platform**: Linux / macOS / Windows
- **LLM Agnostic**: Compatible with OpenAI, DeepSeek, and other OpenAI-compatible APIs
- **Pure Rust Core**: High performance, memory-safe, and zero extra runtime

---

## Architecture
```
User Input
    ↳ Optimize    (enhance prompt with system environment)
    ↳ Refine      (generate structured execution plan)
    ↳ Carry out   (secure execution with user confirmation)
    ↳ Assess      (diagnose errors and repair commands)
```

---

## Quick Start
### Install
```bash
cargo install orca-cli
```

### Configure
```bash
orca --create-config
```

### Run
```bash
orca "Create a project folder on the desktop and initialize a Python script"
```

---

## Configuration
O.R.C.A automatically loads `config.json` in the current directory.

```json
{
  "api_key": "sk-xxxxxxxxxxxxxxxxxxxx",
  "base_url": "https://api.deepseek.com/v1",
  "models": {
    "optimize": "deepseek-chat",
    "refine": "deepseek-chat",
    "assess": "deepseek-chat"
  }
}
```

---

## License
MIT
