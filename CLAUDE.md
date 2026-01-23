# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quantA** is an AI-powered quantitative trading system for A-share (Chinese stock market) that combines reinforcement learning with LLM agent technologies for mid-frequency/intraday trading.

### Technology Stack
- **Language**: Python 3.10+ with Rust (for performance-critical execution engine)
- **LLM**: GLM-4 (ZhipuAI) with LangChain/LangGraph
- **RL Framework**: Stable-Baselines3, Gymnasium
- **Data Storage**: DuckDB (default) / ClickHouse (time-series)
- **Broker Interface**: Huatai XTP (simulation available)

## Architecture

The system follows a layered architecture:

```
LLM Agent Decision Layer (5 agents) -> RL Strategy Layer -> Rust Execution Engine -> Data Infrastructure
```

### Core Components

1. **Data Layer** (`data/`) - Market data collection from AKShare/Tushare, time-series storage
2. **LLM Agents** (`agents/`) - Five specialized agents using GLM-4 for market analysis and strategy generation
3. **Reinforcement Learning** (`rl/`) - PPO/DQN algorithms with custom A-share trading environment
4. **Backtest Engine** (`backtest/`) - Event-driven backtesting with 20+ technical indicators and A-share rules
5. **Execution Engine** (`rust_engine/`) - High-performance Rust execution layer (planned)
6. **Live Trading** (`live/`) - Broker integration and monitoring (planned)

### A-Share Trading Rules

The system implements Chinese stock market specific rules in `backtest/engine/a_share_rules.py`:
- **T+1 rule**: Stocks bought today can only be sold tomorrow
- **Price limits**: +/-10% for main board, +/-20% for SME/STAR board
- **Trading hours**: 9:30-11:30, 13:00-15:00
- **Minimum order size**: 100 shares (1 lot)

## Common Development Commands

### Testing
```bash
make test              # Run all tests
make test-cov          # Run with coverage report (requires 70% minimum)
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-backtest     # Backtest-specific tests
make test-agents       # Agent-specific tests
make test-rl           # RL-specific tests
```

### Code Quality
```bash
make format            # Format code with black and isort
make format-check      # Check formatting without making changes
make lint              # Run flake8 and mypy type checking
```

### Running Examples
```bash
make run               # Run backtest example
make run-agent         # Run agent example
make monitor           # Start monitoring dashboard (Streamlit)
make api               # Start API server (uvicorn)
```

### Database Operations
```bash
make db-init           # Initialize database
make db-backup         # Backup database
```

## Configuration

All configuration is centralized in `/config/settings.py` using dataclasses:
- `market` - Trading hours, price limits, T+1 rules
- `database` - DuckDB/ClickHouse settings
- `llm` - GLM-4 model configuration and API keys
- `rl` - Reinforcement learning parameters
- `execution` - Broker interfaces and risk limits
- `data` - Data source settings (Tushare, AKShare)

Required environment variables in `.env`:
```bash
ZHIPUAI_API_KEY=your_key_here
TUSHARE_TOKEN=your_token_here
# Optional: XTP account credentials for live trading
```

## Key Entry Points

- `examples/backtest_example.py` - Basic backtesting demonstration
- `examples/agent_example.py` - LLM Agent usage
- `backtest/engine/backtest.py` - Main backtest engine
- `agents/base/coordinator.py` - Agent coordination system
- `rl/envs/a_share_trading_env.py` - RL trading environment (Gymnasium)

## Testing Guidelines

- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.backtest`, etc.
- Test coverage minimum: 70% (enforced in CI)
- Test configuration in `pytest.ini` with predefined markers for different test categories

## Important Notes for Development

1. **Project Status**: Phase 1-3 complete (40% overall). See `PROGRESS.md` for detailed status.
2. **Rust Engine**: Planned for 10-100x performance improvement over Python execution
3. **A-Share Rules**: Always enforce T+1, price limits, and trading hours in any trading logic
4. **Agent Coordination**: Agents use structured prompts and JSON responses via LangGraph
5. **Logging**: Use `utils/logging.py` for consistent logging across modules
6. **Configuration**: Always use `config/settings.py` - do not hardcode values

## File Structure Notes

- `/config/` - Centralized configuration using dataclasses
- `/utils/` - Shared utilities (logging, time helpers, general helpers)
- `/backtest/engine/` - Core backtesting logic including A-share rules engine
- `/agents/base/` - Agent base classes and coordinator
- `/docs/` - Comprehensive documentation including architecture guides
