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

### Environment Setup
```bash
# Set Python path (required for imports from project root)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install dependencies
make install           # Install core dependencies
make install-dev       # Install development tools (pytest, black, etc.)
make venv              # Create virtual environment
```

### Testing
```bash
make test              # Run all tests (262 tests, 100% pass rate)
make test-cov          # Run with coverage report (requires 70% minimum, currently 42.47%)
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-backtest     # Backtest-specific tests
make test-agents       # Agent-specific tests
make test-rl           # RL-specific tests

# Run specific test files
pytest tests/backtest/test_indicators.py -v

# Run tests by marker
pytest -m "unit" -v                    # Unit tests only
pytest -m "integration and not slow"  # Integration tests, excluding slow ones
pytest -m "requires_data" -v           # Tests requiring external data
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

# Individual examples
python examples/backtest_example.py              # Basic backtesting
python examples/strategy_guide_example.py        # 5 built-in strategies
python examples/rl_training_guide.py             # RL training (PPO/DQN)
python examples/rl_complete_workflow.py          # Full RL workflow
python examples/advanced_strategy_example.py     # Multi-factor strategies
python examples/agent_coordinator_example.py     # Multi-agent coordination
```

### Docker
```bash
make docker-build      # Build Docker image
make docker-run        # Run container in background
make docker-stop       # Stop and remove container
```

### Database Operations
```bash
make db-init           # Initialize database
make db-backup         # Backup database
```

## Configuration

All configuration is centralized in `config/settings.py` using dataclasses:
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
- `examples/rl_training_guide.py` - RL training with PPO/DQN
- `examples/rl_complete_workflow.py` - Full RL workflow with model management
- `backtest/engine/backtest.py` - Main backtest engine
- `agents/base/coordinator.py` - Agent coordination system
- `rl/envs/a_share_trading_env.py` - RL trading environment (Gymnasium)
- `rl/training/trainer.py` - RL model training (PPO/DQN/A2C/SAC/TD3)
- `rl/evaluation/model_evaluator.py` - Model evaluation and comparison

## RL Model Management

The system includes a complete model management framework:

```python
from rl.models import ModelManager

# Create model manager
manager = ModelManager(models_dir="models")

# Save trained model with metadata
version = manager.save_model(
    model=trained_model,
    algorithm="ppo",
    metadata={"total_timesteps": 50000, "reward": 1234.56}
)

# Load model by version
model = manager.load_model(version.version_id)

# List all versions
versions = manager.list_versions(algorithm="ppo")

# Compare models
evaluator = ModelEvaluator()
results = evaluator.compare_models([model1, model2], env)
```

## Testing Guidelines

- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.backtest`, etc.
- Test coverage minimum: 70% (enforced in CI)
- Test configuration in `pytest.ini` with predefined markers for different test categories

## Important Notes for Development

1. **Project Status**: Phase 1-4 complete (70% overall). See `PROGRESS.md` for detailed status.
   - **Test Status**: 262/262 tests passing (100%)
   - **Code Coverage**: 42.47% (target: 70%)
   - **Rust Engine**: 50% complete (core modules implemented, integration pending)

2. **Python Path**: Always set `PYTHONPATH` to project root before running scripts:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **A-Share Rules**: Always enforce T+1, price limits, and trading hours in any trading logic
   - T+1: Stocks bought today can only be sold tomorrow
   - Price limits: ±10% main board, ±20% SME/STAR board
   - Min order size: 100 shares (1 lot)

4. **Agent Coordination**: Agents use structured prompts and JSON responses via LangGraph
   - 5 specialized agents: market_data, technical, sentiment, strategy, risk
   - Coordinator in `agents/base/coordinator.py` manages message routing

5. **RL Environment**: Gymnasium-compliant environment with 19-20 dimensional observation space
   - Actions: 0=hold, 1=buy, 2=sell
   - Rewards: Configurable (8+ reward functions available)
   - Algorithms: PPO, DQN, A2C, SAC, TD3 supported

6. **Logging**: Use `utils/logging.py` for consistent logging across modules
   - Logs stored in `logs/quant_a.log`
   - Configure level in `config/settings.py`

7. **Configuration**: Always use `config/settings.py` - do not hardcode values
   - Market settings in `config/settings.py:market`
   - LLM config in `config/settings.py:llm` (requires ZHIPUAI_API_KEY)
   - Database in `config/settings.py:database`

## File Structure Notes

- `config/` - Centralized configuration using dataclasses
  - `settings.py` - Main configuration (market, database, LLM, RL, execution, data, logging)
  - `symbols.py` - Stock universe configuration
  - `strategies.py` - Strategy parameter defaults

- `utils/` - Shared utilities (logging, time helpers, general helpers)

- `backtest/` - Event-driven backtesting system
  - `engine/` - Core engine (backtest.py, indicators.py, a_share_rules.py, event_engine.py)
  - `metrics/` - Performance analysis and HTML report generation
  - `portfolio/` - Multi-strategy portfolio backtesting

- `agents/` - LLM Agent system with 5 specialized agents
  - `base/` - Agent base classes, coordinator, GLM-4/LangGraph integration
  - `market_data_agent/`, `technical_agent/`, `sentiment_agent/`, `strategy_agent/`, `risk_agent/`

- `rl/` - Reinforcement learning framework
  - `envs/` - Gymnasium-compliant A-share trading environment
  - `training/` - Trainers for PPO/DQN/A2C/SAC/TD3
  - `rewards/` - 8+ reward function implementations
  - `evaluation/` - Model evaluation and comparison

- `data/` - Market data collection and storage
  - `market/sources/` - Data providers (AKShare, Tushare)
  - `market/storage/` - Time-series database (DuckDB/ClickHouse)

- `live/` - Real-time trading and monitoring
  - `brokers/` - XTP broker interface (simulation available)
  - `monitoring/` - Alerting and web dashboard (Streamlit)

- `rust_engine/` - High-performance execution engine (50% complete)
  - `src/` - Order management, portfolio, execution (PyO3 bindings)

- `docs/` - Comprehensive documentation including architecture guides
