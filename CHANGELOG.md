# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [Unreleased]

### Added
- Comprehensive LLM Agent system with 5 specialized agents (market_data, technical, sentiment, strategy, risk)
- LangGraph integration for multi-agent coordination and workflow management
- GLM-4 (ZhipuAI) model integration for intelligent market analysis
- Reinforcement learning framework with Gymnasium-compliant A-share trading environment
- Support for PPO, DQN, A2C, SAC, and TD3 algorithms via Stable-Baselines3
- 8+ customizable reward functions for RL training (returns, sharpe, sortino, max_drawdown, etc.)
- Model management system with versioning, metadata tracking, and persistence
- Hyperparameter optimization framework (GridSearch, RandomSearch, BayesianOptimization)
- Rust execution engine foundation (50% complete) with order management, order book, and portfolio modules
- PyO3 bindings for Python-Rust integration
- XTP broker interface framework with simulation trading support
- Real-time monitoring and alerting system with configurable thresholds
- Streamlit-based web monitoring dashboard
- Data layer with AKShare and Tushare providers
- Time-series data storage support (DuckDB default, ClickHouse optional)
- 20+ technical indicators (SMA, EMA, MACD, RSI, KDJ, Bollinger Bands, ATR, OBV, STOCH, etc.)
- Event-driven backtest engine with A-share specific rules (T+1, price limits, trading hours)
- 8 built-in trading strategies (BuyAndHold, MovingAverageCross, DualThrust, RSIReversal, etc.)
- Portfolio management and multi-strategy backtesting
- Performance analysis with 20+ metrics (Sharpe, Sortino, Max Drawdown, Win Rate, etc.)
- HTML report generation for backtest results
- Comprehensive test suite (262 tests, 100% pass rate)
- 16 complete examples covering all major features
- Centralized configuration system using dataclasses
- Logging and time utilities
- Error handling and validation framework
- Risk control system with 3-layer protection
- API documentation and usage guides
- Installation and troubleshooting guides
- Development and verification scripts

### Changed
- Improved code quality with Black, isort, and flake8 formatting
- Enhanced test coverage from 41% to 42.47%
- Updated API documentation with corrected class names and observation space dimensions
- Fixed observation space in ASharesTradingEnv from 19 to 20 dimensions
- Standardized import paths across all modules
- Improved project structure and module organization
- Enhanced error messages and logging throughout the system
- Optimized technical indicator calculations for performance
- Refactored agent base class for better extensibility
- Updated configuration management for easier customization

### Fixed
- Fixed ASharesTradingEnv class name inconsistency (was AShareTradingEnv)
- Corrected observation space dimension calculation in RL environment (20 dimensions: 5 price features + 13 technical indicators + 2 account states)
- Fixed DualThrust strategy logic error (list vs float comparison)
- Fixed syntax error in agents/technical_agent/agent.py (exc_info=True() typo)
- Fixed import paths in 4 test files
- Fixed event queue initialization in integration tests
- Fixed Series vs DataFrame issues in hyperparameter optimization tests
- Fixed time utility functions for proper A-share trading hours handling
- Fixed portfolio calculation for accurate position tracking
- Fixed risk control validation logic
- Fixed data handler for missing values and edge cases
- Fixed backtest engine date handling and time zone issues
- Fixed strategy execution order and signal generation
- Fixed performance metric calculations for edge cases

### Removed
- Removed deprecated test files that are now in legacy_integration
- Removed unused imports and variables (flake8 cleanup)
- Removed redundant code after refactoring

### Security
- Added environment variable validation for API keys
- Implemented secure credential management for broker connections
- Added input validation for user-provided data
- Sanitized LLM outputs to prevent injection attacks

## [0.1.0] - 2025-01-30

### Added
- Initial project structure and architecture
- Core configuration system (settings.py, symbols.py, strategies.py)
- Utility modules (logging, time_utils, helpers)
- Backtest engine foundation with event-driven architecture
- Technical indicators calculation module (20+ indicators)
- A-share trading rules engine (T+1, price limits, trading hours, minimum order size)
- Performance analysis module with key metrics
- HTML report generation for backtest results
- 8 built-in trading strategies with examples
- Basic data collection framework
- Agent base classes and message protocol
- GLM-4 integration framework
- 5 specialized LLM agents (market_data, technical, sentiment, strategy, risk)
- Agent coordinator with LangGraph integration
- Gymnasium-compliant A-share trading environment for RL
- RL training framework (PPO, DQN, A2C, SAC, TD3)
- 8+ reward function implementations
- Model evaluation and comparison framework
- Hyperparameter tuning support
- Model versioning and persistence
- XTP broker interface foundation
- Monitoring and alerting system
- Rust execution engine (order management, order book, portfolio)
- Python-Rust bindings with PyO3
- 262 comprehensive tests covering all modules
- 16 complete examples demonstrating all features
- Complete documentation (README, API guides, tutorials)
- Development scripts (install, test, verify, format, lint)
- Docker support for containerized deployment

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Removed
- N/A (initial release)

### Security
- N/A (initial release)

---

## Version Classification

This project uses semantic versioning: MAJOR.MINOR.PATCH

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

## Release Schedule

- **Unreleased**: Changes currently in development
- **0.1.0**: Initial release with core functionality (2025-01-30)
- **0.2.0**: Planned - Enhanced RL model performance and additional strategies
- **0.3.0**: Planned - Complete Rust execution engine integration
- **1.0.0**: Planned - Production-ready with live trading support

## Project Status

- **Overall Progress**: 70% (28/40 tasks completed)
- **Test Pass Rate**: 100% (262/262 tests)
- **Code Coverage**: 42.47% (target: 70%)
- **Rust Engine**: 50% complete
- **Live Trading**: 40% complete

For detailed progress tracking, see [PROGRESS.md](PROGRESS.md)

## Module Completion Status

| Module | Status | Coverage |
|--------|--------|----------|
| Data Layer | ✅ Complete | 100% |
| Backtest Engine | ✅ Complete | 100% |
| LLM Agents | ✅ Complete | 100% |
| RL System | ✅ Complete | 100% |
| Execution Engine | 🔄 In Progress | 50% |
| Live Trading | 🔄 In Progress | 40% |

## Key Features by Version

### 0.1.0 (Current)
- Complete backtest system with A-share rules
- 20+ technical indicators
- 8 built-in strategies
- LLM Agent framework with 5 specialized agents
- RL training with PPO/DQN/A2C/SAC/TD3
- Model management and hyperparameter optimization
- Rust execution engine foundation
- XTP broker interface framework
- Monitoring and alerting system
- 262 tests (100% pass rate)
- 16 complete examples
- Comprehensive documentation

### Planned for 0.2.0
- Enhanced RL model performance
- Additional trading strategies
- Improved hyperparameter optimization
- Multi-factor strategy support
- Advanced risk management features
- Extended test coverage to 60%

### Planned for 0.3.0
- Complete Rust execution engine
- Performance optimization and benchmarking
- Advanced order types and execution algorithms
- Real-time data pipeline improvements
- Extended test coverage to 70%

### Planned for 1.0.0
- Full XTP C++ SDK integration
- Live trading with simulation mode
- Production monitoring dashboard
- Comprehensive compliance reporting
- 100% test coverage
- Complete documentation and tutorials

## Contributors

- Rowan (Project Lead & Developer)
- Claude Code (AI Assistant)

## Support

For detailed documentation, see:
- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - Architecture and development guide
- [QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [API.md](docs/API.md) - Complete API reference
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current project status
- [PROGRESS.md](PROGRESS.md) - Detailed progress tracking

## License

MIT License - See LICENSE file for details

---

**Note**: This changeline follows the [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) format and [Semantic Versioning](https://semver.org/lang/zh-CN/).
