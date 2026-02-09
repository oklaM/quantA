#!/usr/bin/env python
"""Analyze test coverage and generate improvement plan."""

import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_coverage_output():
    """Parse coverage output from pytest-cov."""
    # Read the terminal output or use the coverage data
    coverage_file = Path("/Users/rowan/Projects/quantA/.coverage")

    # We'll parse the terminal output format
    coverage_data = {
        'agents/base/glm4_integration.py': 0,
        'agents/base/langgraph_integration.py': 0,
        'agents/market_data_agent/agent.py': 0,
        'agents/risk_agent/agent.py': 0,
        'agents/sentiment_agent/agent.py': 0,
        'agents/strategy_agent/agent.py': 0,
        'agents/technical_agent/agent.py': 0,
        'backtest/metrics/performance.py': 0,
        'backtest/metrics/report.py': 0,
        'live/brokers/xtp_broker.py': 0,
        'live/monitoring/alerting.py': 0,
        'live/monitoring/monitor.py': 0,
        'live/monitoring/web_dashboard.py': 0,
        'rl/evaluation/model_evaluator.py': 0,
        'rl/training/trainer.py': 21,
        'rl/optimization/hyperparameter_tuning.py': 22,
        'backtest/engine/rust_engine.py': 25,
        'data/market/storage/timeseries_db.py': 27,
        'monitoring/metrics.py': 29,
        'utils/performance.py': 24,
        'monitoring/alerts.py': 42,
        'data/market/sources/akshare_provider.py': 41,
        'data/market/sources/tushare_provider.py': 43,
        'data/market/collector.py': 33,
    }

    return coverage_data

def categorize_by_importance(filepath):
    """Categorize modules by business importance."""
    filepath_lower = filepath.lower()

    # Critical core business logic
    if any(x in filepath_lower for x in [
        'backtest/engine/backtest',
        'backtest/engine/strategies',
        'backtest/engine/portfolio',
        'backtest/engine/execution',
        'backtest/engine/indicators',
        'rl/envs/a_share_trading_env',
        'rl/training/trainer',
        'rl/evaluation/model_evaluator',
        'data/market/sources',
        'data/market/storage',
    ]):
        return 'critical'

    # High importance - supporting core features
    if any(x in filepath_lower for x in [
        'agents/',
        'backtest/metrics/',
        'backtest/optimization/',
        'rl/optimization/',
        'monitoring/',
        'trading/risk',
    ]):
        return 'high'

    # Medium importance - utilities and helpers
    if any(x in filepath_lower for x in [
        'utils/',
        'config/',
    ]):
        return 'medium'

    # Low importance - peripheral features
    if any(x in filepath_lower for x in [
        'live/monitoring/web_dashboard',
        'examples/',
    ]):
        return 'low'

    return 'medium'

def estimate_test_difficulty(filepath):
    """Estimate testing difficulty based on module complexity."""
    filepath_lower = filepath.lower()

    # Easy - pure functions, minimal dependencies
    if any(x in filepath_lower for x in [
        'utils/helpers',
        'utils/time_utils',
        'config/',
    ]):
        return 'easy'

    # Medium - some external dependencies, moderate complexity
    if any(x in filepath_lower for x in [
        'backtest/metrics/',
        'backtest/engine/indicators',
        'data/market/sources',
        'monitoring/alerts',
        'monitoring/metrics',
    ]):
        return 'medium'

    # Hard - complex logic, many dependencies, async, or external APIs
    if any(x in filepath_lower for x in [
        'agents/',
        'rl/',
        'backtest/engine/backtest',
        'backtest/engine/execution',
        'backtest/engine/portfolio',
        'live/',
        'data/market/storage',
    ]):
        return 'hard'

    return 'medium'

def calculate_priority_score(coverage_pct, importance):
    """Calculate priority score for testing."""
    # Base score from coverage gap (0-100)
    coverage_gap = 100 - coverage_pct

    # Importance multiplier
    importance_multipliers = {
        'critical': 3.0,
        'high': 2.0,
        'medium': 1.0,
        'low': 0.5
    }

    return coverage_gap * importance_multipliers.get(importance, 1.0)

def generate_improvement_plan():
    """Generate comprehensive test coverage improvement plan."""

    print("=" * 100)
    print("QUANTA TEST COVERAGE ANALYSIS & IMPROVEMENT PLAN")
    print("=" * 100)
    print()

    # Current overall coverage from the test output
    total_coverage = 58.25
    target_coverage = 70.0
    coverage_gap = target_coverage - total_coverage

    print(f"Current Overall Coverage: {total_coverage:.2f}%")
    print(f"Target Coverage: {target_coverage:.2f}%")
    print(f"Coverage Gap: {coverage_gap:.2f}%")
    print()

    # Define all modules with their coverage from the test output
    modules = [
        # 0% coverage - completely untested
        {'path': 'agents/base/glm4_integration.py', 'coverage': 0, 'lines': 103},
        {'path': 'agents/base/langgraph_integration.py', 'coverage': 0, 'lines': 151},
        {'path': 'agents/market_data_agent/agent.py', 'coverage': 0, 'lines': 133},
        {'path': 'agents/risk_agent/agent.py', 'coverage': 0, 'lines': 93},
        {'path': 'agents/sentiment_agent/agent.py', 'coverage': 0, 'lines': 67},
        {'path': 'agents/strategy_agent/agent.py', 'coverage': 0, 'lines': 80},
        {'path': 'agents/technical_agent/agent.py', 'coverage': 0, 'lines': 122},
        {'path': 'backtest/metrics/performance.py', 'coverage': 0, 'lines': 107},
        {'path': 'backtest/metrics/report.py', 'coverage': 0, 'lines': 104},
        {'path': 'live/brokers/xtp_broker.py', 'coverage': 0, 'lines': 187},
        {'path': 'live/monitoring/alerting.py', 'coverage': 0, 'lines': 207},
        {'path': 'live/monitoring/monitor.py', 'coverage': 0, 'lines': 126},
        {'path': 'live/monitoring/web_dashboard.py', 'coverage': 0, 'lines': 142},
        {'path': 'rl/evaluation/model_evaluator.py', 'coverage': 0, 'lines': 202},
        {'path': 'trading/risk.py', 'coverage': 0, 'lines': 67},
        {'path': 'config/strategies.py', 'coverage': 0, 'lines': 115},

        # Very low coverage (< 30%)
        {'path': 'rl/training/trainer.py', 'coverage': 21, 'lines': 214},
        {'path': 'rl/optimization/hyperparameter_tuning.py', 'coverage': 22, 'lines': 246},
        {'path': 'backtest/engine/rust_engine.py', 'coverage': 25, 'lines': 72},
        {'path': 'data/market/storage/timeseries_db.py', 'coverage': 27, 'lines': 217},
        {'path': 'monitoring/metrics.py', 'coverage': 29, 'lines': 187},

        # Low coverage (30-50%)
        {'path': 'utils/performance.py', 'coverage': 24, 'lines': 207},
        {'path': 'monitoring/alerts.py', 'coverage': 42, 'lines': 253},
        {'path': 'data/market/sources/akshare_provider.py', 'coverage': 41, 'lines': 145},
        {'path': 'data/market/sources/tushare_provider.py', 'coverage': 43, 'lines': 136},
        {'path': 'data/market/collector.py', 'coverage': 33, 'lines': 103},
        {'path': 'agents/collaboration.py', 'coverage': 53, 'lines': 36},
        {'path': 'backtest/engine/indicators.py', 'coverage': 53, 'lines': 146},
        {'path': 'backtest/engine/a_share_rules.py', 'coverage': 58, 'lines': 149},
        {'path': 'backtest/optimization/optimizer.py', 'coverage': 59, 'lines': 246},
        {'path': 'data/market/data_manager.py', 'coverage': 70, 'lines': 46},
    ]

    # Add metadata
    for module in modules:
        module['importance'] = categorize_by_importance(module['path'])
        module['difficulty'] = estimate_test_difficulty(module['path'])
        module['priority_score'] = calculate_priority_score(module['coverage'], module['importance'])
        module['gap'] = 100 - module['coverage']

    # Sort by priority score
    modules.sort(key=lambda x: x['priority_score'], reverse=True)

    print("TOP 20 LOWEST COVERAGE MODULES (Ranked by Priority)")
    print("=" * 100)
    print()

    for i, module in enumerate(modules[:20], 1):
        print(f"{i:2d}. {module['path']}")
        print(f"    Coverage: {module['coverage']:5.1f}% | Gap: {module['gap']:5.1f}% | Lines: {module['lines']}")
        print(f"    Importance: {module['importance'].upper():<10} | Difficulty: {module['difficulty'].upper():<10} | Priority Score: {module['priority_score']:.1f}")
        print()

    # Group by priority tiers
    print()
    print("=" * 100)
    print("PRIORITY TIERS FOR TEST IMPROVEMENT")
    print("=" * 100)
    print()

    # Tier 1: High priority - Critical business logic with very low coverage
    tier1 = [m for m in modules if m['importance'] in ['critical', 'high'] and m['coverage'] < 30]
    print("TIER 1: HIGH PRIORITY (Critical + High Importance, < 30% coverage)")
    print("-" * 100)
    for module in tier1[:10]:
        print(f"  • {module['path']:<60} {module['coverage']:5.1f}% ({module['importance']}, {module['difficulty']})")
    print()

    # Tier 2: Medium priority - Important modules with moderate coverage gap
    tier2 = [m for m in modules if m['importance'] in ['critical', 'high'] and 30 <= m['coverage'] < 60]
    print("TIER 2: MEDIUM PRIORITY (Critical + High Importance, 30-60% coverage)")
    print("-" * 100)
    for module in tier2[:10]:
        print(f"  • {module['path']:<60} {module['coverage']:5.1f}% ({module['importance']}, {module['difficulty']})")
    print()

    # Tier 3: Lower priority - Medium importance or high coverage
    tier3 = [m for m in modules if m['importance'] == 'medium' or m['coverage'] >= 60]
    print("TIER 3: LOWER PRIORITY (Medium importance or > 60% coverage)")
    print("-" * 100)
    for module in tier3[:10]:
        print(f"  • {module['path']:<60} {module['coverage']:5.1f}% ({module['importance']}, {module['difficulty']})")
    print()

    # Specific recommendations
    print()
    print("=" * 100)
    print("DETAILED TEST ADDITION RECOMMENDATIONS")
    print("=" * 100)
    print()

    recommendations = [
        {
            'module': 'RL Training & Evaluation',
            'files': ['rl/training/trainer.py', 'rl/evaluation/model_evaluator.py'],
            'priority': 'CRITICAL',
            'reason': 'Core RL functionality completely untested. Training is essential for system.',
            'difficulty': 'Hard',
            'tests_to_add': [
                'Test PPO/DQN training workflow with mock environment',
                'Test model saving/loading',
                'Test model evaluation and comparison',
                'Test training callbacks and logging',
                'Test hyperparameter integration'
            ],
            'estimated_effort': '3-5 days',
            'coverage_gain': '15-20%'
        },
        {
            'module': 'LLM Agent System',
            'files': ['agents/market_data_agent/agent.py', 'agents/technical_agent/agent.py',
                     'agents/strategy_agent/agent.py', 'agents/risk_agent/agent.py'],
            'priority': 'HIGH',
            'reason': 'Agent system is key differentiator but has 0% coverage.',
            'difficulty': 'Hard - requires mocking GLM-4 API',
            'tests_to_add': [
                'Test agent initialization and configuration',
                'Test message processing with mocked LLM responses',
                'Test agent communication protocols',
                'Test error handling and retries',
                'Test agent coordination workflows'
            ],
            'estimated_effort': '5-7 days',
            'coverage_gain': '10-15%'
        },
        {
            'module': 'Performance Metrics & Reporting',
            'files': ['backtest/metrics/performance.py', 'backtest/metrics/report.py'],
            'priority': 'HIGH',
            'reason': 'Essential for analyzing backtest results. Pure calculation logic.',
            'difficulty': 'Medium - mostly calculation functions',
            'tests_to_add': [
                'Test Sharpe ratio calculation',
                'Test max drawdown calculation',
                'Test win rate and profit factor',
                'Test HTML report generation',
                'Test benchmark comparison'
            ],
            'estimated_effort': '2-3 days',
            'coverage_gain': '8-12%'
        },
        {
            'module': 'Data Storage Layer',
            'files': ['data/market/storage/timeseries_db.py'],
            'priority': 'HIGH',
            'reason': 'Critical infrastructure for market data persistence.',
            'difficulty': 'Medium - requires DuckDB/ClickHouse mocking',
            'tests_to_add': [
                'Test database connection and initialization',
                'Test time-series data insertion',
                'Test time-based queries',
                'Test data aggregation',
                'Test schema validation'
            ],
            'estimated_effort': '2-3 days',
            'coverage_gain': '6-10%'
        },
        {
            'module': 'Market Data Providers',
            'files': ['data/market/sources/akshare_provider.py', 'data/market/sources/tushare_provider.py'],
            'priority': 'MEDIUM-HIGH',
            'reason': 'Data ingestion needs testing. Requires API mocking.',
            'difficulty': 'Medium - requires mocking external APIs',
            'tests_to_add': [
                'Test data fetching for different symbols',
                'Test data format validation',
                'Test error handling for API failures',
                'Test rate limiting',
                'Test data caching'
            ],
            'estimated_effort': '2-3 days',
            'coverage_gain': '5-8%'
        },
        {
            'module': 'Technical Indicators',
            'files': ['backtest/engine/indicators.py'],
            'priority': 'MEDIUM',
            'reason': 'Core for trading signals. Already has some tests.',
            'difficulty': 'Easy - pure calculation functions',
            'tests_to_add': [
                'Test remaining MACD variants',
                'Test Bollinger Bands edge cases',
                'Test RSI overbought/oversold',
                'Test ATR calculation',
                'Test indicator combinations'
            ],
            'estimated_effort': '1-2 days',
            'coverage_gain': '5-8%'
        },
        {
            'module': 'A-Share Trading Rules',
            'files': ['backtest/engine/a_share_rules.py'],
            'priority': 'MEDIUM',
            'reason': 'Domain-specific logic critical for accuracy.',
            'difficulty': 'Easy - mostly validation logic',
            'tests_to_add': [
                'Test T+1 rule enforcement',
                'Test price limit validation',
                'Test trading hours validation',
                'Test minimum order size',
                'Test commission calculation'
            ],
            'estimated_effort': '1 day',
            'coverage_gain': '4-6%'
        },
        {
            'module': 'Monitoring & Alerting',
            'files': ['monitoring/alerts.py', 'monitoring/metrics.py'],
            'priority': 'MEDIUM',
            'reason': 'Important for production monitoring.',
            'difficulty': 'Medium',
            'tests_to_add': [
                'Test alert threshold checking',
                'Test notification sending',
                'Test metric collection',
                'Test alert history',
                'Test metric aggregation'
            ],
            'estimated_effort': '2 days',
            'coverage_gain': '4-6%'
        },
        {
            'module': 'RL Hyperparameter Tuning',
            'files': ['rl/optimization/hyperparameter_tuning.py'],
            'priority': 'MEDIUM',
            'reason': 'Important for model optimization.',
            'difficulty': 'Hard - complex optimization logic',
            'tests_to_add': [
                'Test grid search',
                'Test random search',
                'Test Bayesian optimization',
                'Test parameter space definition',
                'Test optimization result analysis'
            ],
            'estimated_effort': '2-3 days',
            'coverage_gain': '5-8%'
        },
        {
            'module': 'Agent Collaboration',
            'files': ['agents/collaboration.py'],
            'priority': 'LOW-MEDIUM',
            'reason': 'Supports multi-agent workflows.',
            'difficulty': 'Medium',
            'tests_to_add': [
                'Test message routing',
                'Test workflow orchestration',
                'Test parallel agent execution',
                'Test result aggregation'
            ],
            'estimated_effort': '1-2 days',
            'coverage_gain': '3-5%'
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['module']}")
        print(f"   Priority: {rec['priority']} | Difficulty: {rec['difficulty']} | Effort: {rec['estimated_effort']}")
        print(f"   Files: {', '.join(rec['files'])}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Coverage Gain: {rec['coverage_gain']}")
        print("   Tests to Add:")
        for test in rec['tests_to_add']:
            print(f"     - {test}")
        print()

    print()
    print("=" * 100)
    print("IMPLEMENTATION ROADMAP")
    print("=" * 100)
    print()

    # Phase 1: Quick wins (easy tests, high impact)
    print("PHASE 1: QUICK WINS (Week 1) - Target: +8-10% coverage")
    print("-" * 100)
    print("Focus on pure calculation functions and validation logic:")
    print("  1. Technical Indicators (backtest/engine/indicators.py)")
    print("  2. A-Share Trading Rules (backtest/engine/a_share_rules.py)")
    print("  3. Performance Metrics (backtest/metrics/performance.py)")
    print("  4. Risk Controls (trading/risk/controls.py)")
    print()

    # Phase 2: Core business logic
    print("PHASE 2: CORE BUSINESS LOGIC (Weeks 2-3) - Target: +12-15% coverage")
    print("-" * 100)
    print("Focus on critical trading functionality:")
    print("  1. RL Training (rl/training/trainer.py)")
    print("  2. RL Evaluation (rl/evaluation/model_evaluator.py)")
    print("  3. Data Storage (data/market/storage/timeseries_db.py)")
    print("  4. Market Data Providers (data/market/sources/)")
    print()

    # Phase 3: Complex integrations
    print("PHASE 3: COMPLEX INTEGRATIONS (Weeks 4-5) - Target: +8-10% coverage")
    print("-" * 100)
    print("Focus on systems requiring extensive mocking:")
    print("  1. LLM Agents (agents/*/agent.py)")
    print("  2. Agent Collaboration (agents/collaboration.py)")
    print("  3. Hyperparameter Tuning (rl/optimization/hyperparameter_tuning.py)")
    print("  4. Monitoring & Alerting (monitoring/alerts.py, monitoring/metrics.py)")
    print()

    # Phase 4: Reporting and polish
    print("PHASE 4: REPORTING & POLISH (Week 6) - Target: +3-5% coverage")
    print("-" * 100)
    print("Focus on reporting and remaining gaps:")
    print("  1. HTML Report Generation (backtest/metrics/report.py)")
    print("  2. Data Manager (data/market/data_manager.py)")
    print("  3. Agent Integration (agents/base/glm4_integration.py)")
    print("  4. Fix failing integration tests")
    print()

    print()
    print("=" * 100)
    print("EXPECTED OUTCOMES")
    print("=" * 100)
    print()
    print(f"Starting Coverage: {total_coverage:.2f}%")
    print(f"Phase 1 (Quick Wins): {total_coverage + 9:.2f}%")
    print(f"Phase 2 (Core Logic): {total_coverage + 9 + 13.5:.2f}%")
    print(f"Phase 3 (Complex): {total_coverage + 9 + 13.5 + 9:.2f}%")
    print(f"Phase 4 (Polish): {total_coverage + 9 + 13.5 + 9 + 4:.2f}%")
    print()
    print(f"Final Expected Coverage: {total_coverage + 35.5:.2f}%")
    print(f"Target Coverage: {target_coverage:.2f}%")
    print(f"Success: {'✓ YES' if total_coverage + 35.5 >= target_coverage else '✗ NO'}")
    print()

    print()
    print("=" * 100)
    print("TESTING BEST PRACTICES")
    print("=" * 100)
    print()
    print("1. Use pytest fixtures effectively:")
    print("   - Create reusable fixtures for common test data")
    print("   - Use pytest.mark.parametrize for data-driven tests")
    print("   - Mock external dependencies (LLM APIs, databases)")
    print()
    print("2. Test structure:")
    print("   - Unit tests: Test individual functions in isolation")
    print("   - Integration tests: Test module interactions")
    print("   - Edge cases: Test boundary conditions and error cases")
    print()
    print("3. Coverage goals per module:")
    print("   - Critical business logic: ≥ 80%")
    print("   - High importance: ≥ 70%")
    print("   - Medium importance: ≥ 60%")
    print("   - Low importance: ≥ 50%")
    print()
    print("4. Mocking strategy:")
    print("   - Use unittest.mock for LLM API calls")
    print("   - Use pytest fixtures for database connections")
    print("   - Create fake market data for backtesting tests")
    print()

    print()
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print()
    print("1. Run: make test-cov to establish baseline")
    print("2. Start with Phase 1 (Technical Indicators - easiest wins)")
    print("3. Create test file: tests/backtest/test_indicators_extended.py")
    print("4. Run tests incrementally: pytest tests/backtest/test_indicators_extended.py -v")
    print("5. Review coverage: pytest --cov=backtest/engine/indicators --cov-report=html")
    print("6. Iterate until Phase 1 complete")
    print()

if __name__ == '__main__':
    generate_improvement_plan()
