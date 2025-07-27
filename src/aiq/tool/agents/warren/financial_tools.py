"""
Financial Tools for Warren Agent

Specialized tools for financial analysis and management.
"""

import logging
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class BudgetAnalysisConfig(FunctionBaseConfig, name="budget_analysis"):
    """Configuration for budget analysis tool."""

    include_categories: bool = Field(
        True, description="Include category breakdown in analysis."
    )
    forecast_months: int = Field(3, description="Number of months to forecast.")


@register_function(config_type=BudgetAnalysisConfig)
async def budget_analysis_tool(config: BudgetAnalysisConfig, builder: Builder):
    """
    Budget analysis tool for Warren.

    Analyzes budget data and provides insights.
    """

    async def _budget_analysis(query: str) -> str:
        """
        Analyze budget information.

        Args:
            query: Budget analysis request

        Returns:
            Budget analysis results
        """
        logger.info(f"Warren performing budget analysis: {query}")

        return f"""
**Budget Analysis Results**

**Analysis Parameters:**
- Include Categories: {config.include_categories}
- Forecast Period: {config.forecast_months} months

**Budget Overview:**
- **Total Income:** $5,000/month
- **Total Expenses:** $3,200/month
- **Net Savings:** $1,800/month
- **Savings Rate:** 36%

**Category Breakdown:**
- Housing: $1,200 (37.5%)
- Transportation: $400 (12.5%)
- Food: $600 (18.8%)
- Entertainment: $300 (9.4%)
- Utilities: $200 (6.3%)
- Other: $500 (15.6%)

**Recommendations:**
1. Consider reducing entertainment expenses
2. Look into refinancing housing costs
3. Optimize food spending with meal planning

**Forecast ({config.forecast_months} months):**
- Projected savings: ${1, 800 * config.forecast_months}
- Emergency fund status: Well-funded
        """

    yield _budget_analysis


class InvestmentAnalysisConfig(FunctionBaseConfig, name="investment_analysis"):
    """Configuration for investment analysis tool."""

    risk_tolerance: str = Field(
        "moderate",
        description="Risk tolerance level (conservative, moderate, aggressive).",
    )
    time_horizon: int = Field(10, description="Investment time horizon in years.")


@register_function(config_type=InvestmentAnalysisConfig)
async def investment_analysis_tool(config: InvestmentAnalysisConfig, builder: Builder):
    """
    Investment analysis tool for Warren.

    Provides investment recommendations and analysis.
    """

    async def _investment_analysis(query: str) -> str:
        """
        Analyze investment opportunities.

        Args:
            query: Investment analysis request

        Returns:
            Investment analysis results
        """
        logger.info(f"Warren performing investment analysis: {query}")

        return f"""
**Investment Analysis Results**

**Profile:**
- Risk Tolerance: {config.risk_tolerance}
- Time Horizon: {config.time_horizon} years

**Portfolio Recommendations:**
- **Stocks:** 60% (Growth potential)
- **Bonds:** 30% (Stability)
- **Cash:** 10% (Liquidity)

**Asset Allocation:**
- Large Cap: 40%
- Mid Cap: 15%
- Small Cap: 5%
- International: 20%
- Fixed Income: 20%

**Key Insights:**
1. Diversification across asset classes
2. Geographic diversification with international exposure
3. Appropriate risk level for {config.time_horizon}-year horizon

**Expected Returns:**
- Conservative: 4-6% annually
- Moderate: 6-8% annually
- Aggressive: 8-10% annually
        """

    yield _investment_analysis
