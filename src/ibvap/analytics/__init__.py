"""Analytics rules: virtual fencing, behaviour and scene integrity.

Importing this package registers every built-in rule. Registration happens as
an import side effect in :mod:`ibvap.analytics.rules`, so it must be imported
here rather than left to whichever module happens to be loaded first - the
engine would otherwise build an empty rule set and a camera would run with all
of its analytics silently disabled.
"""

from ibvap.analytics import rules as _rules  # noqa: F401  (populates the registry)
from ibvap.analytics.base import (  # noqa: F401
    Rule,
    RuleContext,
    available_rules,
    build_rule,
    register_rule,
)
from ibvap.analytics.engine import AnalyticsEngine, EventGate  # noqa: F401

__all__ = [
    "AnalyticsEngine",
    "EventGate",
    "Rule",
    "RuleContext",
    "available_rules",
    "build_rule",
    "register_rule",
]
