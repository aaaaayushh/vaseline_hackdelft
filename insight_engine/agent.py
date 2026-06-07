"""Financial Agent — the conversational layer over the Insight Engine.

The agent *plans and explains*; it never invents a number. Every figure it
quotes comes from a tool backed by the deterministic ``InsightEngine`` /
``SpendingAnalytics`` layer — the same computed values the dashboard renders.
This is the DESIGN.md §6 differentiator: a co-pilot that turns a passive
insight into an action.

Design (matches the claude-api skill guidance):

* ``claude-opus-4-8`` with **adaptive thinking**.
* A **manual tool-use loop** (not the auto runner) so we can *gate* the
  action tools — ``set_budget`` / ``cancel_subscription`` ask for human
  confirmation before they "execute" (simulated here).
* **Prompt caching** on the system prompt + tool definitions (stable prefix);
  the per-user context goes after, so it doesn't invalidate the cache.
* Read-only tools (insights, breakdowns, upcoming charges, decline history)
  auto-execute; their results are computed, never modelled.

Runs end-to-end only with ``ANTHROPIC_API_KEY`` set and ``anthropic`` installed
(``uv pip install anthropic``). Without them, ``FinancialAgent.available()`` is
False and the CLI prints a clear message — the rest of the engine is unaffected.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .engine import InsightEngine

MODEL = "claude-opus-4-8"

# Tools whose "execution" changes state. They are gated behind an explicit
# human confirmation in the manual loop; here the effect is simulated.
GATED_TOOLS = {"set_budget", "cancel_subscription"}


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Tool schemas (raw JSON — the manual-loop approach)
# --------------------------------------------------------------------------- #
TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_top_insights",
        "description": "Return this user's ranked insights (the dashboard feed): "
                       "title, explanation, severity and type for each. Call this "
                       "first to see what matters for the user right now.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "explain_insight",
        "description": "Return the full computed payload for one insight type so you "
                       "can explain the 'why' with exact numbers. Call this when the "
                       "user asks about a specific insight.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insight_type": {
                    "type": "string",
                    "enum": ["decline_shield", "overspend_alert", "subscription_radar",
                             "cashflow_forecast", "peer_benchmarking", "fx_fee_leakage"],
                },
            },
            "required": ["insight_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "category_spending",
        "description": "Monthly spend by category (history bars) plus a forecast for "
                       "next month. Use for 'how much did I spend on X' / trend questions.",
        "input_schema": {
            "type": "object",
            "properties": {"months": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "upcoming_charges",
        "description": "Known recurring charges due in the next 30 days, predicted from "
                       "cadence (merchant, amount, date).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "set_budget",
        "description": "Set a monthly budget for a category. This is a state-changing "
                       "action and requires user confirmation before it applies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "monthly_amount": {"type": "number"},
            },
            "required": ["category", "monthly_amount"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_subscription",
        "description": "Cancel a recurring subscription. State-changing — requires user "
                       "confirmation before it applies.",
        "input_schema": {
            "type": "object",
            "properties": {"merchant": {"type": "string"}},
            "required": ["merchant"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = (
    "You are Spending IQ, a personal-finance co-pilot inside a banking app. "
    "You help one user understand their spending and act on it.\n\n"
    "Rules:\n"
    "- NEVER invent or estimate a number. Every figure you state must come from a "
    "tool result. If you don't have a number, call a tool to get it.\n"
    "- Lead with the single most important thing (usually the top insight), then be "
    "concise and concrete. Amounts are in GBP (£).\n"
    "- When the user wants to act, use the action tools (set_budget, "
    "cancel_subscription). They require confirmation — propose clearly, then call "
    "the tool; the app handles the confirm.\n"
    "- You cannot see account balances; never claim a specific balance or shortfall."
)


class FinancialAgent:
    """A tool-calling Claude agent bound to one user's data."""

    def __init__(self, engine: InsightEngine, user_id: str,
                 confirm: Callable[[str, dict], bool] | None = None):
        self.engine = engine
        self.user_id = user_id
        # how to confirm a gated action; default = CLI y/n prompt
        self.confirm = confirm if confirm is not None else self._cli_confirm
        self._dashboard: dict | None = None
        self.messages: list[dict[str, Any]] = []

    # -- computed-not-hallucinated tool backends ------------------------- #
    def _dash(self) -> dict:
        if self._dashboard is None:
            self._dashboard = self.engine.dashboard(self.user_id)
        return self._dashboard

    def _run_tool(self, name: str, args: dict) -> dict:
        if name == "get_top_insights":
            return {"insights": [
                {"type": i["type"], "title": i["title"],
                 "explanation": i["explanation"], "severity": i["severity"],
                 "level": i["level"]}
                for i in self._dash()["insights"]
            ]}
        if name == "explain_insight":
            card = self._dash()["sections"].get(args["insight_type"])
            return card or {"error": "no such insight for this user"}
        if name == "category_spending":
            months = int(args.get("months", 6))
            return self.engine.spending_history(self.user_id, months=months)
        if name == "upcoming_charges":
            cf = self._dash()["sections"].get("cashflow_forecast", {})
            return {"upcoming_charges": cf.get("payload", {}).get("upcoming_charges", [])}
        # -- gated actions (simulated) ----------------------------------- #
        if name == "set_budget":
            return {"status": "applied", "simulated": True,
                    "category": args["category"], "monthly_amount": args["monthly_amount"]}
        if name == "cancel_subscription":
            return {"status": "cancelled", "simulated": True,
                    "merchant": args["merchant"]}
        return {"error": f"unknown tool {name}"}

    @staticmethod
    def _cli_confirm(tool_name: str, args: dict) -> bool:
        pretty = ", ".join(f"{k}={v}" for k, v in args.items())
        ans = input(f"\n  ⚠️  Confirm action `{tool_name}` ({pretty})? [y/N] ").strip().lower()
        return ans in ("y", "yes")

    # -- the manual tool-use loop ---------------------------------------- #
    def chat(self, user_message: str, *, max_turns: int = 8) -> str:
        """Send a user turn and run the tool loop until Claude finishes.

        Read-only tools auto-execute; gated tools call ``self.confirm`` first
        and return a declined result if the user says no.
        """
        if not available():
            raise RuntimeError(
                "Financial Agent needs ANTHROPIC_API_KEY and the `anthropic` package. "
                "Run: uv pip install anthropic && export ANTHROPIC_API_KEY=...")
        import anthropic

        client = anthropic.Anthropic()
        self.messages.append({"role": "user", "content": user_message})

        # Stable prefix (system + tools) is cached; only messages vary.
        system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
        tools = list(TOOLS)
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

        for _ in range(max_turns):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=system,
                tools=tools,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name in GATED_TOOLS and not self.confirm(block.name, block.input):
                    out: dict = {"status": "declined_by_user",
                                 "note": "User did not confirm; do not retry."}
                else:
                    out = self._run_tool(block.name, dict(block.input))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, default=str),
                })
            self.messages.append({"role": "user", "content": results})

        return "(stopped: hit max tool turns)"


def main() -> None:
    import argparse
    from .contract import load_enriched

    ap = argparse.ArgumentParser(description="Spending IQ — Financial Agent REPL")
    ap.add_argument("--parquet", default="output/df_enriched.parquet")
    ap.add_argument("--user", default="01a98e52-b964-4b0e-a25f-74ccfc49232c")
    args = ap.parse_args()

    if not available():
        print("⚠️  Financial Agent is offline (need ANTHROPIC_API_KEY + `anthropic`).")
        print("    The detectors/dashboard still run without it — this is just the chat layer.")
        return

    print(f"Loading {args.parquet} ...")
    engine = InsightEngine(load_enriched(args.parquet))
    agent = FinancialAgent(engine, args.user)
    print(f"Spending IQ agent ready for user {args.user}. Type a question (Ctrl-C to exit).\n")
    try:
        while True:
            msg = input("you › ").strip()
            if not msg:
                continue
            print("\niq  ›", agent.chat(msg), "\n")
    except (EOFError, KeyboardInterrupt):
        print("\nbye")


if __name__ == "__main__":
    main()
