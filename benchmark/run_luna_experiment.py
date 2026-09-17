"""Run an explicitly non-official benchmark experiment with gpt-5.6-luna.

The official ``benchmark.runner`` remains locked to gpt-4o-mini.  This wrapper
uses the same scheduler, execution kernel, and oracle, but patches the model
client only in this process because gpt-5.6-luna does not accept ``temperature``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gas_server.core.llm_client as llm_client


MODEL = "gpt-5.6-luna"


def configure_luna_client() -> None:
    """Patch the process-local client without changing official benchmark rules."""
    llm_client.DEFAULT_MODEL = MODEL

    def init(self: Any, api_key: str, model: str = MODEL) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_client()
        self.chat = SimpleNamespace(completions=self)

    def create(
        self: Any,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> Any:
        del model, temperature
        call_messages = list(messages or [])
        if not call_messages and "prompt" in kwargs:
            call_messages = [{"role": "user", "content": kwargs["prompt"]}]
        estimated_tokens = max(
            100, sum(len(message.get("content", "")) for message in call_messages) // 4
        )
        last_error: Exception | None = None
        for attempt in range(max_retries):
            llm_client._GLOBAL_RATE_LIMITER.acquire(estimated_tokens=estimated_tokens)
            try:
                if self._client is None:
                    self._init_client()
                if self._client is None:
                    raise llm_client.InfrastructureError("OpenAI client is unavailable.")
                response = self._client.chat.completions.create(
                    model=MODEL,
                    messages=call_messages,
                )
                total_tokens = getattr(
                    getattr(response, "usage", None), "total_tokens", estimated_tokens
                )
                llm_client._GLOBAL_RATE_LIMITER.record_usage(total_tokens)
                return response
            except Exception as exc:
                last_error = exc
                message = str(exc)
                transient = (
                    "429" in message
                    or "rate_limit" in message.lower()
                    or "quota" in message.lower()
                    or re.search(r"\b5\d{2}\b", message) is not None
                    or "timeout" in message.lower()
                    or "connection" in message.lower()
                )
                if not transient:
                    raise
                if attempt == max_retries - 1:
                    raise llm_client.InfrastructureError(
                        f"Provider infrastructure failure after {max_retries} attempts: {message}"
                    ) from exc
                delay = (2**attempt) * 3.0 + random.uniform(0.5, 2.0)
                llm_client._GLOBAL_RATE_LIMITER.record_retry_wait(delay)
                time.sleep(delay)
        raise llm_client.InfrastructureError("OpenAI request failed.") from last_error

    llm_client.OpenAIClientWrapper.__init__ = init
    llm_client.OpenAIClientWrapper.create = create


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental 600-trajectory gpt-5.6-luna runner")
    parser.add_argument("--suite", choices=["isolated", "compound", "dev", "all"], default="all")
    parser.add_argument("--conditions", default="C0,C1,C2,C3,C4,C5")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_luna_client()
    import benchmark.runner as runner

    runner.build_llm_client = llm_client.build_llm_client
    suites = ["isolated", "compound"] if args.suite == "all" else [args.suite]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    schedule = runner.build_schedule(suites, conditions, num_runs=args.runs, seed=args.seed)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print(json.dumps({"model": MODEL, "total_trajectories": len(schedule), "output_dir": str(output_dir)}, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle = runner.BenchmarkOracle()
    results = []
    for index, item in enumerate(schedule, 1):
        output_file = output_dir / f"{item['trajectory_id']}.json"
        if output_file.exists():
            result = json.loads(output_file.read_text())
            if result.get("status") != "infrastructure_invalid":
                results.append(result)
                print(f"[{index}/{len(schedule)}] SKIP {item['trajectory_id']}", flush=True)
                continue
        print(f"[{index}/{len(schedule)}] RUN {item['trajectory_id']}", flush=True)
        results.append(runner.execute_trajectory(item, oracle, output_dir, model=MODEL))

    valid = [item for item in results if item.get("status") != "infrastructure_invalid"]
    report = {
        "experiment_type": "non_official_luna",
        "model": MODEL,
        "total": len(results),
        "valid": len(valid),
        "passed": sum(bool(item.get("bafo_success")) for item in valid),
        "failed": sum(not bool(item.get("bafo_success")) for item in valid),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
