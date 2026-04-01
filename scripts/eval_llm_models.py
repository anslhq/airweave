# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "rich"]
# ///
"""
LLM Model Comparison for Airweave Search Pipeline

Tests models across the exact prompts used by:
1. Classic search (strategy planning — structured JSON)
2. Knowledge graph (entity extraction — structured JSON)
3. Query expansion (structured JSON)
4. Simple completion (factual accuracy)

Usage:
    # Test free OpenRouter models
    uv run scripts/eval_llm_models.py --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY --preset free

    # Test specific models via local proxy
    uv run scripts/eval_llm_models.py --base-url http://localhost:8317/v1 --api-key airweave --models gpt-5.2 gemini-2.5-flash

    # Test all presets
    uv run scripts/eval_llm_models.py --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY --preset all
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

console = Console()

# ── Model presets ──

PRESETS = {
    "free": [
        "openai/gpt-oss-20b",
        "liquid/lfm2.5-1.2b-instruct",
        "nousresearch/hermes-3-llama-3.1-405b",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-3n-e4b-it:free",
        "google/gemma-3-12b-it:free",
    ],
    "baseline": [
        "gpt-5.2",
        "gemini-2.5-flash",
        "gemini-3.1-pro-preview",
    ],
}
PRESETS["all"] = PRESETS["free"] + PRESETS["baseline"]


@dataclass
class Result:
    model: str
    task: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output: str
    success: bool
    error: str | None = None
    json_valid: bool = False
    json_complete: bool = False
    factual_score: int = 0  # 0-3 for simple_completion


# ── Test prompts ──

SEARCH_STRATEGY = {
    "system": "You are a search strategy planner. Given a user query, plan the optimal search approach. Return JSON with keys: query (primary search string), variations (list of 3-5 alternative queries), retrieval_strategy (hybrid/semantic/keyword), filters (list of metadata filters or empty list).",
    "user": 'Plan a search strategy for the query: "four pixel rule in web design". Return valid JSON only, no markdown fences.',
    "required_keys": ["query", "variations", "retrieval_strategy"],
}

KG_EXTRACTION = {
    "system": "You are a knowledge graph entity extractor. Extract entities and relationships from the given text. Return JSON with keys: entities (list of {name, type, description}), relationships (list of {source, target, description}).",
    "user": """Extract entities and relationships from this text:

Typography is way more complicated than most people think. The fundamental building block of typography is a character, and the fundamental building block of a character is called a point. The point is the default unit of measurement in typography and it translates in the physical world to 1/12th of an inch. In digital design, designers use pixels instead of points. The most important size convention is em and rem. Em sets the size of a character equal to the size of the font. On the web the default size of a font is 16 pixels. Rem stands for root em.

Return valid JSON only, no markdown fences.""",
    "required_keys": ["entities", "relationships"],
}

QUERY_EXPANSION = {
    "system": "You are a query expansion expert. Given a search query, generate 5 alternative phrasings that would find relevant results. Return JSON with key: expansions (list of 5 strings).",
    "user": 'Expand this search query into 5 alternatives: "Perplexity AI as design tool". Return valid JSON only, no markdown fences.',
    "required_keys": ["expansions"],
}

SIMPLE_COMPLETION = {
    "system": "You are a helpful assistant. Answer concisely and accurately.",
    "user": "What is the four pixel rule in web design? Answer in 2-3 sentences.",
    "required_keys": [],
}

TASKS = {
    "search_strategy": SEARCH_STRATEGY,
    "kg_extraction": KG_EXTRACTION,
    "query_expansion": QUERY_EXPANSION,
    "simple_completion": SIMPLE_COMPLETION,
}


def strip_fences(text: str) -> str:
    """Strip markdown code fences from JSON responses."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.index("\n") if "\n" in t else len(t)
        t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    return t


def validate_json(text: str, required_keys: list[str]) -> tuple[bool, bool, dict | None]:
    """Returns (is_valid_json, has_all_keys, parsed_dict)."""
    try:
        cleaned = strip_fences(text)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return True, False, parsed
        has_keys = all(k in parsed for k in required_keys)
        return True, has_keys, parsed
    except (json.JSONDecodeError, ValueError):
        return False, False, None


def score_factual(text: str) -> int:
    """Score factual accuracy of four pixel rule answer (0-3)."""
    t = text.lower()
    score = 0
    if "multiple" in t or "divisible" in t or "increment" in t:
        score += 1
    if "4" in t or "four" in t:
        score += 1
    if any(w in t for w in ["spacing", "margin", "padding", "layout", "dimension", "sizing"]):
        score += 1
    return score


async def run_task(client: AsyncOpenAI, model: str, task_name: str, task: dict) -> Result:
    start = time.monotonic()
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": task["system"]},
                {"role": "user", "content": task["user"]},
            ],
            "temperature": 0.6,
            "max_tokens": 2000,
        }
        if task_name != "simple_completion":
            kwargs["response_format"] = {"type": "json_object"}

        resp = await client.chat.completions.create(**kwargs)
        latency = int((time.monotonic() - start) * 1000)
        content = resp.choices[0].message.content or ""
        usage = resp.usage

        is_json, has_keys, _ = validate_json(content, task.get("required_keys", []))
        fact_score = score_factual(content) if task_name == "simple_completion" else 0

        return Result(
            model=model, task=task_name, latency_ms=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            output=content, success=True,
            json_valid=is_json if task_name != "simple_completion" else True,
            json_complete=has_keys if task_name != "simple_completion" else True,
            factual_score=fact_score,
        )
    except Exception as e:
        latency = int((time.monotonic() - start) * 1000)
        return Result(
            model=model, task=task_name, latency_ms=latency,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            output="", success=False, error=str(e)[:150],
        )


async def run_all(base_url: str, api_key: str, models: list[str], runs: int = 2):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    all_results: list[Result] = []

    for model in models:
        console.print(f"\n[bold cyan]Testing: {model}[/]")
        for task_name, task in TASKS.items():
            task_results = []
            for i in range(runs):
                console.print(f"  {task_name} run {i+1}/{runs}...", end=" ")
                result = await run_task(client, model, task_name, task)
                if result.success:
                    status = f"[green]{result.latency_ms}ms[/]"
                    if not result.json_valid:
                        status += " [red]BAD JSON[/]"
                    elif not result.json_complete:
                        status += " [yellow]MISSING KEYS[/]"
                else:
                    status = f"[red]FAIL: {result.error}[/]"
                console.print(status)
                task_results.append(result)
            all_results.extend(task_results)

    await client.close()
    return all_results


def print_results(results: list[Result], models: list[str]):
    # ── Summary table ──
    table = Table(title="LLM Model Comparison — Airweave Search Pipeline", show_lines=True)
    table.add_column("Model", style="bold", max_width=35)
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("JSON Valid", justify="center")
    table.add_column("JSON Complete", justify="center")
    table.add_column("Factual (0-3)", justify="center")
    table.add_column("Success Rate", justify="center")
    table.add_column("Quality Score", justify="center", style="bold")

    model_scores = {}

    for model in models:
        model_results = [r for r in results if r.model == model]
        successes = [r for r in model_results if r.success]
        json_tasks = [r for r in successes if r.task != "simple_completion"]
        fact_tasks = [r for r in successes if r.task == "simple_completion"]

        if not successes:
            table.add_row(model, "FAIL", "—", "—", "—", "—", "0/" + str(len(model_results)), "[red]0[/]")
            model_scores[model] = 0
            continue

        avg_lat = sum(r.latency_ms for r in successes) // len(successes)
        avg_tok = sum(r.total_tokens for r in successes) // len(successes)
        json_valid_pct = sum(1 for r in json_tasks if r.json_valid) / max(len(json_tasks), 1)
        json_complete_pct = sum(1 for r in json_tasks if r.json_complete) / max(len(json_tasks), 1)
        fact_avg = sum(r.factual_score for r in fact_tasks) / max(len(fact_tasks), 1)
        success_rate = f"{len(successes)}/{len(model_results)}"

        # Quality score: weighted combination
        quality = (
            (json_valid_pct * 30) +       # 30 pts for valid JSON
            (json_complete_pct * 30) +     # 30 pts for complete schema
            (fact_avg / 3 * 20) +          # 20 pts for factual accuracy
            (len(successes) / len(model_results) * 20)  # 20 pts for reliability
        )
        model_scores[model] = quality

        jv = f"[green]{json_valid_pct:.0%}[/]" if json_valid_pct >= 0.8 else f"[red]{json_valid_pct:.0%}[/]"
        jc = f"[green]{json_complete_pct:.0%}[/]" if json_complete_pct >= 0.8 else f"[yellow]{json_complete_pct:.0%}[/]"
        fs = f"[green]{fact_avg:.1f}[/]" if fact_avg >= 2 else f"[yellow]{fact_avg:.1f}[/]"
        qs = f"[green]{quality:.0f}/100[/]" if quality >= 80 else f"[yellow]{quality:.0f}/100[/]" if quality >= 50 else f"[red]{quality:.0f}/100[/]"

        table.add_row(model, f"{avg_lat}ms", str(avg_tok), jv, jc, fs, success_rate, qs)

    console.print()
    console.print(table)

    # ── Per-task breakdown ──
    console.print("\n[bold]Per-Task Breakdown:[/]")
    for task_name in TASKS:
        console.print(f"\n  [bold cyan]{task_name}:[/]")
        for model in models:
            task_results = [r for r in results if r.model == model and r.task == task_name and r.success]
            if not task_results:
                console.print(f"    {model.split('/')[-1][:25]}: [red]ALL FAILED[/]")
                continue
            r = task_results[0]
            output_preview = strip_fences(r.output)[:120].replace("\n", " ")
            flags = ""
            if task_name != "simple_completion":
                flags = f" json={'OK' if r.json_valid else 'BAD'} schema={'OK' if r.json_complete else 'MISSING'}"
            else:
                flags = f" factual={r.factual_score}/3"
            console.print(f"    {model.split('/')[-1][:25]}: {r.latency_ms}ms{flags}")
            console.print(f"      {output_preview}")

    # ── Rankings ──
    console.print("\n[bold]Final Rankings:[/]")
    ranked = sorted(model_scores.items(), key=lambda x: -x[1])
    for i, (model, score) in enumerate(ranked, 1):
        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}."
        console.print(f"  {icon} {model}: {score:.0f}/100")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM models for Airweave search pipeline")
    parser.add_argument("--base-url", default="http://localhost:8317/v1")
    parser.add_argument("--api-key", default="airweave")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--preset", choices=["free", "baseline", "all"], default=None)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.preset:
        models = PRESETS[args.preset]
    else:
        models = PRESETS["baseline"]

    console.print(f"[bold]LLM Model Evaluation for Airweave[/]")
    console.print(f"Base URL: {args.base_url}")
    console.print(f"Models: {len(models)} ({', '.join(m.split('/')[-1][:20] for m in models)})")
    console.print(f"Runs: {args.runs} per task, {len(TASKS)} tasks = {args.runs * len(TASKS) * len(models)} total calls")

    results = asyncio.run(run_all(args.base_url, args.api_key, models, args.runs))
    print_results(results, models)


if __name__ == "__main__":
    main()
