# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "rich", "httpx"]
# ///
"""
LLM Model Comparison for Airweave Search Pipeline

Tests gpt-5.2 vs gemini-3.1-pro-preview on the same prompts used by:
1. Classic search (strategy planning)
2. Knowledge graph (entity extraction)
3. Query expansion

Usage:
    uv run scripts/eval_llm_models.py --base-url http://localhost:8317/v1 --api-key airweave
    uv run scripts/eval_llm_models.py --base-url https://generativelanguage.googleapis.com/v1beta/openai --api-key $GEMINI_API_KEY --model gemini-3.1-pro-preview
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


@dataclass
class Result:
    model: str
    task: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_preview: str
    success: bool
    error: str | None = None


# --- Test prompts (same as Airweave search pipeline uses) ---

SEARCH_STRATEGY_PROMPT = {
    "system": "You are a search strategy planner. Given a user query, plan the optimal search approach. Return JSON with keys: query (primary search string), variations (list of 3-5 alternative queries), retrieval_strategy (hybrid/semantic/keyword), filters (list of metadata filters or empty).",
    "user": 'Plan a search strategy for the query: "four pixel rule in web design". Return valid JSON only.',
}

KG_ENTITY_EXTRACTION_PROMPT = {
    "system": "You are a knowledge graph entity extractor. Extract entities and relationships from the given text. Return JSON with keys: entities (list of {name, type, description}), relationships (list of {source, target, description}).",
    "user": """Extract entities and relationships from this text:

Typography is way more complicated than most people think. The fundamental building block of typography is a character, and the fundamental building block of a character is called a point. The point is the default unit of measurement in typography and it translates in the physical world to 1/12th of an inch. In digital design, designers use pixels instead of points. The most important size convention is em and rem. Em sets the size of a character equal to the size of the font. On the web the default size of a font is 16 pixels. Rem stands for root em because when designing websites in CSS there's this thing called root which defines stylistic variables.

Return valid JSON only.""",
}

QUERY_EXPANSION_PROMPT = {
    "system": "You are a query expansion expert. Given a search query, generate 5 alternative phrasings that would find relevant results. Return JSON with key: expansions (list of strings).",
    "user": 'Expand this search query into 5 alternatives: "Perplexity AI as design tool". Return valid JSON only.',
}

SIMPLE_COMPLETION_PROMPT = {
    "system": "You are a helpful assistant. Answer concisely.",
    "user": "What is the four pixel rule in web design? Answer in 2-3 sentences.",
}

TASKS = {
    "search_strategy": SEARCH_STRATEGY_PROMPT,
    "kg_extraction": KG_ENTITY_EXTRACTION_PROMPT,
    "query_expansion": QUERY_EXPANSION_PROMPT,
    "simple_completion": SIMPLE_COMPLETION_PROMPT,
}


async def run_task(
    client: AsyncOpenAI, model: str, task_name: str, prompts: dict
) -> Result:
    start = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ],
            temperature=0.6,
            max_tokens=2000,
            response_format={"type": "json_object"} if task_name != "simple_completion" else None,
        )
        latency = int((time.monotonic() - start) * 1000)
        content = resp.choices[0].message.content or ""
        usage = resp.usage

        return Result(
            model=model,
            task=task_name,
            latency_ms=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            output_preview=content[:200],
            success=True,
        )
    except Exception as e:
        latency = int((time.monotonic() - start) * 1000)
        return Result(
            model=model,
            task=task_name,
            latency_ms=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            output_preview="",
            success=False,
            error=str(e)[:200],
        )


async def run_all(base_url: str, api_key: str, models: list[str], runs: int = 3):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    all_results: list[Result] = []

    for model in models:
        console.print(f"\n[bold cyan]Testing model: {model}[/]")
        for task_name, prompts in TASKS.items():
            task_results = []
            for i in range(runs):
                console.print(f"  {task_name} run {i+1}/{runs}...", end=" ")
                result = await run_task(client, model, task_name, prompts)
                task_results.append(result)
                status = f"[green]{result.latency_ms}ms[/]" if result.success else f"[red]FAIL: {result.error}[/]"
                console.print(status)

            all_results.extend(task_results)

    await client.close()
    return all_results


def print_summary(results: list[Result], models: list[str]):
    # Group by model and task
    grouped: dict[str, dict[str, list[Result]]] = {}
    for r in results:
        grouped.setdefault(r.model, {}).setdefault(r.task, []).append(r)

    # Summary table
    table = Table(title="LLM Model Comparison", show_lines=True)
    table.add_column("Task", style="bold")
    for model in models:
        table.add_column(f"{model}\nLatency (avg)", justify="right")
        table.add_column(f"{model}\nTokens (avg)", justify="right")
        table.add_column(f"{model}\nSuccess", justify="center")

    for task_name in TASKS:
        row = [task_name]
        for model in models:
            task_results = grouped.get(model, {}).get(task_name, [])
            if not task_results:
                row.extend(["—", "—", "—"])
                continue

            successes = [r for r in task_results if r.success]
            success_rate = f"{len(successes)}/{len(task_results)}"

            if successes:
                avg_latency = sum(r.latency_ms for r in successes) // len(successes)
                avg_tokens = sum(r.total_tokens for r in successes) // len(successes)
                row.extend([f"{avg_latency}ms", str(avg_tokens), success_rate])
            else:
                row.extend(["FAIL", "—", success_rate])

        table.add_row(*row)

    # Total row
    total_row = ["[bold]TOTAL (avg)[/]"]
    for model in models:
        all_model = [r for r in results if r.model == model and r.success]
        if all_model:
            avg_lat = sum(r.latency_ms for r in all_model) // len(all_model)
            avg_tok = sum(r.total_tokens for r in all_model) // len(all_model)
            total_row.extend([f"[bold]{avg_lat}ms[/]", f"[bold]{avg_tok}[/]", ""])
        else:
            total_row.extend(["—", "—", ""])
    table.add_row(*total_row)

    console.print()
    console.print(table)

    # Output quality samples
    console.print("\n[bold]Output Samples (first successful run):[/]")
    for task_name in TASKS:
        console.print(f"\n[bold cyan]{task_name}:[/]")
        for model in models:
            task_results = grouped.get(model, {}).get(task_name, [])
            success = next((r for r in task_results if r.success), None)
            if success:
                console.print(f"  [{model}]: {success.output_preview}")
            else:
                console.print(f"  [{model}]: [red]ALL FAILED[/]")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM models for Airweave search pipeline")
    parser.add_argument("--base-url", default="http://localhost:8317/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="airweave", help="API key")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.2", "gemini-3.1-pro-preview"],
        help="Models to test",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per task per model")
    args = parser.parse_args()

    console.print(f"[bold]LLM Model Comparison[/]")
    console.print(f"Base URL: {args.base_url}")
    console.print(f"Models: {', '.join(args.models)}")
    console.print(f"Runs per task: {args.runs}")
    console.print(f"Tasks: {', '.join(TASKS.keys())}")

    results = asyncio.run(run_all(args.base_url, args.api_key, args.models, args.runs))
    print_summary(results, args.models)


if __name__ == "__main__":
    main()
