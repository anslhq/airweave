// Search tool implementation — V2 tiered search

import { z } from "zod";
import { AirweaveClient } from "../api/airweave-client.js";
import { SearchTier, FilterGroup } from "../api/types.js";
import { formatSearchResponse, formatErrorResponse } from "../utils/error-handling.js";
import { searchDuration, searchTotal } from "../metrics/prometheus.js";

// ── Filter sub-schemas ──────────────────────────────────────────────────────

const filterConditionSchema = z.object({
    field: z.string().describe(
        "Field to filter on. Options: entity_id, name, created_at, updated_at, " +
        "breadcrumbs.entity_id, breadcrumbs.name, breadcrumbs.entity_type, " +
        "airweave_system_metadata.source_name, airweave_system_metadata.entity_type, " +
        "airweave_system_metadata.original_entity_id, airweave_system_metadata.chunk_index, " +
        "airweave_system_metadata.sync_id, airweave_system_metadata.sync_job_id"
    ),
    operator: z.enum([
        "equals", "not_equals", "contains",
        "greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal",
        "in", "not_in"
    ]).describe("Comparison operator"),
    value: z.union([
        z.string(), z.number(), z.boolean(),
        z.array(z.string()), z.array(z.number())
    ]).describe("Value to compare against. Use a list for 'in'/'not_in'"),
});

const filterGroupSchema = z.object({
    conditions: z.array(filterConditionSchema).min(1)
        .describe("Conditions within this group (combined with AND)"),
});

// ── Main tool ───────────────────────────────────────────────────────────────

export function createSearchTool(
    toolName: string,
    collection: string,
    airweaveClient: AirweaveClient
) {
    const searchSchema = {
        query: z.string().min(1).max(1000)
            .describe("The search query text to find relevant documents and data"),
        tier: z.enum(["instant", "classic", "agentic"]).optional().default("classic")
            .describe(
                "Search tier: " +
                "'instant' (fastest, direct vector search), " +
                "'classic' (default, AI-optimized with LLM-planned strategy), " +
                "'agentic' (deepest, multi-step agent with tool calling)"
            ),
        retrieval_strategy: z.enum(["hybrid", "semantic", "keyword"]).optional()
            .describe("Only for instant tier. 'hybrid' (default, semantic + keyword), 'semantic' (dense/neural only), 'keyword' (BM25 only)"),
        limit: z.number().min(1).max(1000).optional().default(100)
            .describe("Maximum number of results to return"),
        offset: z.number().min(0).optional().default(0)
            .describe("Number of results to skip (instant and classic only)"),
        thinking: z.boolean().optional()
            .describe("Only for agentic tier. Enable extended thinking / chain-of-thought"),
        filter: z.array(filterGroupSchema).optional()
            .describe("Filter groups (combined with OR). Each group has conditions combined with AND"),
    };

    const fullSchema = z.object(searchSchema);

    const description = [
        `Search the Airweave collection '${collection}' with three tiers of depth.\n`,
        "**Tiers:**",
        "- `instant`: Direct vector search — fastest, sub-second latency",
        "- `classic` (default): AI-optimized — LLM plans the strategy, ~2-5s",
        "- `agentic`: Multi-step agent — iterative search with tool calling, deepest results\n",
        "**Parameters:**",
        "- `query` (required): Search text",
        "- `tier`: Search tier (default: classic)",
        "- `retrieval_strategy`: hybrid|semantic|keyword (instant tier only)",
        "- `limit`: Max results (default: 100)",
        "- `offset`: Skip results for pagination (instant/classic only)",
        "- `thinking`: Enable chain-of-thought (agentic tier only)",
        "- `filter`: Structured filters for precise matching\n",
        "**Filter example:**",
        '```json',
        '{"filter": [{"conditions": [',
        '  {"field": "airweave_system_metadata.source_name", "operator": "equals", "value": "notion"}',
        ']}]}',
        '```',
    ].join("\n");

    return {
        name: toolName,
        description,
        schema: searchSchema,
        handler: async (params: Record<string, unknown>) => {
            const end = searchDuration.startTimer();
            try {
                const validated = fullSchema.parse(params);
                const tier: SearchTier = validated.tier;
                const filter = validated.filter as FilterGroup[] | undefined;

                let response;

                switch (tier) {
                    case "instant":
                        response = await airweaveClient.searchInstant({
                            query: validated.query,
                            retrieval_strategy: validated.retrieval_strategy,
                            filter,
                            limit: validated.limit,
                            offset: validated.offset,
                        });
                        break;

                    case "agentic":
                        response = await airweaveClient.searchAgentic({
                            query: validated.query,
                            thinking: validated.thinking,
                            filter,
                            limit: validated.limit,
                        });
                        break;

                    case "classic":
                    default:
                        response = await airweaveClient.searchClassic({
                            query: validated.query,
                            filter,
                            limit: validated.limit,
                            offset: validated.offset,
                        });
                        break;
                }

                end({ status: 'success' });
                searchTotal.inc({ status: 'success' });
                return formatSearchResponse(response, tier, collection, validated.limit);
            } catch (error) {
                end({ status: 'error' });
                searchTotal.inc({ status: 'error' });

                if (error instanceof z.ZodError) {
                    const errorMessages = error.errors.map(e => `${e.path.join('.')}: ${e.message}`);
                    return {
                        content: [
                            {
                                type: "text" as const,
                                text: `**Parameter Validation Errors:**\n${errorMessages.join("\n")}`,
                            },
                        ],
                    };
                }

                console.error("Error in search tool:", error);
                return formatErrorResponse(
                    error as Error,
                    params,
                    collection,
                    airweaveClient['config'].baseUrl
                );
            }
        }
    };
}
