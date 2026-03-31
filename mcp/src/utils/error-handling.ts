// Error handling and response formatting utilities

import { SearchV2Response, SearchResult, SearchTier } from "../api/types.js";

const EXCERPT_MAX_CHARS = 500;

/**
 * Truncate text to a maximum character length, breaking at a word boundary.
 * Appends "…" when truncated.
 */
function truncateText(text: string, maxChars: number): string {
    if (text.length <= maxChars) return text;
    const truncated = text.slice(0, maxChars);
    const lastSpace = truncated.lastIndexOf(" ");
    return (lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated) + "…";
}

export function formatSearchResponse(
    searchResponse: SearchV2Response,
    tier: SearchTier,
    collection: string,
    limit: number = 100,
) {
    const results = searchResponse.results ?? [];
    // Include full content only for small result sets (limit <= 5)
    const includeFullContent = limit <= 5;

    const formattedResults = results
        .map((result: SearchResult, index: number) => {
            const parts = [
                `**Result ${index + 1} (Score: ${result.relevance_score.toFixed(3)}):**`,
            ];

            // Name + source
            const source = result.airweave_system_metadata?.source_name;
            parts.push(source ? `${result.name} (${source})` : result.name);

            // Breadcrumbs
            if (result.breadcrumbs?.length > 0) {
                const trail = result.breadcrumbs.map(b => b.name).join(" > ");
                parts.push(`📍 ${trail}`);
            }

            // Content — excerpt by default, full text only for small result sets
            if (result.textual_representation) {
                if (includeFullContent) {
                    parts.push(result.textual_representation);
                } else {
                    parts.push(truncateText(result.textual_representation, EXCERPT_MAX_CHARS));
                }
            }

            // Link
            if (result.web_url) {
                parts.push(`🔗 ${result.web_url}`);
            }

            return parts.join("\n");
        })
        .join("\n\n---\n\n");

    const summaryText = [
        `**Collection:** ${collection} | **Tier:** ${tier}`,
        `**Results:** ${results.length}`,
        "",
        formattedResults || "No results found.",
    ].join("\n");

    return {
        content: [
            {
                type: "text" as const,
                text: summaryText,
            },
        ],
    };
}

export function formatErrorResponse(
    error: Error,
    searchRequest: any,
    collection: string,
    baseUrl: string,
) {
    return {
        content: [
            {
                type: "text" as const,
                text: `**Error:** Failed to search collection.\n\n**Details:** ${error.message}\n\n**Debugging Info:**\n- Collection: ${collection}\n- Base URL: ${baseUrl}\n- Parameters: ${JSON.stringify(searchRequest, null, 2)}`,
            },
        ],
    };
}
