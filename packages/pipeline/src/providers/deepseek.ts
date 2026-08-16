/**
 * DeepSeek storyboard provider.
 *
 * Chosen as the default because it is roughly a quarter the price of the
 * cheapest incumbent, and at ~$0.002 per video the LLM is then a rounding error
 * next to image generation.
 *
 * Two caveats it is worth being explicit about, both handled by keeping this
 * behind the LlmProvider interface rather than by avoiding DeepSeek:
 *  - the API is China-hosted, so EU/UK users' story text crosses that boundary
 *    and needs a privacy-policy disclosure;
 *  - rate limits and availability are less predictable than the big three, hence
 *    the retry and the fallback chain in the pipeline.
 */

import type { LlmProvider, ProviderResult } from "./index";
import type { GenerationRequest, Storyboard } from "../types";

/** Per-million-token list prices, USD. Update alongside docs/cost-model.md. */
export const DEEPSEEK_PRICING = { inputPerMillion: 0.27, outputPerMillion: 1.1 };

const SYSTEM_PROMPT = `You turn a story into a storyboard for a whiteboard animation video.

Return JSON only, matching:
{"title": string, "scenes": [{"imagePrompt": string, "narration": string}]}

Rules:
- Narration is what a voice reads aloud. Write it to be spoken, not read.
- Each scene's narration should take roughly 10-20 seconds to say.
- imagePrompt describes a single clear illustration with a few distinct objects.
  Whiteboard animation draws object by object, so avoid dense crowds, gradients
  and photographic backgrounds; favour clear silhouettes and separable subjects.
- Keep characters visually consistent across scenes: repeat the same descriptive
  phrases for a character in every prompt they appear in.
- No text or lettering in the illustrations.`;

interface ChatResponse {
  choices: { message: { content: string } }[];
  usage?: { prompt_tokens: number; completion_tokens: number };
}

export interface DeepSeekOptions {
  apiKey: string;
  model?: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  maxRetries?: number;
}

export class DeepSeekProvider implements LlmProvider {
  readonly name = "deepseek";

  private readonly model: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly maxRetries: number;

  constructor(private readonly options: DeepSeekOptions) {
    this.model = options.model ?? "deepseek-chat";
    this.baseUrl = options.baseUrl ?? "https://api.deepseek.com";
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.maxRetries = options.maxRetries ?? 3;
  }

  async storyboard(request: GenerationRequest): Promise<ProviderResult<Storyboard>> {
    const userPrompt = [
      `Language: ${request.language}`,
      `Target scenes: ${request.sceneCount}`,
      "",
      request.input,
    ].join("\n");

    const response = await this.callWithRetry({
      model: this.model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.7,
    });

    const parsed = JSON.parse(response.choices[0].message.content) as {
      title?: string;
      scenes?: { imagePrompt?: string; narration?: string }[];
    };

    const scenes = (parsed.scenes ?? [])
      .filter((scene) => scene.narration && scene.imagePrompt)
      .map((scene, index) => ({
        index,
        imagePrompt: scene.imagePrompt!,
        narration: scene.narration!,
      }));

    if (scenes.length === 0) {
      throw new Error("DeepSeek returned no usable scenes");
    }

    const usage = response.usage;
    const cost = usage
      ? (usage.prompt_tokens / 1e6) * DEEPSEEK_PRICING.inputPerMillion +
        (usage.completion_tokens / 1e6) * DEEPSEEK_PRICING.outputPerMillion
      : 0;

    return {
      value: { title: parsed.title ?? "Untitled", scenes },
      usage: [
        {
          provider: this.name,
          operation: "storyboard",
          cost,
          quantity: (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0),
          unit: "tokens",
        },
      ],
    };
  }

  private async callWithRetry(body: unknown): Promise<ChatResponse> {
    let lastError: unknown;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await this.fetchImpl(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.options.apiKey}`,
          },
          body: JSON.stringify(body),
        });

        if (response.status === 429 || response.status >= 500) {
          throw new Error(`retryable status ${response.status}`);
        }
        if (!response.ok) {
          throw new Error(`DeepSeek error ${response.status}: ${await response.text()}`);
        }
        return (await response.json()) as ChatResponse;
      } catch (error) {
        lastError = error;
        // Exponential backoff: DeepSeek's rate limits are the expected failure.
        if (attempt < this.maxRetries - 1) {
          await new Promise((r) => setTimeout(r, 2 ** attempt * 500));
        }
      }
    }

    throw lastError;
  }
}
