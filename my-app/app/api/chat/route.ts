import { NextResponse } from 'next/server';

// Configuration
const MISTRAL_SERVER_URL = process.env.MISTRAL_SERVER_URL || 'http://127.0.0.1:8001';
const MISTRAL_CHAT_ENDPOINT = '/chat_predict';
const MISTRAL_TIMEOUT = 30000; // 30 seconds

interface MistralResponse {
  user_input: string;
  parsed_request: Record<string, any>;
  encoded_features: Record<string, any>;
  predicted_price: number | null;
  predicted_price_source: string;
  recommendation: string;
  probabilities: Record<string, number>;
  best_bucket: string;
  confidence: number;
}

// Generate simple recommendation from parsed data
function generateSimpleRecommendation(parsed: Record<string, any>): string {
  const origin = parsed.source_city || parsed.origin || 'Unknown';
  const dest = parsed.destination_city || parsed.destination || 'Unknown';
  const daysLeft = Number(parsed.days_left) || 7;

  if (daysLeft <= 1) {
    return `Book your flight from ${origin} to ${dest} now! Last-minute flights are available.`;
  } else if (daysLeft <= 7) {
    return `Good timing for your ${origin} to ${dest} flight. Check prices now - best to book within 1-7 days.`;
  } else if (daysLeft <= 21) {
    return `Planning ahead for ${origin} to ${dest}? Wait 8-14 days for potentially better prices.`;
  } else {
    return `Plenty of time for your ${origin} to ${dest} trip. Monitor prices over the next few weeks for the best deals.`;
  }
}

// Fallback handler for /parse endpoint
async function handleParseEndpoint(
  text: string,
  serverUrl: string
): Promise<NextResponse> {
  try {
    console.log('[LLM Chat] Using fallback /parse endpoint');

    const parseUrl = `${serverUrl}/parse`;
    const response = await fetch(parseUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(MISTRAL_TIMEOUT),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[LLM Chat] /parse failed: ${response.status}`, errorText);
      return NextResponse.json(
        { error: `Server error: ${response.status}` },
        { status: response.status }
      );
    }

    const parseData = await response.json();
    const parsed = parseData.parsed || {};

    // Generate a simple recommendation based on parsed data
    const recommendation = generateSimpleRecommendation(parsed);

    // Attempt a lightweight numeric fallback for predicted_price so the UI can show a value.
    // Strategy:
    // - If parsed.price exists, use it as current_price and apply a small expected drop based on days_left
    // - If parsed.price missing but days_left present, use a modest default base (8000 INR) to estimate
    // - Otherwise leave predicted_price null
    let predicted_price: number | null = null;
    let predicted_source = 'fallback_parse';
    try {
      const currentPriceRaw = parsed.price;
      const daysLeftRaw = parsed.days_left;
      const current_price = (currentPriceRaw !== null && currentPriceRaw !== undefined && !Number.isNaN(Number(currentPriceRaw))) ? Number(currentPriceRaw) : null;
      const days_left = (daysLeftRaw !== null && daysLeftRaw !== undefined && !Number.isNaN(Number(daysLeftRaw))) ? Number(daysLeftRaw) : null;

      console.log(
        '[LLM Chat] Fallback /parse: currentPrice=%s, days_left=%s',
        current_price,
        days_left
      );

      // map days_left to bucket
      let best_bucket = 'unknown';
      if (days_left !== null) {
        if (days_left <= 1) best_bucket = 'buy_now';
        else if (days_left <= 7) best_bucket = 'wait_1_7';
        else if (days_left <= 21) best_bucket = 'wait_8_21';
        else best_bucket = 'wait_22_plus';
      }

      const bucket_expected_drop: Record<string, number> = {
        buy_now: 0.0,
        wait_1_7: 0.03,
        wait_8_21: 0.06,
        wait_22_plus: 0.10,
      };

      const expected_drop = bucket_expected_drop[best_bucket] ?? 0.02;

      console.log(
        '[LLM Chat] Fallback: best_bucket=%s, expected_drop=%s',
        best_bucket,
        expected_drop
      );

      if (current_price && current_price > 0) {
        predicted_price = Math.max(0, current_price * (1.0 - expected_drop));
        predicted_source = 'heuristic_from_current_price';
      } else if (days_left !== null) {
        // Use a modest default base price to provide a visible estimate when no explicit price provided
        const fallback_base = 8000; // INR — simple heuristic placeholder
        predicted_price = Math.max(0, fallback_base * (1.0 - expected_drop));
        predicted_source = 'heuristic_from_days_left_fallback';
      } else {
        predicted_price = null;
      }

      console.log(
        '[LLM Chat] Fallback: computed predicted_price=%s, source=%s',
        predicted_price,
        predicted_source
      );
    } catch (e) {
      console.warn('[LLM Chat] Fallback price heuristic failed', e);
      predicted_price = null;
    }

    return NextResponse.json({
      recommendation,
      parsed_request: parsed,
      predicted_price: predicted_price !== null ? Number(predicted_price.toFixed(2)) : null,
      best_bucket: parsed.best_bucket || 'unknown',
      confidence: 0,
      probabilities: {},
      predicted_price_source: predicted_source,
      note: 'Using fallback LLM parsing (ML models not available)',
    });
  } catch (error: any) {
    console.error('[LLM Chat] Fallback endpoint also failed:', error.message);
    return NextResponse.json(
      {
        error: 'Cannot reach Mistral server',
        details: error.message,
        hint: 'Make sure Mistral server is running at ' + serverUrl,
      },
      { status: 503 }
    );
  }
}

export async function POST(req: Request) {
  try {
    console.log('[LLM Chat] Received POST request');
    const body = await req.json();
    const { text } = body;

    console.log('[LLM Chat] Request body:', { text: text?.substring(0, 50) });

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      console.warn('[LLM Chat] Invalid input - empty text');
      return NextResponse.json(
        { error: 'Invalid input: text field is required and must be non-empty' },
        { status: 400 }
      );
    }

    // Call Mistral server
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MISTRAL_TIMEOUT);

    try {
      const mistralUrl = `${MISTRAL_SERVER_URL}${MISTRAL_CHAT_ENDPOINT}`;

      console.log(
        `[LLM Chat] Calling ${mistralUrl} with input: "${text.substring(0, 100)}..."`
      );

      const response = await fetch(mistralUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      console.log(`[LLM Chat] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.warn(
          `[LLM Chat] /chat_predict failed: ${response.status}. Trying /parse endpoint...`,
          errorText
        );

        // Fallback to /parse endpoint
        return await handleParseEndpoint(text, MISTRAL_SERVER_URL);
      }

      const data = (await response.json()) as MistralResponse;

      // Extract recommendation from response
      const recommendation = data.recommendation || 'No recommendation available';

      // Prepare response for frontend
      return NextResponse.json({
        recommendation,
        parsed_request: data.parsed_request,
        predicted_price: data.predicted_price,
        best_bucket: data.best_bucket,
        confidence: data.confidence,
        probabilities: data.probabilities,
      });
    } catch (fetchError: any) {
      clearTimeout(timeoutId);

      if (fetchError.name === 'AbortError') {
        console.error(
          '[LLM Chat] Request timeout after',
          MISTRAL_TIMEOUT / 1000,
          'seconds'
        );
        return NextResponse.json(
          { error: 'Request timeout: Mistral server took too long to respond' },
          { status: 504 }
        );
      }

      console.error(
        '[LLM Chat] Failed to connect to Mistral server:',
        fetchError.message
      );

      // Try fallback endpoint
      console.log('[LLM Chat] Trying fallback /parse endpoint...');
      return await handleParseEndpoint(text, MISTRAL_SERVER_URL);
    }
  } catch (error: any) {
    console.error('[LLM Chat] Unexpected error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    );
  }
}

// Optional: Add GET endpoint to check health
export async function GET() {
  try {
    const mistralUrl = `${MISTRAL_SERVER_URL}/health`;
    const response = await fetch(mistralUrl, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return NextResponse.json(
        { status: 'error', message: 'Mistral server is not responding' },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json({ status: 'ok', mistral: data });
  } catch (error: any) {
    console.error('[LLM Health Check] Failed:', error.message);
    return NextResponse.json(
      {
        status: 'error',
        message: 'Cannot reach Mistral server',
        mistral_url: MISTRAL_SERVER_URL,
      },
      { status: 503 }
    );
  }
}
