import { NextResponse } from "next/server";
import axios from "axios";
import { spawn } from "child_process";
import path from "path";

const AMADEUS_BASE_URL = "https://test.api.amadeus.com";

let amadeusToken: { value: string | null; expiresAt: number } = {
  value: null,
  expiresAt: 0,
};

async function getAmadeusToken() {
  // Return cached token if still valid
  if (amadeusToken.value && Date.now() < amadeusToken.expiresAt) {
    console.log("Using cached Amadeus token");
    return amadeusToken.value;
  }

  const clientId = process.env.AMADEUS_CLIENT_ID ?? "";
  const clientSecret = process.env.AMADEUS_CLIENT_SECRET ?? "";

  if (!clientId || !clientSecret) {
    throw new Error("Missing Amadeus credentials (AMADEUS_CLIENT_ID / AMADEUS_CLIENT_SECRET)");
  }

  try {
    const authUrl = `${AMADEUS_BASE_URL}/v1/security/oauth2/token`;
    const params = new URLSearchParams();
    params.append("grant_type", "client_credentials");
    params.append("client_id", clientId);
    params.append("client_secret", clientSecret);

    const resp = await axios.post(authUrl, params.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      timeout: 10_000,
    });

    const token = resp.data?.access_token;
    const expiresIn = Number(resp.data?.expires_in ?? 1800);

    if (!token) throw new Error("No access_token in Amadeus response");

    amadeusToken = {
      value: token,
      expiresAt: Date.now() + (expiresIn - 5) * 1000,
    };

    console.log("Fetched new Amadeus token");
    return token;
  } catch (err: any) {
    console.error("Failed to fetch Amadeus token:", err.response?.data ?? err.message ?? err);
    throw new Error("Authentication with Amadeus failed");
  }
}

function extractIata(value?: string) {
  if (!value) return "";
  // If value is "City (IATA)" extract IATA, else if it's already 3-letter return as-is
  const m = value.match(/\(([A-Z]{3})\)$/);
  if (m) return m[1];
  if (/^[A-Z]{3}$/.test(value.trim())) return value.trim();
  // fallback: try last token after space (e.g., "Delhi DEL")
  const parts = value.trim().split(/\s+/);
  const last = parts[parts.length - 1];
  return /^[A-Z]{3}$/.test(last) ? last : value;
}

function parseFlightData(amadeusData: any) {
  const flightOffers = amadeusData?.data ?? [];
  const parsed: any[] = [];

  for (const offer of flightOffers) {
    try {
      const itinerary = offer.itineraries?.[0];
      const segments = itinerary?.segments ?? [];
      const fareDetail = offer.travelerPricings?.[0]?.fareDetailsBySegment?.[0];

      // Use last segment as the final arrival
      const lastSegment = segments[segments.length - 1];

      // attempt to extract a flight number from first segment
      const firstSeg = segments[0] ?? {};
      const flightNumber =
        firstSeg.number ??
        firstSeg.flightNumber ??
        (firstSeg.operating && firstSeg.operating.number) ??
        null;

      // Build a readable stops/route summary
      const route = segments.map(s => `${s.departure?.iataCode}→${s.arrival?.iataCode}`).join(", ");
      const stops = Math.max(0, segments.length - 1);

      parsed.push({
        id: offer?.id ?? null,
        totalPrice: offer?.price?.total ?? null,
        currency: offer?.price?.currency ?? null,
        lastTicketingDate: offer?.lastTicketingDate ?? null,
        airline: segments[0]?.carrierCode ?? null,
        aircraft: segments[0]?.aircraft?.code ?? null,
        departureAirport: segments[0]?.departure?.iataCode ?? null,
        departureTime: segments[0]?.departure?.at ?? null,
        arrivalAirport: lastSegment?.arrival?.iataCode ?? null, // final arrival
        arrivalTime: lastSegment?.arrival?.at ?? null,
        cabin: fareDetail?.cabin ?? null,
        stops,
        route, // e.g. "DEL→MAA, MAA→JAI"
        rawSegments: segments, // full segment data if you want to render details
        flightNumber,
      });
    } catch (err) {
      console.warn("Skipped malformed offer:", err);
    }
  }

  return parsed;
}

// add helper to call the Python model (reads/writes JSON via stdin/stdout)
async function callPythonModel(payload: Record<string, any>): Promise<{ predictedPrice?: number; debug?: any }> {
  return new Promise((resolve, reject) => {
    const modelPath = path.resolve(process.cwd(), "../model/predict.py"); // my-app -> ../model
    const py = spawn("python3", [modelPath], {
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let out = "";
    let err = "";

    py.stdout.on("data", (chunk) => (out += chunk.toString()));
    py.stderr.on("data", (chunk) => (err += chunk.toString()));

    py.on("error", (e) => reject(e));

    py.on("close", (code) => {
      // always include stdout/stderr in error message so debugging is easier
      if (code !== 0) {
        return reject(new Error(`Python exited ${code}. stdout: ${out.trim() || "<empty>"} stderr: ${err.trim() || "<empty>"}`));
      }
      try {
        const json = JSON.parse(out);
        resolve(json);
      } catch (e) {
        reject(new Error(`Invalid JSON from python model: ${e.message} / output: ${out}`));
      }
    });

    // send payload
    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();

    // safety timeout
    const t = setTimeout(() => {
      try { py.kill(); } catch {}
      reject(new Error("Python model timed out"));
    }, 20000);

    py.on("close", () => clearTimeout(t));
  });
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    let { origin, destination, date, airlines = "", currency = "INR" } = body ?? {};

    if (!origin || !destination || !date) {
      return NextResponse.json({ error: "Missing required fields: origin, destination, date" }, { status: 400 });
    }

    // Ensure we send IATA codes to Amadeus
    origin = extractIata(origin);
    destination = extractIata(destination);

    if (!/^[A-Z]{3}$/.test(origin) || !/^[A-Z]{3}$/.test(destination)) {
      return NextResponse.json({ error: "origin and destination must resolve to 3-letter IATA codes" }, { status: 400 });
    }

    const token = await getAmadeusToken();

    const searchUrl = `${AMADEUS_BASE_URL}/v2/shopping/flight-offers`;
    const params = {
      originLocationCode: origin,
      destinationLocationCode: destination,
      departureDate: date,
      includedAirlineCodes: airlines || undefined,
      currencyCode: currency,
      adults: "1", // backend default
      max: "10",
    };

    const resp = await axios.get(searchUrl, {
      headers: { Authorization: `Bearer ${token}` },
      params,
      timeout: 15_000,
    });

    const parsed = parseFlightData(resp.data);

    // Compute baseline (average numeric price) for model features
    const numericPrices = parsed
      .map((p: any) => Number(p.totalPrice))
      .filter((n: number) => !isNaN(n));
    const avgPrice = numericPrices.length ? numericPrices.reduce((a: number, b: number) => a + b, 0) / numericPrices.length : null;

    // Prepare model input
    const modelInput = {
      origin,
      destination,
      date,
      cabin: (fareDetailFromParsed(parsed) ?? "ECONOMY"), // helper below
      current_avg_price: avgPrice,
      offers_count: parsed.length,
      currency,
      // add any other numeric/time features needed by your model
    };

    // Call the python model (if python exists). If the model errors, continue without prediction.
    let modelResult: any = {};
    try {
      modelResult = await callPythonModel(modelInput);
    } catch (merr) {
      console.error("Model call failed:", merr instanceof Error ? merr.message : merr);
      modelResult = { predictedPrice: null, debug: { error: String(merr) } };
    }

    // attach prediction to response
    return NextResponse.json({ data: parsed, model: modelResult });

  } catch (err: any) {
    console.error("API /api/search error:", err.response?.data ?? err.message ?? err);
    const message = err.response?.data ?? err.message ?? "Internal Server Error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// small helper: try to get cabin from parsed results
function fareDetailFromParsed(parsedArray: any[]) {
  if (!parsedArray || !Array.isArray(parsedArray) || parsedArray.length === 0) return null;
  const first = parsedArray[0];
  return (first?.cabin ?? null);
}