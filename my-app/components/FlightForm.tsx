"use client"

import React, { useState } from "react";

const airports = [
  'Delhi (DEL)',
  'Mumbai (BOM)',
  'Bengaluru (BLR)',
  'Chennai (MAA)',
  'Kolkata (CCU)',
  'Hyderabad (HYD)',
  'Ahmedabad (AMD)',
  'Goa (GOI)',
  'Pune (PNQ)',
  'Jaipur (JAI)',
  'Cochin (COK)',
  'Lucknow (LKO)',
  'Indore (IDR)',
  'Patna (PAT)',
  'Srinagar (SXR)',
];

const classes = [
  'Economy',
  'Premium Economy',
  'Business',
  'First',
];

const FlightForm: React.FC = () => {
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [date, setDate] = useState('');
  const [flightClass, setFlightClass] = useState(classes[0]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [modelPred, setModelPred] = useState<number | null>(null);

  const selectBase =
    "appearance-none w-full rounded-xl border border-transparent bg-white/60 backdrop-blur-sm px-4 py-2 pr-12 text-sm font-medium text-gray-800 " +
    "shadow-sm hover:shadow-md transition-shadow duration-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2";

  const extractIata = (v: string) => {
    const m = v.match(/\(([A-Z]{3})\)$/);
    if (m) return m[1];
    return v;
  };

  const formatTime = (iso?: string) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return iso;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResults([]);
    setModelPred(null);

    const payload = {
      origin: extractIata(origin),
      destination: extractIata(destination),
      date,
      airlines: "",
      currency: "INR",
    };

    setLoading(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (!res.ok) {
        setError(json?.error || "Search failed");
        setResults([]);
      } else {
        setResults(json.data || []);
        setModelPred(json.model?.predictedPrice ?? null);
      }
    } catch (err: any) {
      setError(err.message || "Network error");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const priceNumber = (p: any) => {
    const n = Number(String(p).replace(/[^0-9.]/g, ""));
    return isNaN(n) ? null : n;
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-7xl mx-auto mt-10 p-6 rounded-2xl shadow-2xl bg-gradient-to-b from-white/70 to-white/50 text-black">
      <h2 className="text-2xl sm:text-3xl font-extrabold mb-2 text-center tracking-tight text-blue-700">Predict Your Flight Price</h2>
      <p className="text-sm text-center text-gray-600 mb-6">Get AI-based price estimates for domestic flights in India.</p>

      <div className="flex flex-col sm:flex-row sm:items-end gap-4">
        <div className="flex-1">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Origin</label>
          <div className="relative group">
            <select value={origin} onChange={(e) => setOrigin(e.target.value)} className={selectBase} required>
              <option value="">Select origin</option>
              {airports.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        </div>

        <div className="flex-1">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Destination</label>
          <div className="relative group">
            <select value={destination} onChange={(e) => setDestination(e.target.value)} className={selectBase} required>
              <option value="">Select destination</option>
              {airports.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        </div>

        <div className="w-44">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Date</label>
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className="w-full rounded-xl border border-transparent bg-white/60 backdrop-blur-sm px-4 py-2 text-sm font-medium text-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400" required />
        </div>

        <div className="w-48">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Cabin Class</label>
          <div className="relative group">
            <select value={flightClass} onChange={(e) => setFlightClass(e.target.value)} className={selectBase} required>
              {classes.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>

        <div className="sm:w-44">
          <button type="submit" disabled={loading} className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-xl shadow-md transform-gpu hover:-translate-y-0.5 transition flex items-center justify-center">
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
                </svg>
                Searching...
              </>
            ) : (
              "Predict Price"
            )}
          </button>
        </div>
      </div>

      {/* show model predicted price */}
      {modelPred !== null && (
        <div className="mt-4 text-center text-sm text-gray-700">
          Model predicted price: <strong className="text-blue-700">₹{Number(modelPred).toFixed(2)}</strong>
        </div>
      )}

      {error && <div className="mt-4 text-red-600 font-medium">{error}</div>}

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-6 space-y-3">
          {results.map((r, i) => {
            const offerPrice = priceNumber(r.totalPrice);
            const predicted = modelPred;
            const diff = (predicted !== null && offerPrice !== null) ? (predicted - offerPrice) : null;
            const isGoodDeal = diff !== null ? (offerPrice <= predicted) : null;

            return (
              <div key={i} className="p-4 bg-white/90 rounded-lg shadow-sm border flex flex-col sm:flex-row sm:items-center justify-between">
                <div>
                  <div className="text-lg font-semibold">{r.airline} {r.departureAirport} → {r.arrivalAirport}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    <span className="mr-4">{formatTime(r.departureTime)} → {formatTime(r.arrivalTime)}</span>
                    {r.flightNumber && <span className="mr-4">Flight: <strong>{r.airline}{r.flightNumber}</strong></span>}
                  </div>
                  <div className="text-sm text-gray-600 mt-2">Route: {r.route}</div>
                </div>

                <div className="mt-3 sm:mt-0 text-right">
                  <div className="text-lg font-semibold">₹{offerPrice !== null ? Number(offerPrice).toFixed(2) : r.totalPrice} {r.currency}</div>
                  <div className="text-sm text-gray-600 mt-1">{String(r.cabin || "").toUpperCase()}</div>

                  {/* deal badge */}
                  {predicted !== null && offerPrice !== null && (
                    <div className="mt-2">
                      {isGoodDeal ? (
                        <span className="inline-flex items-center px-2.5 py-1 rounded-full bg-green-100 text-green-800 text-sm font-medium">Below predicted by ₹{Math.abs(diff).toFixed(2)}</span>
                      ) : (
                        <span className="inline-flex items-center px-2.5 py-1 rounded-full bg-red-100 text-red-800 text-sm font-medium">Above predicted by ₹{Math.abs(diff).toFixed(2)}</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </form>
  );
};

export default FlightForm;