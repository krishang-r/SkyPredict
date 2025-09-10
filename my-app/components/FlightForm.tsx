"use client"
import React, { useState } from 'react';

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

const FlightForm = () => {
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [date, setDate] = useState('');
  const [passengers, setPassengers] = useState(1);
  const [flightClass, setFlightClass] = useState(classes[0]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Add booking logic here
    alert(`Booking flight from ${origin} to ${destination} on ${date} for ${passengers} passenger(s) in ${flightClass} class.`);
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-7xl mx-auto mt-10 p-8 rounded-xl shadow-2xl bg-white text-black">
  <h2 className="text-3xl font-extrabold mb-8 text-center tracking-tight text-blue-700">Predict Your Flight Price</h2>
      <div className="flex flex-col gap-6 md:flex-row md:items-end md:gap-8 justify-between">
        <div className="flex flex-col w-full md:w-1/5">
          <label className="mb-2 font-semibold text-lg text-gray-700">Origin</label>
          <select
            className="p-3 border border-gray-300 rounded-lg text-base font-medium focus:outline-none focus:ring-2 focus:ring-blue-400 h-16 w-full"
            value={origin}
            onChange={e => setOrigin(e.target.value)}
            required
          >
            <option value="" disabled>Select origin airport</option>
            {airports.map((airport) => (
              <option key={airport} value={airport}>{airport}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col w-full md:w-1/5">
          <label className="mb-2 font-semibold text-lg text-gray-700">Destination</label>
          <select
            className="p-3 border border-gray-300 rounded-lg text-base font-medium focus:outline-none focus:ring-2 focus:ring-blue-400 h-16 w-full"
            value={destination}
            onChange={e => setDestination(e.target.value)}
            required
          >
            <option value="" disabled>Select destination airport</option>
            {airports.map((airport) => (
              <option key={airport} value={airport}>{airport}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col w-full md:w-1/5">
          <label className="mb-2 font-semibold text-lg text-gray-700">Departure Date</label>
          <input
            type="date"
            className="p-3 border border-gray-300 rounded-lg text-base font-medium focus:outline-none focus:ring-2 focus:ring-blue-400 h-16 w-full"
            value={date}
            onChange={e => setDate(e.target.value)}
            required
          />
        </div>
        <button
          type="submit"
          className="w-full md:w-auto bg-blue-600 text-white py-3 px-8 rounded-lg font-bold text-lg hover:bg-blue-700 transition mt-4 md:mt-0 shadow-md"
        >
          Predict Price
        </button>
      </div>
    </form>
  );
};

export default FlightForm;