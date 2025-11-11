"use client"
import React, { useState } from 'react';

const LLMBar = () => {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Glassmorphism Bar */}
      <div
        className={`fixed left-1/2 -translate-x-1/2 w-[85%] z-50 rounded-lg border border-white/40 transition-all duration-500 ${open ? 'bottom-0 h-screen bg-white/80 backdrop-blur-2xl shadow-2xl' : 'bottom-0 h-15 bg-white/30 backdrop-blur-lg shadow-lg py-3'} text-center cursor-pointer`}
        onClick={() => !open && setOpen(true)}
      >
        {!open ? (
          <span className="font-semibold text-lg text-blue-700">Chat with AI</span>
        ) : (
          <div className="flex flex-col h-full">
            <button
              className="absolute top-4 right-4 text-gray-700 text-xl font-bold bg-white/60 rounded-full px-3 py-1 hover:bg-white"
              onClick={e => { e.stopPropagation(); setOpen(false); }}
            >
              ×
            </button>
            <div className="flex-1 flex flex-col justify-center items-center p-8">
              <h2 className="text-2xl font-bold mb-4 text-blue-700">AI Chatbot</h2>
              <div className="w-full h-full flex items-center justify-center">
                {/* Chatbot UI Placeholder */}
                <div className="w-full h-64 bg-white/60 rounded-lg flex items-center justify-center text-gray-500">
                  Chatbot interface goes here
                </div>
              </div>
            </div>
            <div className="p-4 border-t border-gray-200 bg-white/60 rounded-b-2xl">
              <input
                type="text"
                placeholder="Type your message..."
                className="w-full p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default LLMBar;