"use client"
import React, { useState, useRef, useEffect } from 'react';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

const LLMBar = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'Hello! 👋 I\'m your AI flight assistant. Tell me about your flight plans, and I\'ll help you find the best deals and booking recommendations.',
      timestamp: new Date(),
    },
  ]);
  const [userMessage, setUserMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userMessage.trim() || loading) return;

    const userMsg: Message = {
      sender: 'user',
      text: userMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setUserMessage('');
    setLoading(true);

    try {
      console.log('Sending message to /api/chat:', { text: userMessage });
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage }),
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('API Error:', errorData);
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      
      const botMsg: Message = {
        sender: 'bot',
        text: data.recommendation || data.message || 'Sorry, I couldn\'t process that request.',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error('Chat Error:', error);
      const errorText = error instanceof Error ? error.message : 'Unknown error occurred';
      const errorMsg: Message = {
        sender: 'bot',
        text: `Error: ${errorText}. Please check if the Mistral server is running at http://127.0.0.1:8001`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Glassmorphism Bar */}
      <div
        className={`fixed left-1/2 -translate-x-1/2 w-[85%] max-w-3xl z-50 rounded-lg border border-white/40 transition-all duration-500 ${
          open
            ? 'bottom-0 h-screen md:h-[600px] bg-white/95 backdrop-blur-2xl shadow-2xl'
            : 'bottom-0 h-16 bg-white/30 backdrop-blur-lg shadow-lg'
        }`}
      >
        {!open ? (
          <div
            className="h-full flex items-center justify-center cursor-pointer hover:bg-white/40 transition-colors"
            onClick={() => setOpen(true)}
          >
            <span className="font-semibold text-lg text-blue-700">💬 Chat with AI Assistant</span>
          </div>
        ) : (
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-blue-100 rounded-t-lg">
              <h2 className="text-xl font-bold text-blue-700">Flight Booking Assistant</h2>
              <button
                className="text-gray-600 text-2xl font-bold hover:text-gray-800 transition-colors"
                onClick={() => setOpen(false)}
              >
                ×
              </button>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                    } shadow-sm`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">{msg.text}</p>
                    <span className={`text-xs mt-2 block ${msg.sender === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white text-gray-800 border border-gray-200 rounded-lg rounded-bl-none px-4 py-3">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <form onSubmit={sendMessage} className="p-4 border-t border-gray-200 bg-white rounded-b-lg">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  placeholder="E.g., 'I want to fly from Delhi to Mumbai on Jan 15, Economy class'"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-800 placeholder-gray-400"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={loading || !userMessage.trim()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {loading ? '...' : 'Send'}
                </button>
              </div>
            </form>
          </div>
        )}
      </div>
    </>
  );
};

export default LLMBar;