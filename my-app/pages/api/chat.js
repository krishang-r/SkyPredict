"use client";

import { useState } from "react";

const LLMBar = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([{ sender: "bot", text: "Hello! How can I assist you today?" }]);
  const [userMessage, setUserMessage] = useState("");

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const newMessages = [...messages, { sender: "user", text: userMessage }];
    setMessages(newMessages);
    setUserMessage("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userMessage }),
      });
      const data = await response.json();
      setMessages([...newMessages, { sender: "bot", text: data.reply }]);
    } catch (error) {
      setMessages([...newMessages, { sender: "bot", text: "Sorry, something went wrong." }]);
    }
  };

  return (
    <div className={`fixed bottom-0 w-full ${open ? "h-64" : "h-12"} bg-gray-800 text-white`}>
      <div className="flex items-center justify-between px-4 py-2">
        <span>Chatbot</span>
        <button onClick={() => setOpen(!open)}>{open ? "Close" : "Open"}</button>
      </div>
      {open && (
        <div className="flex flex-col h-full p-4 overflow-y-auto">
          <div className="flex-grow">
            {messages.map((msg, idx) => (
              <div key={idx} className={`my-2 ${msg.sender === "bot" ? "text-left" : "text-right"}`}>
                <span className={`inline-block px-4 py-2 rounded ${msg.sender === "bot" ? "bg-blue-500" : "bg-green-500"}`}>
                  {msg.text}
                </span>
              </div>
            ))}
          </div>
          <div className="flex items-center">
            <input
              type="text"
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              className="flex-grow px-4 py-2 border rounded"
              placeholder="Type your message..."
            />
            <button onClick={sendMessage} className="ml-2 px-4 py-2 bg-blue-500 text-white rounded">
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default LLMBar;

export default async function handler(req, res) {
  if (req.method === "POST") {
    const { userMessage } = req.body;

    try {
      // Call the Mistral server at the correct endpoint
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: userMessage,
          max_tokens: 150,
        }),
      });

      const data = await response.json();
      res.status(200).json({ reply: data.text }); // Adjust based on Mistral's response format
    } catch (error) {
      console.error("Error calling Mistral server:", error);
      res.status(500).json({ error: "Failed to connect to Mistral server." });
    }
  } else {
    res.status(405).json({ error: "Method not allowed" });
  }
}