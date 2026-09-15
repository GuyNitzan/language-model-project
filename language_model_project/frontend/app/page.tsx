"use client";

import { useEffect, useRef, useState } from "react";
import { sendMessage } from "./lib/api";
import { Message } from "./lib/types";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(text: string) {
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res = await sendMessage(text, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          verification: res.verification,
        },
      ]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="flex-shrink-0 bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-sky-600 flex items-center justify-center text-white text-sm font-bold">
          D
        </div>
        <div>
          <h1 className="font-semibold text-slate-900 text-sm leading-tight">
            Dermatology Assistant
          </h1>
          <p className="text-xs text-slate-500">
            Powered by Bolognia&apos;s Dermatology 5th Ed.
          </p>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center mt-16 space-y-3">
              <p className="text-2xl">🔬</p>
              <p className="text-slate-600 font-medium">
                Ask any dermatology question
              </p>
              <p className="text-slate-400 text-sm max-w-sm mx-auto">
                Answers are sourced directly from Bolognia&apos;s Dermatology
                (5th edition, 2024) with full citations.
              </p>
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                {[
                  "What are the criteria for diagnosing psoriasis?",
                  "How is melanoma staged?",
                  "What causes atopic dermatitis?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="text-xs bg-white border border-slate-200 rounded-full px-3 py-1.5 text-slate-600 hover:border-sky-400 hover:text-sky-700 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage key={i} message={msg} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1 items-center h-4">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="w-2 h-2 bg-sky-400 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 150}ms` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="text-center text-sm text-red-500 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <div className="flex-shrink-0 border-t border-slate-200 bg-slate-50 px-4 py-3">
        <div className="max-w-2xl mx-auto">
          <ChatInput onSend={handleSend} disabled={loading} />
          <p className="text-center text-xs text-slate-400 mt-2">
            For educational use only — not a substitute for clinical judgment.
          </p>
        </div>
      </div>
    </div>
  );
}
