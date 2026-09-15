"use client";

import { useState } from "react";
import { Message } from "../lib/types";
import { SourceCard } from "./SourceCard";
import { VerificationBadge } from "./VerificationBadge";

export function ChatMessage({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] md:max-w-[75%] space-y-2`}>
        {/* Bubble */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-sky-600 text-white rounded-br-sm"
              : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm shadow-sm"
          }`}
        >
          {message.content}
        </div>

        {/* Verifier badge */}
        {!isUser && message.verification && (
          <div className="ml-1">
            <VerificationBadge verification={message.verification} />
          </div>
        )}

        {/* Sources toggle */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="space-y-2">
            <button
              onClick={() => setShowSources((v) => !v)}
              className="text-xs text-sky-600 hover:text-sky-800 font-medium flex items-center gap-1 ml-1"
            >
              <span>{showSources ? "▲" : "▼"}</span>
              {showSources ? "Hide" : "Show"} {message.sources.length} source
              {message.sources.length !== 1 ? "s" : ""}
            </button>

            {showSources && (
              <div className="space-y-1.5">
                {message.sources.map((src, i) => (
                  <SourceCard key={i} source={src} index={i} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
