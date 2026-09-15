"use client";

import { useState } from "react";
import { Source } from "../lib/types";

export function SourceCard({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false);

  const pageLabel =
    source.page_start === source.page_end
      ? `p.${source.page_start}`
      : `pp.${source.page_start}–${source.page_end}`;

  const breadcrumb = [source.chapter, source.section, source.subsection]
    .filter(Boolean)
    .join(" › ");

  return (
    <div className="border border-sky-200 rounded-lg bg-sky-50 text-sm overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-3 py-2 flex items-start justify-between gap-2 hover:bg-sky-100 transition-colors"
      >
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sky-800">#{index + 1}</span>
          <span className="ml-2 text-sky-700 truncate">{breadcrumb}</span>
          <span className="ml-2 text-sky-500 font-mono text-xs">{pageLabel}</span>
        </div>
        <span className="text-sky-400 mt-0.5 flex-shrink-0">
          {open ? "▲" : "▼"}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-sky-200">
          <p className="text-slate-600 leading-relaxed line-clamp-6">{source.excerpt}</p>
        </div>
      )}
    </div>
  );
}
