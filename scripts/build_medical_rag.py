#!/usr/bin/env python3
"""Build local Medical RAG vector database from guideline PDFs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import cfg
from app.medical_rag import get_medical_rag


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pdf_dir = root / cfg.MEDICAL_GUIDELINES_DIR
    db_dir = root / cfg.MEDICAL_RAG_DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)

    rag = get_medical_rag(str(db_dir))
    count = asyncio.run(rag.build_from_pdf_dir(str(pdf_dir)))

    print(f"PDF directory: {pdf_dir}")
    print(f"DB directory: {db_dir}")
    print(f"Indexed chunks: {count}")


if __name__ == "__main__":
    main()
