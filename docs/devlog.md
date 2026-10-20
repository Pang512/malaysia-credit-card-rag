# Development log: Malaysian Credit Card RAG Assistant

A running record of problems found, fixes made, and before/after evidence.
Use it for the README, interview stories and LinkedIn posts.

---

## Knowledge base (frozen for the v1 baseline)

| Bank | Document | Type | Valid as at |
|---|---|---|---|
| Maybank | Cash Back Mastercard Platinum PDS | Conventional, one card | unknown |
| CIMB | Credit Cards PDS (bilingual) | Conventional, multi-card | 2024-08-24 |
| OCBC | Credit Cards PDS | Conventional, multi-card | 2026-07 |
| Public Bank | Credit Cards PDS | Conventional, multi-card | 2026-10-01 |
| Maybank Islamic | Nadi Mastercard Credit Card-i PDS | Islamic (Tawarruq), one card | unknown |
| Bank Islam | Credit Card-i PDS | Islamic (Tawarruq), multi-card | 2025-12-31 |
| Bank Muamalat | Credit Card-i PDS | Islamic (Tawarruq), multi-card | 2026-01-01 |
| BSN | Al-Aiman Credit Card-i PDS | Islamic (Tawarruq), multi-card | 2025-11 |

8 banks, 33 pages. Hong Leong excluded for now (website unreachable).

---

## Fix log

### 1. Contextual chunk headers
- **Problem:** fixed-size chunks often never named their bank. The Bank Islam late payment chunk only said "the Bank".
- **Fix:** structured chunking that keeps whole lines and adds a header: `[Bank | Product | page N]`.
- **Before / after:** "Bank Islam late payment charge" question. Correct chunk ranked #2 with fixed chunks, #1 with structured chunks (score 0.847, highest of all test searches).

### 2. Per-page language detection
- **Problem:** the CIMB PDS has English pages 1 to 6 and Bahasa Malaysia pages 7 to 12, but every page was tagged English.
- **Fix:** a simple word-count rule that labels each page `en` or `ms`. Only English pages are indexed, because the embedding model is English-only and the Malay pages duplicate the English content.
- **Result:** index went from 140 to 120 fixed chunks and 69 to 58 structured chunks, with no duplicate content.

### 3. Stale source detected
- **Problem:** the BSN PDS was valid as at June 2021, years older than the others.
- **Fix:** added a `valid_as_at` field to every document; replaced BSN with the November 2025 PDS.
- **Finding:** BSN had changed the card's Shariah structure from **Ujrah** to **Tawarruq**. An old document would have given wrong answers about the card's core concept.

### 4. Token limit check
- **Problem:** the embedding model silently truncates text over 512 tokens.
- **Fix:** measured token counts for every chunk at index time.
- **Result:** longest structured chunk is 484 tokens. 0 chunks truncated. Chunk size cannot grow without losing text.

### 5. Model selection by measurement
| Model | Correct (2 questions) | LLM time | Availability |
|---|---|---|---|
| gemini-3.8-flash | 1 of 1 answered | 329 s | 503 errors |
| gemini-3.5-flash | 2 of 2 | 9 to 14 s | occasional 503 |
| gemini-3.5-flash-lite | 2 of 2 | 1 to 2 s | no errors |

- **Decision:** `gemini-3.5-flash-lite` as default, `gemini-3.5-flash` as fallback. The newest model was the slowest and least available.

### 6. Resilience
- **Problem:** Google returned 503 (overloaded) and the script crashed.
- **Fix:** retry 429/500/503 errors with exponential backoff, then switch to a fallback model; record which model answered. Fallback is disabled during evaluation for fair comparisons.
- **Result:** answers still delivered during 503 periods.

### 7. Grounding works
- Hong Leong WISE question (not in the knowledge base): the model answered "I don't know based on the available documents" instead of using its training-data knowledge.

### 8. Known failures logged for the improvement round
| Question | Failure stage | Cause |
|---|---|---|
| Maybank Nadi profit rate, 10 of 12 months | Extraction | Merged table cell (14% p.a.) extracted next to the first row only |
| "Can I get a card without proof of income?" | Retrieval | Vocabulary mismatch: documents say "fixed deposit", "pledge", "income evidence". Top score only 0.667 |
| "Annual fee for OCBC 365" | Retrieval ranking | Multi-card fee table ranked #2 behind a less relevant OCBC chunk |

---

## Test set
- 50 questions in `eval/test_set.csv`, drafted with AI assistance and manually verified against the source PDFs.
- Categories: fact 19, table 9, eligibility 5, islamic 4, comparison 4, unanswerable 4, advisory 2, aggregation 1, multi_source 1, extraction_trap 1.
- 5 questions written by the author as a real user (shortlist by income, criteria, list all cards, travel, Islamic concepts). Open-ended ones are graded with a rubric.

---

## Coming next (before/after stories to fill in)
- [ ] Baseline evaluation: hit@k, MRR, answer correctness per category
- [ ] Hybrid search (BM25 + dense) and reranker
- [ ] Table-aware extraction (Maybank Nadi merged cell)
- [ ] Card catalog + tool calling for shortlist, filter and compare questions
- [ ] v2: Maybank multi-card PDS, Hong Leong, RHB, AmBank, card benefit documents
