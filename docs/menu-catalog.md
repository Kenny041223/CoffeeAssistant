# Consolidated menu catalog

Run after generating `structure.json`:

```powershell
.\.venv\Scripts\python.exe -m app.services.build_catalog
```

`structure.json` preserves the original per-image LLM extraction. `menu_catalog.json`
is the consolidated draft for the next retrieval stage. It combines repeated
products, descriptions and price variants, with source references on each claim.
The initial consolidated draft has 34 product entries and two series records.
This count is not a claim of complete or verified menu accuracy.

`data/catalog-review.json` holds explicit correction decisions, separate from code.
These are based on the existing OCR text and user feedback. In particular, source
12 temperatures were corrected to iced by the user; this fact was not in the OCR
text. The original raw extraction is unchanged. No model or image verification is
run by this command.

Vienna latte joins its descriptive poster with the price menus. Einspanner's
misread name is repaired from its source transcription. Coconut series and tea
series become group records, not standalone drinks. Group descriptions remain
group-level context: not every coconut drink contains every listed topping.
Strawberry coconut cloud remains distinct from strawberry coconut. House blend
and seasonal specials remain distinct. Mocha hot and iced become variants of one
product. S/L sizes normalize to small/large; ounce sizes are not guessed to match.

The review file is bound to a digest of the entire input extraction. Regenerating
or editing the extraction requires updating the review before rebuilding. This
prevents index-based decisions applying silently to different records. Updating
the digest alone is not a review: check the selectors and decisions first.

Exact normalized names merge within the configured blend context. Additional
aliases are explicit review decisions. Conflicting non-null prices for the same
size and temperature are retained and flagged. Unknown fields remain unknown;
the builder does not infer a temperature from a drink's name unless reviewed,
and does not resolve ambiguous table alignment. Source 09/14 table rows still
need image review, as do model-produced add-on applicability lists. This is a
partial content cleanup, not a complete verification pipeline.

For future vector ingestion, prepare one readable document per consolidated
product, including descriptions, applicable series context and source IDs. Keep
the series documents separately typed. Do not embed the whole JSON as one chunk
or count each source occurrence as a different drink. The raw sources remain
useful for citations and investigation. Store exact prices and variants in
structured storage for lookup; embeddings do not resolve contradictions or prove
facts. No vector database has been created or populated by this step.
