# Customer assistant design notes

These behavior requirements were captured before the assistant existed, and
then actually implemented in [`generate_prompt/chatbot.py`](../generate_prompt/chatbot.py)'s
`SYSTEM_PROMPT`. Kept here as the design record; the prompt itself is the
current source of truth for exact wording.

## Bean-choice ordering flow

For a coffee product that has both a `"permanent"` (house blend) and a
`"seasonal"` variant of the same drink (identified by matching `name` with
different `context`/`availability`), the assistant:

1. Asks the customer which bean they want (house blend vs seasonal) before
   confirming an order, rather than assuming one.
2. For the **house blend** choice: quotes the price directly from
   `structure.json` -- it's stable and safe to state as fact.
3. For the **seasonal** choice: states the price as what was on the menu as
   of the last capture, and passes along the caveat to confirm the current
   price in-store, since seasonal batches (and their prices) rotate. This is
   already reflected in the data: every seasonal variant's `price_note` says
   as much (see `generate_embedding/menu.py`'s `MenuVariant.price_note`), and
   the assistant is instructed to surface it verbatim.

Currently `"latte"` (aliased `"white"`) and `"black"` are the only two
products with both a permanent and seasonal record in `structure.json` --
every other product has exactly one record. The prompt is explicit that the
bean-choice question is scoped to only those two names; an earlier version
phrased the rule generically enough that the model occasionally implied
other coffees also had a bean choice, which they don't.

Verified working: asking about a drink with both variants gets a
clarifying question before any price; following up with "the seasonal one"
correctly retrieves and quotes the seasonal variant with its price_note
attached to every size; asking about any other coffee does not trigger the
question at all.

## Price formatting and broad-question conciseness

Prices are always quoted one line per variant, in the compact form
`<name> <size>(<temperature>): <price>` (e.g. `white 5oz(hot): 13.0`) rather
than folded into a sentence -- `format_context()` in
[`generate_prompt/chatbot.py`](../generate_prompt/chatbot.py) pre-formats
each variant this way in the context block specifically so the model reuses
it verbatim instead of reformatting on its own.

For a broad question ("what coffee do you have"), the assistant names every
relevant product it was actually given in context, but lists names only --
it doesn't dump every size/temperature/price line for every product
unprompted, which reads as a database dump rather than a barista's answer.
Full pricing follows once the customer narrows to a specific drink or asks
for prices directly.

Retrieval's `top_k` was raised from an initial 5 to 15 (see `chatbot.py` and
`run_chatbot.ps1`) after live testing showed a narrower `top_k` let a
semantically "less central" but still clearly relevant item (an iced-only
specialty drink) fall just outside the cut on a broad coffee query.

## Best sellers and recommendations

`is_best_seller` (schema v5, `generate_embedding/menu.py`) is a marketing
fact with no OCR evidence behind it -- unlike every other product field, it
can't be read off a menu photo, so it's set by direct human confirmation
only and documented in that product's `issues` entry when set (see `"iced
vienna latte"` in `structure.json`, the only product currently flagged).
`format_context()` surfaces it as `best_seller: true` in the context block,
and the prompt allows the assistant to mention it proactively for "any
recommendations" / "what's popular" style questions, but never to claim a
product is a best seller unless that flag says so.

For a vague, uncategorized request ("any recommendations", "what's good"),
the assistant asks one short narrowing question (coffee, tea, or something
refreshing/cold) instead of dumping a mixed-category product list -- this is
a `SYSTEM_PROMPT` conversation rule, not a retrieval change; Pinecone still
returns whatever's semantically closest to the raw message; the prompt is
what decides to hold off listing until the customer narrows down.

Verified working: "what is the best seller" answers with the flagged
product and its prices; "any recommendations" asks the narrowing question
first; a follow-up "coffee please" then recommends the best seller and
names the other coffee options without dumping every price.

## Shop policies (not menu data)

Some customer questions aren't about any product at all -- "can I bring my
own coffee beans", opening hours, payment methods, wifi. These don't belong
in `structure.json` (every product fact there is grounded in a menu photo's
`source_ids`/`evidence`; a shop policy has no photo to cite) and they
shouldn't go through Pinecone retrieval either, since the whole list is tiny
and always relevant regardless of what the customer's message is about.

[`generate_prompt/shop_policies.py`](../generate_prompt/shop_policies.py)
loads [`generate_prompt/shop_policies.json`](../generate_prompt/shop_policies.json)
-- a flat list of `{topic, answer}` facts, entered directly by the shop and
reviewed by a human (the same provenance model as `is_best_seller`) -- and
hands the whole list to the model as a second block in its system
instruction, alongside `SYSTEM_PROMPT`, present on every turn. The prompt
treats "Menu context" and "Shop policies" as the two places facts are
allowed to come from; anything in neither gets the same "ask a barista"
fallback as an unanswerable menu question.

Verified working: "can I bring my own coffee beans" is answered directly
from the configured policy instead of falling back to "I'm not sure."

## Why this isn't encoded as pipeline/schema logic

`availability`, `price_note`, and `is_best_seller` (added to the schema and
to `structure.json` for exactly this reason) carry the facts the assistant
needs: filter/prefer by `availability`, surface `price_note` verbatim when
quoting a seasonal price, and mention `is_best_seller` when relevant. The
*branching conversation behavior itself* (ask first, phrase the caveat,
narrow before recommending, stay grounded in retrieved context, decline
off-topic requests) is assistant-prompt design, not menu data -- it lives in
`SYSTEM_PROMPT`, not in the pipeline that generates `structure.json`.

## Still open

- No actual ordering/checkout flow -- the assistant answers questions and
  states prices, but doesn't confirm or record an order.
- No structured price/size filtering ("show me anything under 15") --
  retrieval is semantic similarity only.
- Single-process, terminal-only; no persistence across sessions, no web/API
  front end yet.
