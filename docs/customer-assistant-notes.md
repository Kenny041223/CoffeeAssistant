# Customer assistant design notes

Requirements captured ahead of actually building the customer-facing
assistant (still future work per the README's "Next milestones" -- there is
no assistant code yet, so these are notes for that milestone, not a
description of anything running today).

## Bean-choice ordering flow

For a coffee product that has both a `"permanent"` (house blend) and a
`"seasonal"` variant of the same drink (identified by matching `name` with
different `context`/`availability` -- see `latte` in `structure.json` for the
current example), the assistant should:

1. Ask the customer which bean they want (house blend vs seasonal) before
   confirming an order, rather than assuming one.
2. For the **house blend** choice: quote the price directly from
   `structure.json` -- it's stable and safe to state as fact.
3. For the **seasonal** choice: state the price as what was on the menu as of
   the last capture, but tell the customer to confirm the current price
   in-store, since seasonal batches (and their prices) rotate. This is
   already reflected in the data: every seasonal variant's `price_note`
   says as much (see `app/models/menu.py`'s `MenuVariant.price_note`).

## Why this isn't encoded as pipeline/schema logic

`availability` and `price_note` (added to the schema and to `structure.json`
for exactly this reason) already carry the facts an assistant needs to
implement this flow: filter/prefer by `availability`, and surface
`price_note` verbatim when quoting a seasonal price. The *branching
conversation behavior itself* (ask first, phrase the caveat, etc.) is
assistant-prompt design, not menu data, so it belongs here until the
assistant exists, then in its own prompt/tool-calling logic.
