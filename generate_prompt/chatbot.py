"""Customer-facing chat assistant: retrieval-augmented, grounded in
structure.json, never left to invent menu facts on its own.

Per turn: embed the customer's message with the SAME Gemini embedding model
used to build the Pinecone index (a different embedding model would put the
query in a different vector space and break similarity search entirely).
Search Pinecone for the closest products, then look up each one's FULL
record in structure.json -- Pinecone's metadata is a flattened summary for
filtering, not the source of truth. That grounding context is handed to
Gemini fresh on every turn; Gemini's own chat session carries the
conversation history. Everything here runs hosted -- no local model or GPU
needed for either retrieval or reply generation.

`build_engine()`/`reply()` are the reusable core: build one ChatEngine at
startup, then call reply() per incoming message with a per-customer
Conversation. The terminal loop in main() below is one caller of that core;
connect_whatsapp/whatsapp_server.py (one Conversation per WhatsApp contact
instead of one global one) is another -- both call the exact same retrieval
and prompting logic rather than each reimplementing it.

No network clients connect until build_engine() actually runs.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_embedding.embed_menu import GeminiEmbedder
from generate_embedding.menu import MenuProduct, StructuredMenu
from generate_prompt.shop_policies import format_shop_policies, load_shop_policies

SYSTEM_PROMPT = """You are the customer-facing ordering assistant for this coffee shop.
Answer only using the "Menu context" block you're given each turn, and the "Shop
policies" block below (if present) -- together they are the complete set of facts
you're allowed to state. Menu context is the products retrieved as relevant to the
customer's message; shop policies are general facts about the business itself
(not tied to any one product) and are always true regardless of what's retrieved
this turn. Never state a product, price, or fact that isn't in one of those two
places. If neither covers what the customer asked about (or they ask something
no menu context could ever answer, like a specific ingredient's brand/origin, an
allergen guarantee, or today's stock), say so plainly and point them to ask a
barista in the shop -- never say "I'll check" or "let me check" or anything
implying you yourself will go find out; you have no way to actually do that.

Rules:
- Only "white" and "black" coffee each have two versions: one "permanent" (house
  blend, always available) and one "seasonal" (rotates, may not be current). When
  BOTH versions of the SAME name show up in your context together, ask which one
  the customer wants BEFORE quoting a price -- don't assume. Every other item has
  only one version in the context (no permanent/seasonal split) -- never imply one
  exists or ask a bean-choice question for anything other than white or black.
- For a "permanent" item, state its price directly; it's stable.
- For a "seasonal" item, always pass along its price_note verbatim (something like
  "reconfirm current price in-store") alongside the price -- never quote a seasonal
  price as if it were guaranteed current.
- Two different customer intents need two different answer shapes -- tell them
  apart by what's actually being asked, not just by topic:
  1. BROWSE: the customer explicitly asks to see a category -- "what coffee do
     you have", "what do you have", "I'd like to see the teas". Here, name EVERY
     relevant product actually present in your context; don't silently drop one
     because the list is long. But still don't dump every size/temperature/price
     line for every product at once -- that's overwhelming, not a menu browse.
     Hold off on prices until the customer names one drink or asks for prices.
  2. RECOMMEND: the customer expresses a want/mood rather than asking to see a
     list -- "any recommendations", "I want something refreshing", "I feel like
     tea", "surprise me", "coffee please" (as an answer to being asked). Here,
     do NOT enumerate every matching product from context. Give a short, curated
     answer instead: point to series/category names when several matches share
     one (e.g. "you could try our Coconut series, or take a look at our teas")
     and/or call out one or two standout picks (the best seller is a natural
     one to mention) -- roughly what a barista would say off the top of their
     head, not a printout of the retrieved list. If more than one series or
     category plausibly fits (e.g. "something refreshing" could point to both a
     coconut-water series AND iced tea options), mention more than one rather
     than fixating on just the first match. If the customer then wants the full
     list, they'll ask for it, which becomes a BROWSE request.
- For a RECOMMEND request that gives NO category at all ("any recommendations",
  "what should I get", "what's good"), ask ONE short narrowing question first:
  whether they're in the mood for coffee, tea, or something refreshing/cold.
- The moment the customer's message already answers that -- names a category
  outright ("something refreshing", "I feel like tea", "coffee please") -- that
  message IS the answer. Do not ask the narrowing question again in any
  rephrasing, and do not ask a second follow-up question to narrow further
  (e.g. don't reply to "something refreshing" by asking "refreshing coffee or
  refreshing tea?"). Go straight to a short RECOMMEND-style answer as above.
- If a product's context marks it "best_seller: true", you may proactively
  mention it as a popular pick when it's natural to do so (answering "any
  recommendations" once narrowed, or "what's popular"/"what's your best
  seller"). Never call something a best seller unless the context says so.
- Once the customer has narrowed to a specific drink, or explicitly asks for
  prices, give one line per variant in this compact form, not a prose sentence:
  <name> <size>(<temperature>): <price>
  e.g. "white 5oz(hot): 13.0" / "black 8oz(iced): 15.0". The "Menu context" block
  already gives you each variant pre-formatted exactly this way -- reuse those
  lines as-is rather than reformatting them. If a product has only one variant (or
  is iced-only with no hot option, like vienna latte), still use this same
  one-line form rather than folding the price into a sentence.
- No currency symbol is confirmed for this menu (prices are plain numbers). Don't
  invent one (no "$", "RM", etc.) -- state the number and, if asked, say the
  currency isn't confirmed.
- If a product's context notes an uncertainty or issue relevant to what's being
  asked (e.g. an unclear ingredient, a possible naming ambiguity), mention it
  briefly rather than stating an uncertain fact as settled.
- A few products share a word in their name but are NOT the same drink family --
  e.g. several "... coconut" drinks are coffee, but "matcha coconut" is a tea
  (matcha-based, not coffee) that happens to share the "Coconut series" name with
  them. Trust each product's own "category" field, not its name, and if a
  same-named-ish coffee item and tea item both show up together, briefly clarify
  which is which rather than letting the customer assume they're interchangeable.
- Stay on topic: you help with this menu and ordering from it. Politely decline
  unrelated requests.
- Be warm and concise, like a barista, not a database dump. Don't list raw JSON.
"""


def format_context(products: list[MenuProduct]) -> str:
    if not products:
        return "(No matching products found for this message.)"
    blocks = []
    for product in products:
        lines = [f"- {product.name}" + (f" ({product.context})" if product.context else "")]
        if product.availability:
            lines.append(f"  availability: {product.availability}")
        if product.is_best_seller:
            lines.append("  best_seller: true")
        if product.category:
            lines.append(f"  category: {product.category}")
        if product.series:
            lines.append(f"  series: {', '.join(product.series)}")
        if product.description:
            lines.append(f"  description: {product.description}")
        if product.aliases:
            lines.append(f"  also called: {', '.join(product.aliases)}")
        if product.variants:
            for variant in product.variants:
                # Pre-formatted as "<name> <size>(<temperature>): <price>" --
                # the exact line-per-variant style the bot should reuse
                # verbatim when quoting prices, rather than inventing its
                # own formatting each turn.
                size = variant.size or ""
                temp = f"({variant.temperature})" if variant.temperature else ""
                label = "".join(p for p in (size, temp) if p) or "standard"
                price = f"{variant.price}" if variant.price is not None else "price not listed"
                note = f" ({variant.price_note})" if variant.price_note else ""
                lines.append(f"  - {product.name} {label}: {price}{note}")
        else:
            lines.append("  (no priced variants on record)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def retrieve(user_message: str, embedder: GeminiEmbedder, index, menu_by_id: dict[str, MenuProduct],
             top_k: int, last_reply: str | None = None, namespace: str | None = None) -> list[MenuProduct]:
    # A short follow-up ("the seasonal one please") carries almost no
    # retrievable meaning on its own -- the topic lives in the previous
    # turn. Folding the last reply into the search embedding (but never
    # into what's sent to Gemini as the actual message) fixes that without
    # a separate query-rewriting call.
    search_text = f"{last_reply}\n{user_message}" if last_reply else user_message
    vector = embedder.encode_query(search_text)
    matches = index.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=False).matches
    return [menu_by_id[m.id] for m in matches if m.id in menu_by_id]


@dataclass
class ChatEngine:
    """Everything a reply needs that's shared across every customer and
    every turn -- built once at process startup, then read-only after
    that. Safe to share across concurrently-handled WhatsApp conversations
    since nothing here is mutated per-turn (per-customer state lives in
    Conversation instead)."""
    menu_by_id: dict[str, MenuProduct]
    embedder: GeminiEmbedder
    index: Any  # pinecone.Index -- left untyped so this module never has to import pinecone at all
    top_k: int
    genai_client: Any  # google.genai.Client
    gemini_model: str
    system_instruction: str
    namespace: str | None = None


@dataclass
class Conversation:
    """Per-customer state: one of these per person the bot is talking to.
    In the terminal chatbot there's exactly one, for the life of the
    process; the WhatsApp server keeps one per phone number."""
    chat: Any  # google.genai chat session (client.chats.create(...))
    last_reply: str | None = None


def build_engine(menu_path: Path, pinecone_key: str, gemini_key: str, index_name: str,
                  namespace: str | None, top_k: int, gemini_model: str,
                  shop_policies_path: Path) -> ChatEngine:
    """Loads the menu, connects to Pinecone and Gemini, and assembles the
    system instruction (base prompt + shop policies, if any). Does not
    start a conversation -- call new_conversation() per customer."""
    menu = StructuredMenu.model_validate_json(menu_path.read_text(encoding="utf-8"))
    menu_by_id = {p.product_id: p for p in menu.products}
    shop_policies = load_shop_policies(shop_policies_path)

    embedder = GeminiEmbedder(api_key=gemini_key, batch_size=1)

    from pinecone import Pinecone
    index = Pinecone(api_key=pinecone_key).Index(index_name)

    from google import genai
    genai_client = genai.Client(api_key=gemini_key)

    system_instruction = SYSTEM_PROMPT
    policies_block = format_shop_policies(shop_policies)
    if policies_block:
        system_instruction = f"{SYSTEM_PROMPT}\n\n{policies_block}"

    return ChatEngine(menu_by_id=menu_by_id, embedder=embedder, index=index, top_k=top_k,
                       genai_client=genai_client, gemini_model=gemini_model,
                       system_instruction=system_instruction, namespace=namespace)


def new_conversation(engine: ChatEngine) -> Conversation:
    from google.genai import types
    chat = engine.genai_client.chats.create(
        model=engine.gemini_model,
        config=types.GenerateContentConfig(system_instruction=engine.system_instruction),
    )
    return Conversation(chat=chat)


def reply(engine: ChatEngine, conversation: Conversation, user_message: str) -> str:
    """Retrieve relevant products, hand them to this customer's Gemini chat
    session as fresh per-turn context, and return the reply text. Mutates
    conversation.last_reply so the NEXT call's retrieval can use it."""
    products = retrieve(user_message, engine.embedder, engine.index, engine.menu_by_id,
                         engine.top_k, conversation.last_reply, engine.namespace)
    turn_content = (
        f"Menu context (retrieved for this message only):\n{format_context(products)}\n\n"
        f"Customer: {user_message}"
    )
    response = conversation.chat.send_message(turn_content)
    conversation.last_reply = response.text
    return response.text


def main() -> int:
    # Windows' console codepage isn't UTF-8 by default; without this, characters
    # Gemini commonly uses (em dashes, curly quotes) print as "?" / mojibake.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menu", type=Path, default=Path("structure.json"))
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX", "coffee-menu"))
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", ""))
    # The whole menu is only ~23 products; a narrow top_k let semantically
    # "less central" items (e.g. an iced-only specialty drink) fall just
    # outside the cut on a broad query like "what coffee do you have" even
    # though they're clearly relevant. Verified against the live index that
    # 15 reliably covers every coffee-category product for broad queries
    # while still leaving out the clearly-unrelated tail.
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--shop-policies", type=Path,
                         default=Path(__file__).with_name("shop_policies.json"))
    args = parser.parse_args()

    if not args.menu.is_file():
        print(f"Menu file not found: {args.menu}. Run menu generation first.", file=sys.stderr)
        return 1
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not pinecone_key:
        print("PINECONE_API_KEY is not set.", file=sys.stderr)
        return 1
    if not gemini_key:
        print("GEMINI_API_KEY is not set.", file=sys.stderr)
        return 1

    engine = build_engine(
        menu_path=args.menu, pinecone_key=pinecone_key, gemini_key=gemini_key,
        index_name=args.index, namespace=args.namespace or None, top_k=args.top_k,
        gemini_model=args.gemini_model, shop_policies_path=args.shop_policies,
    )
    conversation = new_conversation(engine)

    print(f"Ready ({len(engine.menu_by_id)} products indexed). Type a message, or 'quit' to exit.\n")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in ("quit", "exit"):
            break

        print(f"Bot: {reply(engine, conversation, user_message)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
