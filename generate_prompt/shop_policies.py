"""Shop-level facts that apply to the whole business, not any one menu
product -- e.g. whether customers can bring their own coffee beans, opening
hours, payment methods. Kept separate from structure.json because these
aren't menu items: every product fact in structure.json is grounded in a
menu photo (source_ids/evidence), but a shop policy has no menu photo to
cite. It's entered directly by the shop and reviewed by a human -- the same
provenance model already used for `MenuProduct.is_best_seller`.

Small and static enough to just hand the whole list to the model in its
system instruction every turn, rather than retrieving a subset per message
the way menu products are.
"""
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Nonempty = Annotated[str, Field(min_length=1)]


class ShopPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    topic: Nonempty
    answer: Nonempty


def load_shop_policies(path: Path) -> list[ShopPolicy]:
    """Missing file means "no policies configured yet", not an error -- the
    bot still works, it just has nothing extra to say beyond menu context."""
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of shop policy objects")
    return [ShopPolicy.model_validate(item) for item in raw]


def format_shop_policies(policies: list[ShopPolicy]) -> str:
    if not policies:
        return ""
    lines = "\n".join(f"- {p.topic}: {p.answer}" for p in policies)
    return (
        "Shop policies (general facts about the shop itself, always true "
        "regardless of what's in the Menu context below):\n" + lines
    )
