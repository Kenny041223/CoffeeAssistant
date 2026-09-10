"""Consolidate source extractions using explicit, version-bound review decisions."""
import argparse
import hashlib
import json
from pathlib import Path

from app.models.menu import StructuredMenu, MenuItem


def normalized(value):
    return " ".join(value.casefold().split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def build_catalog(document, review):
    # Bind decisions to the exact extraction: never silently apply an old review
    # to a new model response, reordered items, or changed OCR evidence.
    if digest(document.model_dump()) != review["input_digest"]:
        raise ValueError("Extraction changed; update catalog review before rebuilding")
    products = {}
    series = {}
    for entry in review["series"]:
        series[entry["name"]] = {**entry, "product_ids": []}
    for source in document.sources:
        for index, original in enumerate(source.menu.items):
            ref = {"source_id": source.source_id, "item_index": index}
            rule = review["items"].get(f"{source.source_id}:{index}", {})
            if rule.get("exclude"):
                continue
            data = original.model_dump()
            data.update(rule.get("set", {}))
            item = MenuItem.model_validate(data)
            name = rule.get("canonical_name", item.name)
            context = item.section if item.section in review["separate_sections"] else None
            key = (normalized(name), context)
            if key not in products:
                products[key] = {"product_id": "drink-" + digest(key)[:12],
                    "name": name, "context": context, "aliases": [], "categories": [],
                    "descriptions": [], "variants": [], "series": [], "source_refs": [],
                    "issues": [], "status": "draft"}
            product = products[key]
            product["source_refs"].append(ref)
            if original.name not in product["aliases"]:
                product["aliases"].append(original.name)
            if item.category:
                add_claim(product["categories"], {"text": item.category}, ref)
            if item.description:
                add_claim(product["descriptions"], {"text": item.description}, ref)
            for variant in item.variants:
                value = variant.model_dump()
                value["size"] = {"s": "small", "l": "large"}.get(
                    normalized(value["size"] or ""), value["size"])
                add_claim(product["variants"], value, ref)
            for group in rule.get("series", []):
                if group not in product["series"]:
                    product["series"].append(group)
                if product["product_id"] not in series[group]["product_ids"]:
                    series[group]["product_ids"].append(product["product_id"])
            product["issues"].extend(rule.get("issues", []))
    for product in products.values():
        variants = product["variants"]
        for i, left in enumerate(variants):
            for right in variants[i + 1:]:
                same = (left["size"], left["temperature"]) == (right["size"], right["temperature"])
                if same and left["price"] is not None and right["price"] is not None and left["price"] != right["price"]:
                    product["issues"].append("Conflicting prices for the same size and temperature; resolve before quoting.")
        product["issues"] = list(dict.fromkeys(product["issues"]))
    return {"schema_version": 1, "status": "draft", "currency": document.currency,
        "input_digest": review["input_digest"], "product_count": len(products),
        "products": list(products.values()), "series": list(series.values()),
        "addons_by_source": [{"source_id": s.source_id, "addons": [a.model_dump() for a in s.menu.addons]}
                             for s in document.sources if s.menu.addons],
        "review_decisions": review["items"],
        "sources": [{"source_id": s.source_id, "image_file": s.image_file,
                     "image_sha256": s.image_sha256, "ocr_file": s.ocr_file} for s in document.sources],
        "notes": ["Catalog combines overlapping source records; original structure.json is preserved.",
                  "Review decisions are based on OCR text and user feedback, not a full image verification.",
                  "Series descriptions describe the group; they are not a list of ingredients in every drink.",
                  "Missing values and unresolved variants remain unknown. Do not treat draft prices as verified.",
                  "Add-on scope is inherited from the model and still requires review."]}


def add_claim(claims, value, ref):
    for claim in claims:
        if {k: v for k, v in claim.items() if k != "source_refs"} == value:
            if ref not in claim["source_refs"]:
                claim["source_refs"].append(ref)
            return
    claims.append({**value, "source_refs": [ref]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("structure.json"))
    parser.add_argument("--review", type=Path, default=Path("data/catalog-review.json"))
    parser.add_argument("--output", type=Path, default=Path("menu_catalog.json"))
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.review.resolve()}:
        parser.error("Output must not overwrite extraction or review input")
    document = StructuredMenu.model_validate_json(args.input.read_bytes())
    result = build_catalog(document, json.loads(args.review.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".tmp")
    pending.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pending.replace(args.output)
    print(f"Saved {args.output}: {result['product_count']} products, {len(result['series'])} series")


if __name__ == "__main__":
    main()
