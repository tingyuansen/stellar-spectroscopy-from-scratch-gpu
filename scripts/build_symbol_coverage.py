#!/usr/bin/env python3
"""Join the Payne Zero AST inventory to the human coverage ledger.

``COVERAGE.md`` owns the pedagogical placement and verification gate for every
module. ``paynezero_symbols.json`` owns the pinned source surface. This script
fails when either side contains a module absent from the other, then expands the
module assignment to every public export, function, class, constructor, method,
field, and named module datum.

The first expansion deliberately records ``mapping_precision=module_default``.
Chapter construction may add narrower symbol overrides, but no public source
object is allowed to remain invisible while that refinement proceeds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from apply_symbol_ownership_overrides import apply_overrides


PACKAGE_SECTIONS = {
    "payne_zero_atmosphere": ("## Atmosphere Package Modules", "### Atmosphere Branch Boundaries"),
    "payne_zero_synthesis": ("## Synthesis Package Modules", "### Synthesis Feature Matrix"),
}


def parse_locations(text: str) -> list[str]:
    """Expand a compact chapter/appendix cell into stable location identifiers."""

    locations: list[str] = []
    chapter_text = text.split("App.", maxsplit=1)[0]
    for start_text, end_text in re.findall(r"(\d+)(?:[–-](\d+))?", chapter_text):
        start = int(start_text)
        end = int(end_text) if end_text else start
        if end < start:
            raise ValueError(f"reversed chapter range in {text!r}")
        locations.extend(f"chapter-{chapter}" for chapter in range(start, end + 1))

    appendix_match = re.search(r"App\.\s*([A-Z](?:/[A-Z])*)", text)
    if appendix_match:
        locations.extend(
            f"appendix-{letter.lower()}" for letter in appendix_match.group(1).split("/")
        )
    if not locations:
        raise ValueError(f"no chapter or appendix location in {text!r}")
    return locations


def parse_module_rows(markdown: str, package: str) -> dict[str, dict[str, Any]]:
    """Read one package table from ``COVERAGE.md``."""

    start_marker, end_marker = PACKAGE_SECTIONS[package]
    try:
        section = markdown.split(start_marker, maxsplit=1)[1].split(end_marker, maxsplit=1)[0]
    except IndexError as error:
        raise SystemExit(f"coverage section markers not found for {package}") from error

    rows: dict[str, dict[str, Any]] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 5:
            raise SystemExit(f"unexpected coverage row: {line}")
        module = columns[0].strip("`")
        if module.endswith(".py"):
            module = module[:-3]
        locations = parse_locations(columns[2])
        rows[module] = {
            "responsibility": columns[1],
            "primary_location": locations[0],
            "supporting_locations": locations[1:],
            "gate": columns[3],
            "status": columns[4],
        }
    return rows


def symbol_record(
    *,
    qualified_name: str,
    kind: str,
    line: int | None,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """Create one inherited symbol-coverage record."""

    return {
        "qualified_name": qualified_name,
        "kind": kind,
        "line": line,
        "mapping_precision": "module_default",
        **mapping,
    }


def expand_module_symbols(
    package: str,
    module: dict[str, Any],
    mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand one inventory module into exhaustive public-object records."""

    prefix = f"{package}.{module['module']}"
    records: list[dict[str, Any]] = []

    for name in module["static_all_exports"]:
        records.append(
            symbol_record(
                qualified_name=f"{prefix}.{name}",
                kind="public_export",
                line=None,
                mapping=mapping,
            )
        )
    for datum in module["public_module_data"]:
        if datum["name"] == "__all__":
            continue
        records.append(
            symbol_record(
                qualified_name=f"{prefix}.{datum['name']}",
                kind=datum["kind"],
                line=datum["line"],
                mapping=mapping,
            )
        )
    for function in module["public_functions"]:
        records.append(
            symbol_record(
                qualified_name=f"{prefix}.{function['name']}",
                kind="public_function",
                line=function["line"],
                mapping=mapping,
            )
        )
    for class_item in module["public_classes"]:
        class_name = f"{prefix}.{class_item['name']}"
        records.append(
            symbol_record(
                qualified_name=class_name,
                kind="public_class",
                line=class_item["line"],
                mapping=mapping,
            )
        )
        if class_item["constructor"] is not None:
            records.append(
                symbol_record(
                    qualified_name=f"{class_name}.__init__",
                    kind="constructor",
                    line=class_item["constructor"]["line"],
                    mapping=mapping,
                )
            )
        for field in class_item["fields"]:
            records.append(
                symbol_record(
                    qualified_name=f"{class_name}.{field['name']}",
                    kind="annotated_field",
                    line=field["line"],
                    mapping=mapping,
                )
            )
        for method in class_item["public_methods"]:
            records.append(
                symbol_record(
                    qualified_name=f"{class_name}.{method['name']}",
                    kind="public_method",
                    line=method["line"],
                    mapping=mapping,
                )
            )
    return records


def build_coverage(
    inventory: dict[str, Any],
    coverage_markdown: str,
) -> dict[str, Any]:
    """Build symbol coverage and fail on a module-set mismatch."""

    packages: dict[str, Any] = {}
    for package, payload in inventory["packages"].items():
        table_rows = parse_module_rows(coverage_markdown, package)
        inventory_names = {module["module"] for module in payload["modules"]}
        mapped_names = set(table_rows)
        if inventory_names != mapped_names:
            missing = sorted(inventory_names - mapped_names)
            extra = sorted(mapped_names - inventory_names)
            raise SystemExit(
                f"{package} module coverage mismatch; missing={missing}, extra={extra}"
            )

        module_records: list[dict[str, Any]] = []
        symbol_records: list[dict[str, Any]] = []
        for module in payload["modules"]:
            mapping = table_rows[module["module"]]
            module_records.append({"module": module["module"], **mapping})
            symbol_records.extend(expand_module_symbols(package, module, mapping))

        packages[package] = {
            "module_count": len(module_records),
            "symbol_count": len(symbol_records),
            "modules": module_records,
            "symbols": symbol_records,
        }

    return {
        "schema_version": 1,
        "purpose": "Exhaustive public-source placement and verification ledger",
        "payne_zero_commit": inventory["payne_zero_commit"],
        "mapping_policy": (
            "Every public object inherits a complete module assignment until a "
            "narrower chapter-owned override is reviewed."
        ),
        "packages": packages,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("audit/paynezero_symbols.json"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("COVERAGE.md"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/paynezero_symbol_coverage.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the exhaustive public-symbol placement ledger."""

    arguments = parse_args()
    inventory = json.loads(arguments.inventory.read_text(encoding="utf-8"))
    coverage = build_coverage(
        inventory,
        arguments.ledger.read_text(encoding="utf-8"),
    )
    override_count = apply_overrides(coverage)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    module_count = sum(item["module_count"] for item in coverage["packages"].values())
    symbol_count = sum(item["symbol_count"] for item in coverage["packages"].values())
    status_counts = Counter(
        symbol["status"]
        for package in coverage["packages"].values()
        for symbol in package["symbols"]
    )
    status_summary = ", ".join(
        f"{status}={status_counts[status]}"
        for status in (
            "integrated",
            "verified",
            "implemented",
            "planned",
            "boundary",
        )
    )
    print(
        f"wrote {arguments.output}: {module_count} mapped modules, "
        f"{symbol_count} mapped public objects, "
        f"{override_count} reviewed overrides; {status_summary}"
    )


if __name__ == "__main__":
    main()
