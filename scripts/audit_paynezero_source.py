#!/usr/bin/env python3
"""Create a deterministic public-surface inventory for a Payne Zero checkout.

The source checkout is read only. This script parses Python with ``ast`` and hashes
files; it never imports Payne Zero and never writes inside the source tree.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PACKAGES = ("payne_zero_atmosphere", "payne_zero_synthesis")
DEFAULT_EXPECTED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"


@dataclass(frozen=True)
class InventoryOptions:
    source_root: Path
    output_path: Path
    expected_commit: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_doc_line(node: ast.AST) -> str:
    """Return the first non-empty docstring line for an AST node."""

    document = ast.get_docstring(node, clean=True) or ""
    return next((line.strip() for line in document.splitlines() if line.strip()), "")


def expression_text(node: ast.AST) -> str:
    """Return stable source-like text for a decorator or annotation."""

    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def literal_string_sequence(node: ast.AST) -> list[str] | None:
    """Read a literal list/tuple of strings without evaluating source code."""

    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def static_all_exports(tree: ast.Module) -> list[str]:
    """Return a statically declared ``__all__`` list, when present."""

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        values = literal_string_sequence(node.value) if node.value is not None else None
        if values is not None:
            return values
    return []


def function_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Describe one function or method."""

    positional = [argument.arg for argument in node.args.posonlyargs + node.args.args]
    keyword_only = [argument.arg for argument in node.args.kwonlyargs]
    return {
        "name": node.name,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "parameters": positional,
        "keyword_only_parameters": keyword_only,
        "has_varargs": node.args.vararg is not None,
        "has_varkw": node.args.kwarg is not None,
        "decorators": [expression_text(decorator) for decorator in node.decorator_list],
        "summary": first_doc_line(node),
    }


def assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """Return simple names written by one module-level assignment."""

    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def public_assignment_records(tree: ast.Module) -> list[dict[str, Any]]:
    """Inventory public module-level data names without evaluating their values."""

    records: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        annotation = (
            expression_text(node.annotation)
            if isinstance(node, ast.AnnAssign) and node.annotation is not None
            else ""
        )
        for name in assignment_names(node):
            if name.startswith("_"):
                continue
            records.append(
                {
                    "name": name,
                    "line": node.lineno,
                    "annotation": annotation,
                    "kind": "constant" if name.isupper() else "module_data",
                }
            )
    return records


def class_record(node: ast.ClassDef) -> dict[str, Any]:
    """Describe one public class, its public methods, and annotated fields."""

    fields: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    constructor: dict[str, Any] | None = None
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            fields.append(
                {
                    "name": child.target.id,
                    "annotation": expression_text(child.annotation),
                    "line": child.lineno,
                }
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name == "__init__":
                constructor = function_record(child)
                continue
            if child.name.startswith("_") and child.name not in {"__call__", "__iter__"}:
                continue
            methods.append(function_record(child))
    return {
        "name": node.name,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "bases": [expression_text(base) for base in node.bases],
        "decorators": [expression_text(decorator) for decorator in node.decorator_list],
        "summary": first_doc_line(node),
        "fields": fields,
        "constructor": constructor,
        "public_methods": methods,
    }


def local_imports(tree: ast.Module, package: str) -> list[str]:
    """Return same-package module dependencies found in import statements."""

    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module:
                    dependencies.add(node.module.split(".")[0])
                else:
                    dependencies.update(alias.name.split(".")[0] for alias in node.names)
            elif node.module and node.module.startswith(f"{package}."):
                dependencies.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{package}."):
                    dependencies.add(alias.name.split(".")[1])
    return sorted(dependencies)


def module_record(path: Path, package: str) -> dict[str, Any]:
    """Parse one package module into a deterministic inventory record."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = [
        function_record(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    classes = [
        class_record(node)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]
    return {
        "module": path.stem,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": sha256_file(path),
        "line_count": source.count("\n") + (0 if source.endswith("\n") else 1),
        "summary": first_doc_line(tree),
        "local_imports": local_imports(tree, package),
        "static_all_exports": static_all_exports(tree),
        "public_module_data": public_assignment_records(tree),
        "public_functions": functions,
        "public_classes": classes,
    }


def git_commit(source_root: Path) -> str:
    """Read the checkout commit without invoking filters or changing the tree."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_inventory(options: InventoryOptions) -> dict[str, Any]:
    """Build the complete atmosphere/synthesis source inventory."""

    actual_commit = git_commit(options.source_root)
    if actual_commit != options.expected_commit:
        raise SystemExit(
            "Payne Zero source commit mismatch: "
            f"expected {options.expected_commit}, found {actual_commit}"
        )

    packages: dict[str, Any] = {}
    for package in PACKAGES:
        package_dir = options.source_root / package
        if not package_dir.is_dir():
            raise SystemExit(f"missing package directory: {package_dir}")
        modules = [
            module_record(path, package)
            for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name)
        ]
        packages[package] = {
            "module_count": len(modules),
            "modules": modules,
        }

    return {
        "schema_version": 1,
        "purpose": "Pinned public-surface and dependency inventory for textbook coverage",
        "source_root_role": "read-only Payne Zero checkout",
        "payne_zero_commit": actual_commit,
        "packages": packages,
    }


def parse_args() -> InventoryOptions:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/ysting/payne-zero"),
        help="read-only Payne Zero checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/paynezero_symbols.json"),
        help="generated inventory path inside the textbook repository",
    )
    parser.add_argument(
        "--expected-commit",
        default=DEFAULT_EXPECTED_COMMIT,
        help="refuse to inventory a different Payne Zero revision",
    )
    arguments = parser.parse_args()
    return InventoryOptions(
        source_root=arguments.source_root.expanduser().resolve(),
        output_path=arguments.output,
        expected_commit=arguments.expected_commit,
    )


def main() -> None:
    """Generate and write the deterministic JSON inventory."""

    options = parse_args()
    inventory = build_inventory(options)
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {options.output_path} for "
        f"{sum(item['module_count'] for item in inventory['packages'].values())} modules"
    )


if __name__ == "__main__":
    main()
