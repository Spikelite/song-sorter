"""Guard against README drift: every interactive menu option must be documented.

This parses the `options` list out of main.py (without importing it, so no heavy
deps are needed) and checks each one appears in the README's menu reference. It
exists because the menu reference is hand-written and easy to forget to update.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _menu_options() -> list[str]:
    """Extract the `options = [...]` string list from main.py via AST."""
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "options" for t in node.targets
        ):
            if isinstance(node.value, ast.List):
                vals = [
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if vals:
                    return vals
    raise AssertionError("could not find the `options` menu list in main.py")


def test_every_menu_option_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # README labels follow Choice(o.capitalize(), ...): e.g. "tag-swap" -> "Tag-swap".
    missing = [o for o in _menu_options() if f"**{o.capitalize()}**" not in readme]
    assert not missing, f"Menu options missing from README.md menu reference: {missing}"
