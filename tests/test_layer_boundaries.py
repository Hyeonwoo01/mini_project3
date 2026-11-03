"""계층 분리 검증 (가이드 7 구조 점검 목록).

각 계층이 금지된 계층·라이브러리를 import 하지 않는지 정적으로 확인한다.
- 챗봇(app)과 체인(rag)은 indexer·collector 없이 구동되어야 한다
- 수집기는 차트·UI 코드를 포함하지 않는다
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LAYERS = ("common", "collector", "indexer", "rag", "eval", "app")
UI_LIBS = {"streamlit", "plotly"}

FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "common": {"collector", "indexer", "rag", "eval", "app"} | UI_LIBS,
    "collector": {"indexer", "rag", "eval", "app"} | UI_LIBS,
    "indexer": {"collector", "rag", "eval", "app"} | UI_LIBS,
    "rag": {"collector", "indexer", "eval", "app", "streamlit", "bs4"},
    "eval": {"collector", "indexer", "app", "streamlit"},
    "app": {"collector", "indexer", "eval", "bs4"},
}


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_does_not_import_forbidden_modules(layer: str) -> None:
    violations = []
    for path in sorted((PROJECT_ROOT / layer).rglob("*.py")):
        bad = _imported_top_level_modules(path) & FORBIDDEN_IMPORTS[layer]
        if bad:
            violations.append(f"{path.relative_to(PROJECT_ROOT)}: {sorted(bad)}")
    assert not violations, "계층 분리 위반:\n" + "\n".join(violations)
