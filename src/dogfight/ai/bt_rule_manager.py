from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
from typing import NoReturn


RULE_XML_NAME = "Rule_forTraining.xml"


def _exit_with_rule_xml_error(message: str) -> NoReturn:
    print(f"[bt_rule_manager] {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_rule_xml_source(
    rule_xml_path: str | Path | None,
    workspace_root: Path,
) -> Path:
    default_source = (workspace_root / RULE_XML_NAME).resolve()

    if not rule_xml_path:
        if default_source.exists():
            return default_source
        _exit_with_rule_xml_error(
            f"Rule XML path is empty and fallback does not exist: {default_source}"
        )

    source = Path(rule_xml_path)
    if not source.is_absolute():
        source = workspace_root / source
    source = source.resolve()

    if source.is_dir():
        source = (source / RULE_XML_NAME).resolve()

    if source.suffix.lower() == ".xml" and source.exists():
        return source

    if default_source.exists():
        print(
            "[bt_rule_manager] "
            f"Rule XML not found or not an .xml file: {source}. "
            f"Using fallback: {default_source}",
            file=sys.stderr,
        )
        return default_source

    _exit_with_rule_xml_error(
        "Rule XML not found and fallback is unavailable. "
        f"requested={source}, fallback={default_source}"
    )


@contextmanager
def activate_rule_xml(
    rule_xml_path: str | Path | None,
    workspace_root: str | Path,
) -> Iterator[None]:
    """Temporarily activate a BT rule XML as the workspace rule file."""
    workspace_root = Path(workspace_root).resolve()
    source = _resolve_rule_xml_source(rule_xml_path, workspace_root)

    target = workspace_root / RULE_XML_NAME
    if source == target.resolve():
        # 2026-05-26: Log the exact XML path consumed by the native BT DLL.
        print(
            f"[bt_rule_manager] active Rule XML already in place: {target}",
            file=sys.stderr,
        )
        yield
        return

    backup = None
    if target.exists():
        backup = target.with_suffix(".xml.bak")
        shutil.copy2(target, backup)

    shutil.copy2(source, target)
    # 2026-05-26: Make XML activation explicit for DLL/cwd mismatch diagnosis.
    print(f"[bt_rule_manager] activated Rule XML: {source} -> {target}", file=sys.stderr)
    try:
        yield
    finally:
        if backup and backup.exists():
            shutil.copy2(backup, target)
            backup.unlink()
