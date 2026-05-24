from pathlib import Path


THEMES = [
    "Commissioning",
    "Common_documents",
    "Cycles",
    "Functional_description",
    "Interfaces",
    "Manuals",
    "PLC-libraries",
]


_CANONICAL_BY_NORMALIZED = {theme.lower(): theme for theme in THEMES}

_ALIASES = {
    "common documents": "Common_documents",
    "common_documents": "Common_documents",
    "functional description": "Functional_description",
    "functional_description": "Functional_description",
    "plc libraries": "PLC-libraries",
    "plc-libraries": "PLC-libraries",
    "plc_libraries": "PLC-libraries",
}

_INFERENCE_PRIORITY = [
    "Commissioning",
    "Common_documents",
    "Cycles",
    "Functional_description",
    "Interfaces",
    "PLC-libraries",
    "Manuals",
]


def normalize_theme(theme: str | None) -> str:
    """Return a canonical theme value when possible, else 'Unknown'."""
    if theme is None:
        return "Unknown"

    raw = str(theme).strip()
    if not raw:
        return "Unknown"

    if raw in THEMES:
        return raw

    normalized = raw.lower().replace("-", " ").replace("_", " ").strip()
    normalized = " ".join(normalized.split())

    if normalized in _ALIASES:
        return _ALIASES[normalized]

    normalized_underscore = normalized.replace(" ", "_")
    if normalized_underscore in _CANONICAL_BY_NORMALIZED:
        return _CANONICAL_BY_NORMALIZED[normalized_underscore]

    normalized_dash = normalized.replace(" ", "-")
    if normalized_dash in _CANONICAL_BY_NORMALIZED:
        return _CANONICAL_BY_NORMALIZED[normalized_dash]

    return "Unknown"


def infer_theme_from_path(file_path: str) -> str:
    """Infer theme from folder names or file name tokens."""
    path = Path(file_path)
    normalized_parts = [part.lower().replace("-", " ").replace("_", " ") for part in path.parts]
    joined_path = " ".join(normalized_parts)
    normalized_name = path.name.lower().replace(".", " ").replace("-", " ").replace("_", " ")

    for theme in _INFERENCE_PRIORITY:
        aliases = [
            alias
            for alias, canonical in _ALIASES.items()
            if canonical == theme
        ]
        aliases.append(theme.lower().replace("_", " ").replace("-", " "))

        for alias in aliases:
            token = " ".join(alias.split())
            if token and (token in joined_path or token in normalized_name):
                return theme

    return "Unknown"