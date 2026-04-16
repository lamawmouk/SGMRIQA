"""Structured output schema for model responses.

Defines a unified JSON output format that all models are instructed to produce,
plus parsing utilities with fallback for legacy free-text responses.
"""

import json
import re
from typing import List, Optional

from pydantic import BaseModel


class BBox(BaseModel):
    """A single bounding box annotation."""

    frame: int  # frame/slice number
    label: str  # finding description
    bbox: List[float]  # [x, y, w, h] pixels or [min_x, min_y, max_x, max_y] normalized
    normalized_01: bool = False  # True if bbox is Gemini normalized [0,1] format


class ModelResponse(BaseModel):
    """Unified structured output from any model."""

    reasoning: str  # chain-of-thought reasoning
    answer: str  # Yes/No, (A), number, or free text
    bboxes: List[BBox]  # grounding boxes (empty list when N/A)


def format_schema_instruction() -> str:
    """Return the JSON schema text for embedding in system prompts."""
    return (
        'You MUST respond with a JSON object in this exact format:\n'
        '{\n'
        '  "reasoning": "<your step-by-step clinical reasoning>",\n'
        '  "answer": "<your final answer>",\n'
        '  "bboxes": [{"frame": 1, "label": "finding", "bbox": [x, y, w, h]}, ...]\n'
        '}\n'
        'Bounding box coordinates are [x, y, width, height] in pixels (top-left corner x, y and box width, height). '
        'Do NOT use [x1, y1, x2, y2] format. '
        'Frame numbers correspond to the frame labels (Frame 1, Frame 2, etc.). '
        'For a single image, use frame 1.\n'
        'If no bounding boxes are applicable, set "bboxes" to an empty list [].'
    )


def parse_model_response(raw: str) -> Optional[ModelResponse]:
    """Parse structured JSON from model output with fallbacks.

    Tries in order:
        1. Direct json.loads(raw)
        2. Extract from ```json ... ``` fences
        3. Regex find first {...} with all required keys
        4. Return None if unparseable (caller falls back to legacy)

    Args:
        raw: The raw model output string.

    Returns:
        ModelResponse if successfully parsed, None otherwise.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # 1. Direct parse
    parsed = _try_parse_json(raw)
    if parsed:
        return parsed

    # 2. Extract from ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if fence_match:
        parsed = _try_parse_json(fence_match.group(1).strip())
        if parsed:
            return parsed

    # 3. Regex find first {...} that contains all required keys
    # Use a greedy match to find the outermost braces
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        parsed = _try_parse_json(candidate)
        if parsed:
            return parsed

    return None


def _try_parse_json(text: str) -> Optional[ModelResponse]:
    """Attempt to parse text as a ModelResponse JSON object."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    # Check required keys
    if "reasoning" not in data or "answer" not in data:
        return None

    # Default bboxes to empty list if missing
    if "bboxes" not in data:
        data["bboxes"] = []

    # Handle Gemini normalized bbox format: {min_x, min_y, max_x, max_y} floats
    # Convert to standard bbox list but keep as floats (denormalized later in eval)
    converted_bboxes = []
    for bb in data["bboxes"]:
        if isinstance(bb, dict) and "min_x" in bb:
            # Gemini normalized format → store as [min_x, min_y, max_x, max_y]
            # with a flag so evaluation knows to denormalize
            converted_bboxes.append({
                "frame": bb.get("frame", 1),
                "label": bb.get("label", ""),
                "bbox": [bb["min_x"], bb["min_y"], bb["max_x"], bb["max_y"]],
                "normalized_01": True,
            })
        else:
            converted_bboxes.append(bb)
    data["bboxes"] = converted_bboxes

    try:
        return ModelResponse(**data)
    except Exception:
        return None
