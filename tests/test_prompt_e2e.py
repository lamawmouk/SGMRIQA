"""End-to-end prompt builder test: sends real samples to gpt-4o-mini
and verifies the model returns valid structured JSON output.

Picks 1 sample per task (10 total: 5 image-level + 5 volume-level)
to keep cost minimal (~$0.01).

Run:
    python -m pytest tests/test_prompt_e2e
"""

import json
import sys
import os

# Ensure we can import evaluation package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgmriqa.data.loader import load_qa_samples
from sgmriqa.data.prompt_builder import build_system_prompt, build_user_prompt
from sgmriqa.schemas import parse_model_response
from sgmriqa.run_inference import load_runner, load_images_for_sample
from sgmriqa.config.model_configs import get_model_config

MODEL_KEY = "gpt-4o-mini"
# One sample per (level, task)
TARGET_COMBOS = [
    ("image_level", "detection"),
    ("image_level", "classification"),
    ("image_level", "localization"),
    ("image_level", "captioning"),
    ("image_level", "diagnosis"),
    ("volume_level", "detection"),
    ("volume_level", "counting"),
    ("volume_level", "classification"),
    ("volume_level", "localization"),
    ("volume_level", "captioning"),
]


def pick_samples():
    """Pick 1 sample per (level, task) combo from brain val data."""
    samples = load_qa_samples(datasets=["brain"], split="val")
    picked = {}
    for s in samples:
        key = (s.level, s.task)
        if key not in picked and key in TARGET_COMBOS:
            # Make sure sample has valid images
            if s.image_paths and os.path.exists(s.image_paths[0]):
                picked[key] = s
        if len(picked) == len(TARGET_COMBOS):
            break
    return picked


def main():
    print(f"Loading {MODEL_KEY}...")
    config = get_model_config(MODEL_KEY)
    runner = load_runner(MODEL_KEY)
    runner.load_model()

    picked = pick_samples()
    print(f"Found {len(picked)}/{len(TARGET_COMBOS)} combos\n")

    results = []
    total_pass = 0
    total_fail = 0

    for combo in TARGET_COMBOS:
        sample = picked.get(combo)
        if not sample:
            print(f"  SKIP  {combo[0]:15s} | {combo[1]:15s} | no sample found")
            continue

        level, task = combo

        # Build prompts
        system_prompt = build_system_prompt(sample, config)
        user_prompt = build_user_prompt(sample, config)

        # Load images
        images = load_images_for_sample(sample, MODEL_KEY)
        if not images:
            print(f"  SKIP  {level:15s} | {task:15s} | no images")
            continue

        # Run inference
        try:
            result = runner.run_inference(images, system_prompt, user_prompt)
            raw_output = result.model_output
        except Exception as e:
            print(f"  FAIL  {level:15s} | {task:15s} | API error: {e}")
            total_fail += 1
            continue

        # Parse output
        parsed = parse_model_response(raw_output)

        # Validate
        checks = []

        # Check 1: Parsed successfully
        if parsed is None:
            checks.append("PARSE_FAIL")
        else:
            # Check 2: Has reasoning
            if not parsed.reasoning or len(parsed.reasoning) < 5:
                checks.append("NO_REASONING")

            # Check 3: Has answer
            if not parsed.answer:
                checks.append("NO_ANSWER")

            # Check 4: Answer format matches task
            if task == "detection":
                ans_lower = parsed.answer.strip().lower().rstrip(".")
                if ans_lower not in ("yes", "no"):
                    checks.append(f"BAD_ANSWER(expected yes/no, got '{parsed.answer}')")

            elif task == "counting":
                ans_clean = parsed.answer.strip().rstrip(".")
                if not any(c.isdigit() for c in ans_clean):
                    checks.append(f"BAD_ANSWER(expected number, got '{parsed.answer}')")

            elif task in ("classification", "diagnosis"):
                if "(" not in parsed.answer:
                    checks.append(f"BAD_ANSWER(expected letter, got '{parsed.answer}')")

            # Check 5: bboxes for localization/captioning
            if task in ("localization", "captioning"):
                if not parsed.bboxes:
                    checks.append("NO_BBOXES")
                else:
                    for bb in parsed.bboxes:
                        if len(bb.bbox) != 4:
                            checks.append(f"BAD_BBOX({bb.bbox})")
                            break

            # Check 6: bboxes have valid frame numbers for volume
            if level == "volume_level" and parsed.bboxes:
                for bb in parsed.bboxes:
                    if bb.frame < 1:
                        checks.append(f"BAD_FRAME({bb.frame})")
                        break

        passed = len(checks) == 0

        if passed:
            total_pass += 1
            status = "PASS"
        else:
            total_fail += 1
            status = "FAIL"

        print(f"  {status:4s}  {level:15s} | {task:15s} | cost=${result.cost:.4f}", end="")
        if checks:
            print(f" | issues: {', '.join(checks)}", end="")
        print()

        # Show condensed output
        if parsed:
            ans_short = parsed.answer[:80] + ("..." if len(parsed.answer) > 80 else "")
            reas_short = parsed.reasoning[:80] + ("..." if len(parsed.reasoning) > 80 else "")
            n_bbox = len(parsed.bboxes)
            print(f"         answer:    {ans_short}")
            print(f"         reasoning: {reas_short}")
            print(f"         bboxes:    {n_bbox} boxes")
        else:
            print(f"         raw: {raw_output[:150]}")
        print()

        results.append({
            "combo": combo,
            "passed": passed,
            "checks": checks,
            "raw_output": raw_output,
        })

    runner.unload_model()

    # Summary
    print("=" * 70)
    print(f"RESULTS: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail}")
    if total_fail == 0:
        print("All tests passed! JSON output format is correct.")
    else:
        print("Some tests failed. Check output above.")

    return total_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
