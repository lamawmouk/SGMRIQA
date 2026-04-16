"""Test cases for prompt builder against real val QA samples.

Verifies that system + user prompts contain the correct instructions for each
task/type/level/dataset combination.

Run:
    python -m pytest evaluation/tests/test_prompt_builder.py -v
    python -m pytest tests/test_prompt_builder  # standalone
"""

import json
import unittest

from sgmriqa.data.loader import EvalSample, load_qa_samples
from sgmriqa.data.prompt_builder import build_system_prompt, build_user_prompt


def _get_samples_by_combo():
    """Load val samples and index one per (task, qa_type, level, dataset) combo."""
    samples = load_qa_samples(split="val")
    by_combo = {}
    for s in samples:
        key = (s.task, s.qa_type, s.level, s.dataset)
        if key not in by_combo:
            by_combo[key] = s
    return by_combo


def _full_prompt(sample):
    """Concatenate system + user prompt for combined assertions."""
    return build_system_prompt(sample) + "\n" + build_user_prompt(sample)


SAMPLES = _get_samples_by_combo()


# ═══════════════════════════════════════════════════════════════════════════
# 1. System context
# ═══════════════════════════════════════════════════════════════════════════


class TestSystemContext(unittest.TestCase):
    """System prompt must contain expert role, research context, and JSON schema."""

    def test_brain_has_neuroradiologist(self):
        for key, s in SAMPLES.items():
            if s.dataset == "brain":
                sys = build_system_prompt(s)
                self.assertIn("neuroradiologist", sys, f"Missing for {key}")

    def test_knee_has_musculoskeletal(self):
        for key, s in SAMPLES.items():
            if s.dataset == "knee":
                sys = build_system_prompt(s)
                self.assertIn("musculoskeletal radiologist", sys, f"Missing for {key}")

    def test_research_context(self):
        for key, s in SAMPLES.items():
            sys = build_system_prompt(s)
            self.assertIn("research study", sys, f"Missing research context for {key}")

    def test_json_schema_present(self):
        """System prompt must include the JSON output schema contract."""
        for key, s in SAMPLES.items():
            sys = build_system_prompt(s)
            self.assertIn('"reasoning"', sys, f"Missing reasoning key in schema for {key}")
            self.assertIn('"answer"', sys, f"Missing answer key in schema for {key}")
            self.assertIn('"bboxes"', sys, f"Missing bboxes key in schema for {key}")

    def test_json_schema_has_format(self):
        """Schema must describe bbox format [x, y, w, h]."""
        s = list(SAMPLES.values())[0]
        sys = build_system_prompt(s)
        self.assertIn("x, y, width, height", sys)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Image / frame context
# ═══════════════════════════════════════════════════════════════════════════


class TestImageContext(unittest.TestCase):
    """Image-level gets single image context, volume-level gets frame context."""

    def test_image_level_single_image(self):
        for key, s in SAMPLES.items():
            if s.level == "image_level":
                sys = build_system_prompt(s)
                self.assertIn("MRI image", sys, f"Missing for {key}")

    def test_volume_level_frame_context(self):
        for key, s in SAMPLES.items():
            if s.level == "volume_level":
                sys = build_system_prompt(s)
                self.assertIn("frames", sys.lower(), f"Missing frame context for {key}")
                self.assertIn("Frame 1", sys, f"Missing Frame 1 for {key}")

    def test_volume_has_bbox_frame_instruction(self):
        for key, s in SAMPLES.items():
            if s.level == "volume_level":
                sys = build_system_prompt(s)
                self.assertIn("which frame", sys.lower(), f"Missing frame bbox instruction for {key}")

    def test_image_has_frame1_instruction(self):
        for key, s in SAMPLES.items():
            if s.level == "image_level":
                sys = build_system_prompt(s)
                self.assertIn("frame 1", sys.lower(), f"Missing frame 1 instruction for {key}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ICL examples (JSON format)
# ═══════════════════════════════════════════════════════════════════════════


class TestICLExamples(unittest.TestCase):
    """User prompt must include a JSON-formatted ICL example for each task."""

    def test_icl_example_is_valid_json(self):
        """The ICL example output in the user prompt must be valid JSON."""
        for key, s in SAMPLES.items():
            user = build_user_prompt(s)
            if "Your response:" in user:
                # Extract JSON between "Your response:" and "Now answer"
                start = user.index("Your response:") + len("Your response:")
                end = user.index("Now answer the following:")
                json_str = user[start:end].strip()
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    self.fail(f"ICL example is not valid JSON for {key}: {json_str[:100]}")
                self.assertIn("reasoning", parsed, f"Missing reasoning in ICL for {key}")
                self.assertIn("answer", parsed, f"Missing answer in ICL for {key}")
                self.assertIn("bboxes", parsed, f"Missing bboxes in ICL for {key}")

    def test_icl_example_present_for_all_tasks(self):
        """Every task should have an ICL example."""
        for key, s in SAMPLES.items():
            user = build_user_prompt(s)
            self.assertIn("Example:", user, f"Missing ICL example for {key}")
            self.assertIn("Your response:", user, f"Missing JSON output in ICL for {key}")

    def test_localization_icl_has_bboxes(self):
        """Localization ICL example must show non-empty bboxes."""
        for key, s in SAMPLES.items():
            if s.task == "localization":
                user = build_user_prompt(s)
                start = user.index("Your response:") + len("Your response:")
                end = user.index("Now answer the following:")
                parsed = json.loads(user[start:end].strip())
                self.assertTrue(len(parsed["bboxes"]) > 0, f"No bboxes in localization ICL for {key}")

    def test_detection_icl_has_empty_bboxes(self):
        """Detection ICL example must show empty bboxes."""
        for key, s in SAMPLES.items():
            if s.task == "detection":
                user = build_user_prompt(s)
                start = user.index("Your response:") + len("Your response:")
                end = user.index("Now answer the following:")
                parsed = json.loads(user[start:end].strip())
                self.assertEqual(parsed["bboxes"], [], f"Detection ICL should have empty bboxes for {key}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Detection task (closed_ended yes/no)
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectionPrompt(unittest.TestCase):
    """Detection tasks must instruct Yes/No answer."""

    def test_brain_image_detection(self):
        s = SAMPLES[("detection", "closed_ended", "image_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("Yes or No", user)

    def test_knee_image_detection(self):
        s = SAMPLES[("detection", "closed_ended", "image_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("Yes or No", user)

    def test_brain_volume_detection(self):
        s = SAMPLES[("detection", "closed_ended", "volume_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("Yes or No", user)

    def test_knee_volume_detection(self):
        s = SAMPLES[("detection", "closed_ended", "volume_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("Yes or No", user)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Counting task (closed_ended numeric)
# ═══════════════════════════════════════════════════════════════════════════


class TestCountingPrompt(unittest.TestCase):
    """Counting tasks must NOT say Yes/No — must ask for a number."""

    def test_brain_volume_counting(self):
        s = SAMPLES[("counting", "closed_ended", "volume_level", "brain")]
        user = build_user_prompt(s)
        self.assertNotIn("Yes or No", user)
        self.assertIn("number", user.lower())

    def test_knee_volume_counting(self):
        s = SAMPLES[("counting", "closed_ended", "volume_level", "knee")]
        user = build_user_prompt(s)
        self.assertNotIn("Yes or No", user)
        self.assertIn("number", user.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 6. Localization task (bbox instructions)
# ═══════════════════════════════════════════════════════════════════════════


class TestLocalizationPrompt(unittest.TestCase):
    """Localization tasks must include bbox instructions, NOT Yes/No."""

    def test_brain_image_localization_has_bbox(self):
        s = SAMPLES[("localization", "closed_ended", "image_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("bounding box", user.lower())
        self.assertNotIn("Yes or No", user)

    def test_knee_image_localization_has_bbox(self):
        s = SAMPLES[("localization", "closed_ended", "image_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("bounding box", user.lower())
        self.assertNotIn("Yes or No", user)

    def test_brain_volume_localization_has_bbox(self):
        s = SAMPLES[("localization", "closed_ended", "volume_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("bounding box", user.lower())
        self.assertNotIn("Yes or No", user)

    def test_knee_volume_localization_has_bbox(self):
        s = SAMPLES[("localization", "open_ended", "volume_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("bounding box", user.lower())

    def test_system_prompt_localization_has_bbox_schema(self):
        """System prompt for localization must have bbox format in JSON schema."""
        s = SAMPLES[("localization", "closed_ended", "image_level", "brain")]
        sys = build_system_prompt(s)
        self.assertIn("bounding box", sys.lower())
        self.assertIn("frame", sys.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 7. Classification / Diagnosis (single_choice / multiple_choice)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCQPrompt(unittest.TestCase):
    """MCQ tasks must include inline options and letter instruction."""

    def test_brain_image_classification(self):
        s = SAMPLES[("classification", "single_choice", "image_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("(A)", user)
        self.assertIn("letter", user.lower())

    def test_knee_image_classification(self):
        s = SAMPLES[("classification", "single_choice", "image_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("(A)", user)

    def test_brain_image_diagnosis(self):
        s = SAMPLES[("diagnosis", "single_choice", "image_level", "brain")]
        user = build_user_prompt(s)
        self.assertIn("(A)", user)
        self.assertIn("letter", user.lower())

    def test_knee_volume_multiple_choice(self):
        s = SAMPLES[("classification", "multiple_choice", "volume_level", "knee")]
        user = build_user_prompt(s)
        self.assertIn("(A)", user)
        self.assertIn("Select all", user)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Captioning (open_ended)
# ═══════════════════════════════════════════════════════════════════════════


class TestCaptioningPrompt(unittest.TestCase):
    """Captioning prompts should ask for summary and bboxes."""

    def test_brain_image_captioning(self):
        s = SAMPLES[("captioning", "open_ended", "image_level", "brain")]
        user = build_user_prompt(s)
        self.assertNotIn("Yes or No", user)
        self.assertIn("summary", user.lower())
        self.assertIn("bounding box", user.lower())

    def test_knee_volume_captioning(self):
        s = SAMPLES[("captioning", "open_ended", "volume_level", "knee")]
        user = build_user_prompt(s)
        self.assertNotIn("Yes or No", user)
        self.assertIn("summary", user.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 9. Question text is included
# ═══════════════════════════════════════════════════════════════════════════


class TestQuestionIncluded(unittest.TestCase):
    """Every user prompt must include the original question text."""

    def test_all_prompts_include_question(self):
        for key, s in SAMPLES.items():
            user = build_user_prompt(s)
            stem = s.question.split("\n")[0]
            self.assertIn(stem, user, f"Question missing in prompt for {key}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. No contradictory instructions
# ═══════════════════════════════════════════════════════════════════════════


class TestNoContradictions(unittest.TestCase):
    """Ensure user prompts don't have conflicting instructions."""

    def test_captioning_no_letter_instruction(self):
        """Captioning should not ask for letter response."""
        for key, s in SAMPLES.items():
            if s.task == "captioning":
                user = build_user_prompt(s)
                self.assertNotIn("Respond with the letter", user, f"Letter in captioning: {key}")

    def test_counting_no_yes_no(self):
        """Counting should never say Yes or No."""
        for key, s in SAMPLES.items():
            if s.task == "counting":
                user = build_user_prompt(s)
                self.assertNotIn("Yes or No", user, f"Yes/No in counting: {key}")

    def test_localization_no_yes_no(self):
        """Localization should never say Yes or No."""
        for key, s in SAMPLES.items():
            if s.task == "localization":
                user = build_user_prompt(s)
                self.assertNotIn("Yes or No", user, f"Yes/No in localization: {key}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. JSON output instruction
# ═══════════════════════════════════════════════════════════════════════════


class TestJSONInstruction(unittest.TestCase):
    """Every prompt must instruct JSON-only output."""

    def test_user_prompt_says_json_only(self):
        for key, s in SAMPLES.items():
            user = build_user_prompt(s)
            self.assertIn("JSON", user, f"Missing JSON instruction for {key}")

    def test_system_prompt_says_must_respond_json(self):
        for key, s in SAMPLES.items():
            sys = build_system_prompt(s)
            self.assertIn("MUST respond with a JSON", sys, f"Missing MUST JSON for {key}")


if __name__ == "__main__":
    unittest.main()
