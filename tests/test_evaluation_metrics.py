"""Test cases for all evaluation metrics and parsing logic.

Verifies that each of the 5 QA tasks (detection, counting, classification,
localization, captioning) and the grounding task are scored correctly across
different model output formats and edge cases.

Run:
    python -m pytest evaluation/tests/test_evaluation_metrics.py -v
    python -m pytest tests/test_evaluation_metrics  # standalone
"""

import unittest

from sgmriqa.metrics.a_score import compute_a_score
from sgmriqa.metrics.utils import parse_bboxes, parse_choice_letters, parse_yes_no
from sgmriqa.metrics.v_score import compute_v_score
from sgmriqa.run_evaluation import _extract_video_answer


# ═══════════════════════════════════════════════════════════════════════════
# 1. Detection (closed_ended, yes/no)
# ═══════════════════════════════════════════════════════════════════════════


class TestDetection(unittest.TestCase):
    """Detection task: closed_ended yes/no extracted from step-by-step output."""

    def test_correct_yes(self):
        output = (
            "Looking at the MRI volume, I observe several abnormalities.\n"
            "Therefore, the final answer is: **Yes**"
        )
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, "Yes")
        result = compute_a_score(extracted, "Yes", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)

    def test_correct_no(self):
        output = "The volume appears normal. Therefore, the final answer is: No"
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, "No")
        result = compute_a_score(extracted, "No", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)

    def test_wrong_answer(self):
        output = "Therefore, the final answer is: **No**"
        extracted = _extract_video_answer(output, "closed_ended")
        result = compute_a_score(extracted, "Yes", "closed_ended")
        self.assertEqual(result["a_score"], 0.0)

    def test_verbose_yes_extraction(self):
        """Model says 'Yes, there is evidence' — should extract just 'Yes'."""
        output = (
            "Therefore, the final answer is: Yes, there is evidence of abnormalities"
        )
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, "Yes")
        result = compute_a_score(extracted, "Yes", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)

    def test_bold_markdown_stripped(self):
        """Markdown bold markers (**Yes**) should be stripped."""
        output = "Therefore, the final answer is: **Yes**"
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, "Yes")

    def test_no_pattern_match_fallback(self):
        """When 'final answer is:' pattern is missing, returns full text."""
        output = "Yes, there are abnormalities."
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, output)
        result = compute_a_score(extracted, "Yes", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Counting (closed_ended, numeric)
# ═══════════════════════════════════════════════════════════════════════════


class TestCounting(unittest.TestCase):
    """Counting task: closed_ended with numeric answer extraction."""

    def test_correct_count(self):
        output = (
            "I can identify 4 types of injuries.\n"
            "Therefore, the final answer is: **4**"
        )
        extracted = _extract_video_answer(output, "closed_ended")
        self.assertEqual(extracted, "4")
        result = compute_a_score(extracted, "4", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)
        self.assertEqual(result["details"]["method"], "closed_ended_numeric")

    def test_wrong_count(self):
        output = "Therefore, the final answer is: 3"
        extracted = _extract_video_answer(output, "closed_ended")
        result = compute_a_score(extracted, "4", "closed_ended")
        self.assertEqual(result["a_score"], 0.0)

    def test_count_zero(self):
        output = "Therefore, the final answer is: 0"
        extracted = _extract_video_answer(output, "closed_ended")
        result = compute_a_score(extracted, "0", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)

    def test_number_in_verbose_output(self):
        """Model says 'There are 4 distinct injuries' — should extract 4."""
        output = "Therefore, the final answer is: There are 4 distinct injuries"
        extracted = _extract_video_answer(output, "closed_ended")
        result = compute_a_score(extracted, "4", "closed_ended")
        self.assertEqual(result["a_score"], 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Classification (single_choice and multiple_choice)
# ═══════════════════════════════════════════════════════════════════════════


class TestClassification(unittest.TestCase):
    """Classification: MCQ letter extraction and F1 scoring."""

    def test_single_choice_correct(self):
        output = "Therefore, the final answer is: B"
        extracted = _extract_video_answer(output, "single_choice")
        self.assertEqual(extracted, "B")
        result = compute_a_score(extracted, "(B)", "single_choice")
        self.assertEqual(result["a_score"], 1.0)

    def test_single_choice_wrong(self):
        output = "Therefore, the final answer is: C"
        extracted = _extract_video_answer(output, "single_choice")
        result = compute_a_score(extracted, "(B)", "single_choice")
        self.assertEqual(result["a_score"], 0.0)

    def test_multiple_choice_all_correct(self):
        output = "Therefore, the final answer is: (A), (B), (C), (D)"
        extracted = _extract_video_answer(output, "multiple_choice")
        result = compute_a_score(extracted, "(A), (B), (C), (D)", "multiple_choice")
        self.assertEqual(result["a_score"], 1.0)

    def test_multiple_choice_partial(self):
        """Only selects 2 of 4 correct options: precision=1, recall=0.5, F1=0.667."""
        output = "Therefore, the final answer is: (A), (B)"
        extracted = _extract_video_answer(output, "multiple_choice")
        result = compute_a_score(extracted, "(A), (B), (C), (D)", "multiple_choice")
        self.assertAlmostEqual(result["a_score"], 2 / 3, places=2)

    def test_multiple_choice_overclaim(self):
        """Selects 4 but only 2 are correct: precision=0.5, recall=1, F1=0.667."""
        output = "Therefore, the final answer is: (A), (B), (C), (D)"
        extracted = _extract_video_answer(output, "multiple_choice")
        result = compute_a_score(extracted, "(A), (B)", "multiple_choice")
        self.assertAlmostEqual(result["a_score"], 2 / 3, places=2)

    def test_multiple_choice_single_answer(self):
        """Model outputs one letter for multi-choice."""
        output = "Therefore, the final answer is: A"
        extracted = _extract_video_answer(output, "multiple_choice")
        result = compute_a_score(
            extracted, "(A), (B), (C), (D)", "multiple_choice"
        )
        self.assertAlmostEqual(result["a_score"], 0.4, places=1)

    def test_multiple_choice_no_match(self):
        """Completely wrong letters."""
        output = "Therefore, the final answer is: (C), (D)"
        extracted = _extract_video_answer(output, "multiple_choice")
        result = compute_a_score(extracted, "(A), (B)", "multiple_choice")
        self.assertEqual(result["a_score"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Localization (open_ended, keyword recall)
# ═══════════════════════════════════════════════════════════════════════════


class TestLocalization(unittest.TestCase):
    """Localization QA: keyword recall + semantic similarity for A-Score."""

    def test_all_keywords_mentioned(self):
        gt_labels = ["Meniscus Tear", "Cartilage - Partial Thickness loss/defect"]
        prediction = (
            "The Meniscus Tear is in the medial meniscus. "
            "Cartilage - Partial Thickness loss/defect is in the medial compartment."
        )
        result = compute_a_score(
            prediction, "reference text", "open_ended", gt_labels=gt_labels
        )
        self.assertEqual(result["details"]["keyword_recall"], 1.0)

    def test_partial_keywords(self):
        gt_labels = ["Meniscus Tear", "Cartilage - Partial Thickness loss/defect"]
        prediction = "There is a Meniscus Tear in the medial meniscus."
        result = compute_a_score(
            prediction, "reference text", "open_ended", gt_labels=gt_labels
        )
        self.assertEqual(result["details"]["keyword_recall"], 0.5)

    def test_no_keywords(self):
        gt_labels = ["Meniscus Tear", "Cartilage defect"]
        prediction = "The volume appears completely normal."
        result = compute_a_score(
            prediction, "reference text", "open_ended", gt_labels=gt_labels
        )
        self.assertEqual(result["details"]["keyword_recall"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Captioning (open_ended, keyword recall)
# ═══════════════════════════════════════════════════════════════════════════


class TestCaptioning(unittest.TestCase):
    """Captioning: same scoring as localization (keyword recall + semantic sim)."""

    def test_good_caption(self):
        gt_labels = [
            "Cartilage - Partial Thickness loss/defect",
            "Meniscus Tear",
            "Periarticular cysts",
        ]
        prediction = (
            "The volume shows Cartilage - Partial Thickness loss/defect, "
            "Meniscus Tear, and Periarticular cysts."
        )
        result = compute_a_score(
            prediction, "reference", "open_ended", gt_labels=gt_labels
        )
        self.assertEqual(result["details"]["keyword_recall"], 1.0)
        self.assertEqual(result["details"]["matched_keywords"], 3)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Grounding (V-Score: bbox parsing + IoU)
# ═══════════════════════════════════════════════════════════════════════════


class TestBBoxParsing(unittest.TestCase):
    """Test bbox parsing across all supported model output formats."""

    def test_bbx_tags(self):
        """Format: <bbx>[x, y, w, h]</bbx> (GPT-4o, Gemini default)."""
        text = "<bbx>[210, 210, 10, 17]</bbx> and <bbx>[176, 247, 10, 11]</bbx>"
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])
        self.assertEqual(bboxes[1], [176.0, 247.0, 10.0, 11.0])

    def test_qwen_format(self):
        """Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|> — raw (no image dims)."""
        text = "<|box_start|>(210,210),(220,227)<|box_end|>"
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])

    def test_qwen_format_denormalize_brain(self):
        """Qwen 0-1000 normalized coords denormalized to brain 256x256."""
        # Qwen outputs (820, 820),(860, 886) in 0-1000 space
        # → pixel: x1=0.82*256=209.92, y1=209.92, x2=0.86*256=220.16, y2=0.886*256=226.816
        text = "<|box_start|>(820,820),(860,886)<|box_end|>"
        bboxes = parse_bboxes(text, image_width=256, image_height=256)
        self.assertEqual(len(bboxes), 1)
        x, y, w, h = bboxes[0]
        self.assertAlmostEqual(x, 209.92, places=1)
        self.assertAlmostEqual(y, 209.92, places=1)
        self.assertAlmostEqual(w, 10.24, places=1)   # (860-820)/1000*256
        self.assertAlmostEqual(h, 16.896, places=1)   # (886-820)/1000*256

    def test_qwen_format_denormalize_knee(self):
        """Qwen 0-1000 normalized coords denormalized to knee 320x320."""
        text = "<|box_start|>(500,500),(600,700)<|box_end|>"
        bboxes = parse_bboxes(text, image_width=320, image_height=320)
        self.assertEqual(len(bboxes), 1)
        x, y, w, h = bboxes[0]
        self.assertAlmostEqual(x, 160.0, places=1)   # 500/1000*320
        self.assertAlmostEqual(y, 160.0, places=1)
        self.assertAlmostEqual(w, 32.0, places=1)    # (600-500)/1000*320
        self.assertAlmostEqual(h, 64.0, places=1)    # (700-500)/1000*320

    def test_qwen_format_multiple_denormalized(self):
        """Multiple Qwen boxes denormalized correctly."""
        text = (
            "<|box_start|>(100,200),(300,400)<|box_end|> "
            "<|box_start|>(600,700),(800,900)<|box_end|>"
        )
        bboxes = parse_bboxes(text, image_width=256, image_height=256)
        self.assertEqual(len(bboxes), 2)
        # Box 1: (100,200)→(300,400) in 0-1000 → pixel: (25.6,51.2) w=51.2 h=51.2
        self.assertAlmostEqual(bboxes[0][0], 25.6, places=1)
        self.assertAlmostEqual(bboxes[0][2], 51.2, places=1)
        # Box 2: (600,700)→(800,900) → pixel: (153.6,179.2) w=51.2 h=51.2
        self.assertAlmostEqual(bboxes[1][0], 153.6, places=1)

    def test_coordinate_pairs(self):
        """Format: (x1,y1),(x2,y2)."""
        text = "Finding at (210,210),(220,227)"
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])

    def test_json_objects(self):
        """Format: {"x": N, "y": N, "width": N, "height": N} (any key order)."""
        text = '{"x": 210, "y": 210, "width": 10, "height": 17}'
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])

    def test_json_reversed_key_order(self):
        """JSON with keys in non-standard order."""
        text = '{"height": 17, "width": 10, "y": 210, "x": 210}'
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])

    def test_json_with_extra_keys(self):
        """JSON with frame, label, and other extra keys."""
        text = '{"frame": 1, "label": "lesion", "x": 210, "y": 210, "width": 10, "height": 17}'
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)

    def test_json_array(self):
        """Full JSON array of bbox objects."""
        text = (
            '[{"frame": 1, "x": 210, "y": 210, "width": 10, "height": 17, "label": "a"}, '
            '{"frame": 1, "x": 176, "y": 247, "width": 10, "height": 11, "label": "b"}]'
        )
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 2)

    def test_plain_brackets(self):
        """Format: [x, y, w, h] (plain, no tags)."""
        text = "Box: [210, 210, 10, 17]"
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [210.0, 210.0, 10.0, 17.0])

    def test_no_bboxes(self):
        """No parseable bboxes in output."""
        text = "I cannot identify any abnormalities."
        bboxes = parse_bboxes(text)
        self.assertEqual(len(bboxes), 0)


class TestVScore(unittest.TestCase):
    """V-Score: IoU computation with Hungarian matching."""

    GT_BBOXES = [
        {"x": 210, "y": 210, "width": 10, "height": 17, "label": "lesion"},
        {"x": 176, "y": 247, "width": 10, "height": 11, "label": "lesion"},
    ]

    def test_perfect_match(self):
        text = "<bbx>[210, 210, 10, 17]</bbx> <bbx>[176, 247, 10, 11]</bbx>"
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertEqual(result["v_score"], 1.0)
        self.assertEqual(result["details"]["hits"], 2)

    def test_perfect_match_json(self):
        text = (
            '[{"x": 210, "y": 210, "width": 10, "height": 17}, '
            '{"x": 176, "y": 247, "width": 10, "height": 11}]'
        )
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertEqual(result["v_score"], 1.0)

    def test_partial_overlap(self):
        text = "<bbx>[212, 212, 12, 15]</bbx> <bbx>[178, 249, 8, 9]</bbx>"
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertGreater(result["v_score"], 0.0)
        self.assertLess(result["v_score"], 1.0)

    def test_no_predictions(self):
        text = "I cannot detect any abnormalities."
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertEqual(result["v_score"], 0.0)
        self.assertEqual(result["details"]["reason"], "no_pred_bboxes")

    def test_extra_predictions(self):
        """More predictions than GT — should still match correctly."""
        text = (
            "<bbx>[210, 210, 10, 17]</bbx> "
            "<bbx>[176, 247, 10, 11]</bbx> "
            "<bbx>[50, 50, 20, 20]</bbx>"
        )
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertEqual(result["v_score"], 1.0)

    def test_fewer_predictions(self):
        """Fewer predictions than GT — unmatched GT gets IoU=0."""
        text = "<bbx>[210, 210, 10, 17]</bbx>"
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertAlmostEqual(result["v_score"], 0.5, places=1)

    def test_completely_wrong_bbox(self):
        """Predicted bbox has zero overlap with GT."""
        text = "<bbx>[0, 0, 5, 5]</bbx> <bbx>[1, 1, 5, 5]</bbx>"
        result = compute_v_score(text, self.GT_BBOXES)
        self.assertEqual(result["v_score"], 0.0)

    def test_qwen_denormalized_match(self):
        """Qwen 0-1000 coords denormalized to pixel coords should match GT."""
        # GT is at pixel (210,210) w=10 h=17 in 256x256 image
        # In 0-1000 space: x1=210/256*1000≈820, y1≈820, x2=220/256*1000≈859, y2=227/256*1000≈887
        text = "<|box_start|>(820,820),(859,887)<|box_end|>"
        result = compute_v_score(
            text, self.GT_BBOXES[:1], image_width=256, image_height=256
        )
        # Should have reasonable IoU (not perfect due to rounding, but close)
        self.assertGreater(result["v_score"], 0.7)

    def test_qwen_without_image_dims_no_match(self):
        """Qwen 0-1000 coords without image dims stay raw — poor match vs pixel GT."""
        # Same Qwen output but without image dimensions → raw 820,820 vs GT 210,210
        text = "<|box_start|>(820,820),(859,887)<|box_end|>"
        result = compute_v_score(text, self.GT_BBOXES[:1])
        # Without denormalization, the raw 820,820 bbox won't overlap GT at 210,210
        self.assertEqual(result["v_score"], 0.0)

    def test_no_gt_bboxes(self):
        """No ground truth bboxes — skip scoring."""
        result = compute_v_score("some text", [])
        self.assertIsNone(result["v_score"])


class TestVScoreFrameAware(unittest.TestCase):
    """V-Score: Frame-aware matching for volume-level evaluation."""

    # GT bboxes on different frames
    GT_VOLUME_BBOXES = [
        {"x": 100, "y": 100, "width": 20, "height": 20, "label": "lesion", "frame": 3, "slice_num": 3},
        {"x": 150, "y": 150, "width": 30, "height": 30, "label": "lesion", "frame": 7, "slice_num": 7},
    ]

    def test_correct_frame_match(self):
        """Predictions on correct frames should match."""
        text = "Frame 3: <bbx>[100, 100, 20, 20]</bbx> Frame 7: <bbx>[150, 150, 30, 30]</bbx>"
        result = compute_v_score(text, self.GT_VOLUME_BBOXES)
        self.assertEqual(result["v_score"], 1.0)
        self.assertTrue(result["details"]["frame_aware"])

    def test_wrong_frame_no_match(self):
        """Correct bbox coords on wrong frame should NOT match."""
        # Both predictions are on frame 3, but GT has one on frame 3 and one on frame 7
        text = "Frame 3: <bbx>[100, 100, 20, 20]</bbx> Frame 3: <bbx>[150, 150, 30, 30]</bbx>"
        result = compute_v_score(text, self.GT_VOLUME_BBOXES)
        # First bbox matches frame 3 GT (IoU=1.0), second bbox on frame 3 can't match frame 7 GT
        self.assertAlmostEqual(result["v_score"], 0.5, places=1)

    def test_image_level_no_frame_field(self):
        """Image-level GT (no frame field) should use frame-agnostic matching."""
        gt = [
            {"x": 100, "y": 100, "width": 20, "height": 20, "label": "lesion"},
        ]
        text = "<bbx>[100, 100, 20, 20]</bbx>"
        result = compute_v_score(text, gt)
        self.assertEqual(result["v_score"], 1.0)
        self.assertFalse(result["details"]["frame_aware"])

    def test_no_frame_in_prediction_still_matches(self):
        """If model doesn't mention frames, bboxes can still match any GT frame."""
        text = "<bbx>[100, 100, 20, 20]</bbx> <bbx>[150, 150, 30, 30]</bbx>"
        result = compute_v_score(text, self.GT_VOLUME_BBOXES)
        # pred_frames are None → no frame constraint → spatial matching only
        self.assertEqual(result["v_score"], 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Video answer extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestVideoAnswerExtraction(unittest.TestCase):
    """Test _extract_video_answer for all qa_types."""

    def test_single_choice_letter(self):
        output = "After analysis... Therefore, the final answer is: B"
        self.assertEqual(_extract_video_answer(output, "single_choice"), "B")

    def test_single_choice_bold_letter(self):
        output = "Therefore, the final answer is: **A**"
        self.assertEqual(_extract_video_answer(output, "single_choice"), "A")

    def test_multiple_choice_letters(self):
        output = "Therefore, the final answer is: (A), (B), (C)"
        result = _extract_video_answer(output, "multiple_choice")
        # Should keep the full letter list
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)

    def test_closed_ended_yes(self):
        output = "Therefore, the final answer is: **Yes**"
        self.assertEqual(_extract_video_answer(output, "closed_ended"), "Yes")

    def test_closed_ended_no(self):
        output = "Therefore, the final answer is: No"
        self.assertEqual(_extract_video_answer(output, "closed_ended"), "No")

    def test_closed_ended_verbose(self):
        """'Yes, there is evidence...' should extract just 'Yes'."""
        output = "Therefore, the final answer is: Yes, there is evidence of injury"
        self.assertEqual(_extract_video_answer(output, "closed_ended"), "Yes")

    def test_open_ended_passthrough(self):
        """Open-ended should return the full answer after 'final answer is:'."""
        output = "Therefore, the final answer is: Multiple lesions in white matter"
        result = _extract_video_answer(output, "open_ended")
        self.assertEqual(result, "Multiple lesions in white matter")

    def test_no_pattern_returns_full(self):
        """When pattern not found, returns full text."""
        output = "The answer is yes."
        self.assertEqual(_extract_video_answer(output, "closed_ended"), output)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Utility functions
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilities(unittest.TestCase):
    """Test parse_yes_no and parse_choice_letters."""

    def test_parse_yes(self):
        self.assertTrue(parse_yes_no("Yes"))
        self.assertTrue(parse_yes_no("yes, there is"))
        self.assertTrue(parse_yes_no("YES"))

    def test_parse_no(self):
        self.assertFalse(parse_yes_no("No"))
        self.assertFalse(parse_yes_no("no findings"))
        self.assertFalse(parse_yes_no("NO"))

    def test_parse_ambiguous(self):
        # "yes and no" -> first word is "yes" -> True (not ambiguous)
        self.assertTrue(parse_yes_no("yes and no"))
        # Pure number -> no yes/no signal
        self.assertIsNone(parse_yes_no("4"))

    def test_parse_choice_letters(self):
        self.assertEqual(parse_choice_letters("(A)"), {"A"})
        self.assertEqual(parse_choice_letters("(A), (B), (C)"), {"A", "B", "C"})
        self.assertEqual(parse_choice_letters("(A), (D)"), {"A", "D"})

    def test_parse_choice_no_parens(self):
        result = parse_choice_letters("A")
        self.assertIn("A", result)


if __name__ == "__main__":
    unittest.main()
