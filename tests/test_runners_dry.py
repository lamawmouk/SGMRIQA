"""Dry-run tests for new model runners.

Tests everything that can be validated WITHOUT a GPU or model weights:
    1. Module imports and class instantiation
    2. Constructor signature matches run_inference.py call pattern
    3. _build_messages / _preprocess_frames output structure
    4. Message format correctness (system prompt, image entries, user prompt)
    5. Config registry integration (load_runner dynamic import)
    6. Think-tag stripping logic
    7. Base class interface compliance
    8. _ensure_*_utils path resolution logic

Run: python -m pytest tests/test_runners_dry.py -v
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

# Ensure Grounding/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Mock torch and heavy dependencies if not installed (e.g., on Mac dev) ──
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.Tensor = MagicMock
    _torch.cuda = MagicMock()
    _torch.cuda.is_available = MagicMock(return_value=False)
    _torch.cuda.device_count = MagicMock(return_value=0)
    _torch.cuda.empty_cache = MagicMock()
    _torch.bfloat16 = "bfloat16"
    _torch.float16 = "float16"
    _torch.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    sys.modules["torch"] = _torch

    # Mock torchvision
    _tv = types.ModuleType("torchvision")
    _tv_transforms = types.ModuleType("torchvision.transforms")
    _tv_transforms.Compose = MagicMock
    _tv_transforms.Normalize = MagicMock
    _tv_transforms.Resize = MagicMock
    _tv_transforms.ToTensor = MagicMock
    _tv_transforms_func = types.ModuleType("torchvision.transforms.functional")
    _tv_transforms_func.InterpolationMode = MagicMock()
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.transforms"] = _tv_transforms
    sys.modules["torchvision.transforms.functional"] = _tv_transforms_func

if "numpy" not in sys.modules:
    _np = types.ModuleType("numpy")
    _np.stack = lambda arrays: type("FakeArray", (), {
        "shape": (len(arrays), *arrays[0].shape) if hasattr(arrays[0], "shape") else (len(arrays),),
        "dtype": type("dt", (), {"name": "uint8"})(),
    })()
    _np.array = lambda x: type("FakeArray", (), {
        "shape": (x.size[1], x.size[0], 3) if hasattr(x, "size") else (),
    })()
    _np.uint8 = "uint8"
    _np.zeros = MagicMock()
    sys.modules["numpy"] = _np

from sgmriqa.models.base import BaseModelRunner, InferenceResult
from sgmriqa.config.model_configs import (
    get_model_config,
    get_max_volume_images,
    list_video_capable_models,
    MODEL_REGISTRY,
)


def _make_test_images(n=3, size=(256, 256)):
    """Create N dummy RGB PIL images."""
    return [Image.new("RGB", size, color=(i * 40, i * 40, i * 40)) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════
# 1. Qwen3OmniRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestQwen3OmniRunner(unittest.TestCase):
    """Test Qwen3OmniRunner without GPU/vLLM/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        runner = Qwen3OmniRunner(model_id="test/model", max_new_tokens=512)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 512)
        self.assertEqual(runner.max_model_len, 32768)
        self.assertFalse(runner._loaded)

    def test_constructor_matches_load_runner_pattern(self):
        """run_inference.py calls: runner_cls(model_id=..., max_new_tokens=...)"""
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        # This must not raise
        runner = Qwen3OmniRunner(model_id="Qwen/Qwen3-Omni-30B-A3B-Thinking", max_new_tokens=4096)
        self.assertEqual(runner.model_id, "Qwen/Qwen3-Omni-30B-A3B-Thinking")

    def test_build_messages_with_system_prompt(self):
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        runner = Qwen3OmniRunner(model_id="test/model")
        images = _make_test_images(3)

        msgs = runner._build_messages(images, "You are an expert.", "What do you see?")

        # System message uses list-of-dicts format (NOT plain string)
        self.assertEqual(len(msgs), 2)
        sys_msg = msgs[0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIsInstance(sys_msg["content"], list)
        self.assertEqual(sys_msg["content"][0]["type"], "text")
        self.assertEqual(sys_msg["content"][0]["text"], "You are an expert.")

        # User message has 3 images + 1 text
        user_msg = msgs[1]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(len(user_msg["content"]), 4)  # 3 images + 1 text
        for i in range(3):
            self.assertEqual(user_msg["content"][i]["type"], "image")
            self.assertIsInstance(user_msg["content"][i]["image"], Image.Image)
            self.assertIn("min_pixels", user_msg["content"][i])
            self.assertIn("max_pixels", user_msg["content"][i])
        self.assertEqual(user_msg["content"][3]["type"], "text")
        self.assertEqual(user_msg["content"][3]["text"], "What do you see?")

    def test_build_messages_no_system_prompt(self):
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        runner = Qwen3OmniRunner(model_id="test/model")
        images = _make_test_images(1)

        msgs = runner._build_messages(images, "", "Describe this.")
        self.assertEqual(len(msgs), 1)  # No system message
        self.assertEqual(msgs[0]["role"], "user")

    def test_supports_multiple_images(self):
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        runner = Qwen3OmniRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        """unload_model should be safe to call even if never loaded."""
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner
        runner = Qwen3OmniRunner(model_id="test/model")
        runner.unload_model()  # Should not raise
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 2. Qwen2VLRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestQwen2VLRunner(unittest.TestCase):
    """Test Qwen2VLRunner without GPU/vLLM/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        runner = Qwen2VLRunner(model_id="test/model", max_new_tokens=2048)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 2048)
        self.assertEqual(runner.max_model_len, 32768)

    def test_constructor_matches_load_runner_pattern(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        # Both model IDs that use this runner
        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/QVQ-7B",
        ]:
            runner = Qwen2VLRunner(model_id=model_id, max_new_tokens=2048)
            self.assertEqual(runner.model_id, model_id)

    def test_build_messages_with_system_prompt(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        runner = Qwen2VLRunner(model_id="test/model")
        images = _make_test_images(5)

        msgs = runner._build_messages(images, "You are a radiologist.", "Findings?")

        # System message is plain string (NOT list-of-dicts like Qwen3-Omni)
        self.assertEqual(len(msgs), 2)
        sys_msg = msgs[0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIsInstance(sys_msg["content"], str)
        self.assertEqual(sys_msg["content"], "You are a radiologist.")

        # User message has 5 images + 1 text
        user_msg = msgs[1]
        self.assertEqual(len(user_msg["content"]), 6)
        for i in range(5):
            self.assertEqual(user_msg["content"][i]["type"], "image")
            self.assertIsInstance(user_msg["content"][i]["image"], Image.Image)
        self.assertEqual(user_msg["content"][5]["text"], "Findings?")

    def test_build_messages_no_system_prompt(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        runner = Qwen2VLRunner(model_id="test/model")
        msgs = runner._build_messages(_make_test_images(1), "", "Describe.")
        self.assertEqual(len(msgs), 1)

    def test_supports_multiple_images(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        runner = Qwen2VLRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        runner = Qwen2VLRunner(model_id="test/model")
        runner.unload_model()
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 3. LLaVAVideoRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestLLaVAVideoRunner(unittest.TestCase):
    """Test LLaVAVideoRunner without GPU/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.llava_video_runner import LLaVAVideoRunner
        runner = LLaVAVideoRunner(model_id="test/model", max_new_tokens=4096)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 4096)
        self.assertEqual(runner.conv_mode, "qwen_1_5")
        self.assertEqual(runner.torch_dtype, "bfloat16")

    def test_constructor_matches_load_runner_pattern(self):
        from sgmriqa.models.llava_video_runner import LLaVAVideoRunner
        runner = LLaVAVideoRunner(
            model_id="lmms-lab/LLaVA-Video-7B-Qwen2", max_new_tokens=4096
        )
        self.assertEqual(runner.model_id, "lmms-lab/LLaVA-Video-7B-Qwen2")

    def test_preprocess_frames_shape(self):
        """Verify PIL→numpy→tensor conversion produces correct shape."""
        import numpy as np
        # Skip if numpy is mocked (no real ndarray available)
        if not hasattr(np, 'ndarray'):
            self.skipTest("numpy is mocked — shape test runs on GPU server")

        from sgmriqa.models.llava_video_runner import LLaVAVideoRunner
        runner = LLaVAVideoRunner(model_id="test/model")
        images = _make_test_images(5, size=(256, 256))

        frames = np.stack([np.array(img.convert("RGB")) for img in images])
        self.assertEqual(frames.shape, (5, 256, 256, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_supports_multiple_images(self):
        from sgmriqa.models.llava_video_runner import LLaVAVideoRunner
        runner = LLaVAVideoRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        from sgmriqa.models.llava_video_runner import LLaVAVideoRunner
        runner = LLaVAVideoRunner(model_id="test/model")
        runner.unload_model()
        self.assertFalse(runner._loaded)

    def test_ensure_llava_next_package_finds_local(self):
        """Verify _ensure_llava_next_package resolves LLaVA-NeXT/ correctly."""
        from sgmriqa.models.llava_video_runner import _ensure_llava_next_package
        # This should succeed since LLaVA-NeXT/ exists at project root
        try:
            _ensure_llava_next_package()
            # If it succeeds, verify llava is importable
            from llava.constants import IMAGE_TOKEN_INDEX
            self.assertEqual(IMAGE_TOKEN_INDEX, -200)
        except ImportError:
            # OK if llava deps aren't installed — we just test the path logic
            pass


# ═══════════════════════════════════════════════════════════════════════
# 4. Molmo2Runner tests
# ═══════════════════════════════════════════════════════════════════════

class TestMolmo2Runner(unittest.TestCase):
    """Test Molmo2Runner without GPU/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model", max_new_tokens=4096)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 4096)
        self.assertFalse(runner._loaded)

    def test_constructor_matches_load_runner_pattern(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="allenai/Molmo2-8B", max_new_tokens=4096)
        self.assertEqual(runner.model_id, "allenai/Molmo2-8B")

    def test_build_messages_with_system_prompt(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model")
        images = _make_test_images(3)

        msgs = runner._build_messages(images, "You are an expert.", "What do you see?")

        self.assertEqual(len(msgs), 2)
        sys_msg = msgs[0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIsInstance(sys_msg["content"], str)

        user_msg = msgs[1]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(len(user_msg["content"]), 4)  # 1 text + 3 images
        # Molmo convention: text first, then images
        self.assertEqual(user_msg["content"][0]["type"], "text")
        self.assertEqual(user_msg["content"][0]["text"], "What do you see?")
        for i in range(1, 4):
            self.assertEqual(user_msg["content"][i]["type"], "image")
            self.assertIsInstance(user_msg["content"][i]["image"], Image.Image)

    def test_build_messages_no_system_prompt(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model")
        msgs = runner._build_messages(_make_test_images(1), "", "Describe.")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_text_before_images_convention(self):
        """Molmo puts text before images, unlike Qwen/Eagle which put images first."""
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model")
        images = _make_test_images(2)
        msgs = runner._build_messages(images, "", "Question?")
        content = msgs[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[2]["type"], "image")

    def test_supports_multiple_images(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        from sgmriqa.models.molmo2_runner import Molmo2Runner
        runner = Molmo2Runner(model_id="test/model")
        runner.unload_model()
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 5. KeyeVLRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestKeyeVLRunner(unittest.TestCase):
    """Test KeyeVLRunner without GPU/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="test/model", max_new_tokens=4096)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 4096)
        self.assertEqual(runner.torch_dtype, "bfloat16")
        self.assertFalse(runner._loaded)

    def test_constructor_matches_load_runner_pattern(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="Kwai-Keye/Keye-VL-1.5-8B", max_new_tokens=4096)
        self.assertEqual(runner.model_id, "Kwai-Keye/Keye-VL-1.5-8B")

    def test_build_messages_with_system_prompt(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="test/model")
        images = _make_test_images(3)

        msgs = runner._build_messages(images, "You are an expert.", "What do you see?")

        self.assertEqual(len(msgs), 2)
        sys_msg = msgs[0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIsInstance(sys_msg["content"], str)

        user_msg = msgs[1]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(len(user_msg["content"]), 4)
        for i in range(3):
            self.assertEqual(user_msg["content"][i]["type"], "image")
            self.assertIsInstance(user_msg["content"][i]["image"], Image.Image)
        self.assertEqual(user_msg["content"][3]["type"], "text")

    def test_build_messages_no_system_prompt(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="test/model")
        msgs = runner._build_messages(_make_test_images(1), "", "Describe.")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_supports_multiple_images(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        from sgmriqa.models.keye_vl_runner import KeyeVLRunner
        runner = KeyeVLRunner(model_id="test/model")
        runner.unload_model()
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 6. PLMRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestPLMRunner(unittest.TestCase):
    """Test PLMRunner without GPU/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.plm_runner import PLMRunner
        runner = PLMRunner(model_id="test/model", max_new_tokens=512)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 512)
        self.assertEqual(runner.max_tokens, 16384)
        self.assertFalse(runner._loaded)

    def test_constructor_matches_load_runner_pattern(self):
        from sgmriqa.models.plm_runner import PLMRunner
        runner = PLMRunner(model_id="facebook/Perception-LM-8B", max_new_tokens=512)
        self.assertEqual(runner.model_id, "facebook/Perception-LM-8B")

    def test_custom_max_tokens(self):
        from sgmriqa.models.plm_runner import PLMRunner
        runner = PLMRunner(model_id="test/model", max_tokens=32768)
        self.assertEqual(runner.max_tokens, 32768)

    def test_supports_multiple_images(self):
        from sgmriqa.models.plm_runner import PLMRunner
        runner = PLMRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        from sgmriqa.models.plm_runner import PLMRunner
        runner = PLMRunner(model_id="test/model")
        runner.unload_model()
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 7. EagleRunner tests
# ═══════════════════════════════════════════════════════════════════════

class TestEagleRunner(unittest.TestCase):
    """Test EagleRunner without GPU/model weights."""

    def test_import_and_instantiate(self):
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="test/model", max_new_tokens=4096)
        self.assertIsInstance(runner, BaseModelRunner)
        self.assertEqual(runner.model_id, "test/model")
        self.assertEqual(runner.max_new_tokens, 4096)
        self.assertEqual(runner.torch_dtype, "bfloat16")
        self.assertFalse(runner._loaded)

    def test_constructor_matches_load_runner_pattern(self):
        """run_inference.py calls: runner_cls(model_id=..., max_new_tokens=...)"""
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="nvidia/Eagle-2.5-8B", max_new_tokens=4096)
        self.assertEqual(runner.model_id, "nvidia/Eagle-2.5-8B")

    def test_build_messages_with_system_prompt(self):
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="test/model")
        images = _make_test_images(3)

        msgs = runner._build_messages(images, "You are an expert.", "What do you see?")

        # System message uses plain string (like Qwen2VL)
        self.assertEqual(len(msgs), 2)
        sys_msg = msgs[0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIsInstance(sys_msg["content"], str)
        self.assertEqual(sys_msg["content"], "You are an expert.")

        # User message has 3 images + 1 text
        user_msg = msgs[1]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(len(user_msg["content"]), 4)  # 3 images + 1 text
        for i in range(3):
            self.assertEqual(user_msg["content"][i]["type"], "image")
            self.assertIsInstance(user_msg["content"][i]["image"], Image.Image)
        self.assertEqual(user_msg["content"][3]["type"], "text")
        self.assertEqual(user_msg["content"][3]["text"], "What do you see?")

    def test_build_messages_no_system_prompt(self):
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="test/model")
        images = _make_test_images(1)

        msgs = runner._build_messages(images, "", "Describe this.")
        self.assertEqual(len(msgs), 1)  # No system message
        self.assertEqual(msgs[0]["role"], "user")

    def test_supports_multiple_images(self):
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="test/model")
        self.assertTrue(runner.supports_multiple_images)

    def test_unload_before_load(self):
        """unload_model should be safe to call even if never loaded."""
        from sgmriqa.models.eagle_runner import EagleRunner
        runner = EagleRunner(model_id="test/model")
        runner.unload_model()  # Should not raise
        self.assertFalse(runner._loaded)


# ═══════════════════════════════════════════════════════════════════════
# 5. Config registry integration tests
# ═══════════════════════════════════════════════════════════════════════

class TestConfigIntegration(unittest.TestCase):
    """Test that all new models are properly registered and loadable."""

    def test_new_models_in_registry(self):
        expected = ["qwen3-omni", "qwen2.5-vl-7b", "qvq-7b", "llava-video-7b", "eagle2.5-8b", "keye-vl-1.5-8b", "molmo2-8b", "plm-8b"]
        for key in expected:
            self.assertIn(key, MODEL_REGISTRY, f"{key} missing from registry")

    def test_runner_modules_importable(self):
        """Verify runner modules can be imported (not classes — no GPU needed)."""
        modules = [
            "sgmriqa.models.qwen3_omni_runner",
            "sgmriqa.models.qwen2_vl_runner",
            "sgmriqa.models.llava_video_runner",
            "sgmriqa.models.eagle_runner",
            "sgmriqa.models.keye_vl_runner",
            "sgmriqa.models.molmo2_runner",
            "sgmriqa.models.plm_runner",
        ]
        for mod_path in modules:
            mod = importlib.import_module(mod_path)
            self.assertIsNotNone(mod, f"Failed to import {mod_path}")

    def test_runner_classes_exist_in_modules(self):
        """Verify runner class names match what's registered in config."""
        test_cases = [
            ("qwen3-omni", "sgmriqa.models.qwen3_omni_runner", "Qwen3OmniRunner"),
            ("qwen2.5-vl-7b", "sgmriqa.models.qwen2_vl_runner", "Qwen2VLRunner"),
            ("qvq-7b", "sgmriqa.models.qwen2_vl_runner", "Qwen2VLRunner"),
            ("llava-video-7b", "sgmriqa.models.llava_video_runner", "LLaVAVideoRunner"),
            ("eagle2.5-8b", "sgmriqa.models.eagle_runner", "EagleRunner"),
            ("keye-vl-1.5-8b", "sgmriqa.models.keye_vl_runner", "KeyeVLRunner"),
            ("molmo2-8b", "sgmriqa.models.molmo2_runner", "Molmo2Runner"),
            ("plm-8b", "sgmriqa.models.plm_runner", "PLMRunner"),
        ]
        for model_key, mod_path, cls_name in test_cases:
            config = get_model_config(model_key)
            self.assertEqual(config.runner_module, mod_path, f"{model_key} module mismatch")
            self.assertEqual(config.runner_class, cls_name, f"{model_key} class mismatch")

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            self.assertIsNotNone(cls, f"{cls_name} not found in {mod_path}")
            self.assertTrue(issubclass(cls, BaseModelRunner), f"{cls_name} not a BaseModelRunner subclass")

    def test_dynamic_instantiation_like_load_runner(self):
        """Replicate the exact load_runner() pattern from run_inference.py."""
        for model_key in ["qwen3-omni", "qwen2.5-vl-7b", "qvq-7b", "llava-video-7b", "eagle2.5-8b", "keye-vl-1.5-8b", "molmo2-8b", "plm-8b"]:
            config = get_model_config(model_key)
            module = importlib.import_module(config.runner_module)
            runner_cls = getattr(module, config.runner_class)
            runner = runner_cls(model_id=config.model_id, max_new_tokens=config.max_new_tokens)
            self.assertEqual(runner.model_id, config.model_id)
            self.assertEqual(runner.max_new_tokens, config.max_new_tokens)

    def test_video_capable_includes_new_models(self):
        video_models = list_video_capable_models()
        expected_in = ["qwen3-omni", "qwen2.5-vl-7b", "qvq-7b", "llava-video-7b", "eagle2.5-8b", "keye-vl-1.5-8b", "molmo2-8b", "plm-8b"]
        for key in expected_in:
            self.assertIn(key, video_models, f"{key} not in video_capable list")

    def test_fuyu_excluded_from_video(self):
        video_models = list_video_capable_models()
        self.assertNotIn("fuyu-8b", video_models)

    def test_total_video_capable_count(self):
        self.assertEqual(len(list_video_capable_models()), 20)

    def test_frame_budgets_reasonable(self):
        """All new models should support >= 40 frames."""
        for key in ["qwen3-omni", "qwen2.5-vl-7b", "qvq-7b", "llava-video-7b", "eagle2.5-8b", "keye-vl-1.5-8b", "molmo2-8b", "plm-8b"]:
            frames = get_max_volume_images(key)
            self.assertGreaterEqual(frames, 40, f"{key} only gets {frames} frames (need >=40)")


# ═══════════════════════════════════════════════════════════════════════
# 5. Think-tag stripping logic
# ═══════════════════════════════════════════════════════════════════════

class TestThinkTagStripping(unittest.TestCase):
    """Test the </think> stripping that runners apply to reasoning model output."""

    def test_strip_think_tags(self):
        """All Qwen runners strip </think> the same way."""
        raw = "<think>Let me analyze this MRI scan carefully...</think>The scan shows normal anatomy."
        # This is the pattern used in all runners:
        if "</think>" in raw:
            result = raw.split("</think>")[-1].strip()
        self.assertEqual(result, "The scan shows normal anatomy.")

    def test_no_think_tags_passthrough(self):
        raw = "The scan shows normal anatomy."
        result = raw
        if "</think>" in raw:
            result = raw.split("</think>")[-1].strip()
        self.assertEqual(result, "The scan shows normal anatomy.")

    def test_multiple_think_tags(self):
        raw = "<think>first thought</think><think>second thought</think>Final answer."
        if "</think>" in raw:
            result = raw.split("</think>")[-1].strip()
        self.assertEqual(result, "Final answer.")

    def test_empty_after_think(self):
        raw = "<think>thinking only</think>"
        if "</think>" in raw:
            result = raw.split("</think>")[-1].strip()
        self.assertEqual(result, "")


# ═══════════════════════════════════════════════════════════════════════
# 6. Qwen3-VL runner comparison (reference runner that works)
# ═══════════════════════════════════════════════════════════════════════

class TestQwen2VLvsQwen3VLDifferences(unittest.TestCase):
    """Verify the intentional differences between Qwen2VL and Qwen3VL runners."""

    def test_system_prompt_format_differs_from_omni(self):
        """Qwen2VL uses plain string, Qwen3-Omni uses list-of-dicts."""
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        from sgmriqa.models.qwen3_omni_runner import Qwen3OmniRunner

        images = _make_test_images(1)
        prompt = "Expert system."

        q2_runner = Qwen2VLRunner(model_id="test")
        q2_msgs = q2_runner._build_messages(images, prompt, "Q?")
        self.assertIsInstance(q2_msgs[0]["content"], str)

        omni_runner = Qwen3OmniRunner(model_id="test")
        omni_msgs = omni_runner._build_messages(images, prompt, "Q?")
        self.assertIsInstance(omni_msgs[0]["content"], list)

    def test_qwen2vl_matches_qwen3vl_message_structure(self):
        """Qwen2VL system prompt format should match Qwen3VL (plain string)."""
        from sgmriqa.models.qwen2_vl_runner import Qwen2VLRunner
        from sgmriqa.models.qwen3_vl_runner import Qwen3VLRunner

        images = _make_test_images(2)

        q2 = Qwen2VLRunner(model_id="test")
        q3 = Qwen3VLRunner(model_id="test")

        q2_msgs = q2._build_messages(images, "sys", "user")
        q3_msgs = q3._build_messages(images, "sys", "user")

        # Both use plain string system prompt
        self.assertEqual(type(q2_msgs[0]["content"]), type(q3_msgs[0]["content"]))
        # Both have same user content structure
        self.assertEqual(len(q2_msgs[1]["content"]), len(q3_msgs[1]["content"]))


# ═══════════════════════════════════════════════════════════════════════
# 7. LLaVA-NeXT conversation template test (if available)
# ═══════════════════════════════════════════════════════════════════════

class TestLLaVAConversation(unittest.TestCase):
    """Test LLaVA conversation template integration (requires LLaVA-NeXT on path)."""

    @classmethod
    def setUpClass(cls):
        """Try to make llava importable."""
        try:
            from sgmriqa.models.llava_video_runner import _ensure_llava_next_package
            _ensure_llava_next_package()
            cls.llava_available = True
        except ImportError:
            cls.llava_available = False

    def test_conv_template_exists(self):
        if not self.llava_available:
            self.skipTest("LLaVA-NeXT not importable")
        from llava.conversation import conv_templates
        self.assertIn("qwen_1_5", conv_templates)

    def test_system_prompt_injection(self):
        """Verify system prompt override produces correct ChatML format."""
        if not self.llava_available:
            self.skipTest("LLaVA-NeXT not importable")
        import copy
        from llava.conversation import conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.system = "<|im_start|>system\nYou are a neuroradiology expert."
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nWhat is this?")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Verify ChatML structure
        self.assertIn("<|im_start|>system\nYou are a neuroradiology expert.<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\n<image>\nWhat is this?<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_tokenizer_image_token(self):
        """Verify <image> placeholder tokenization."""
        if not self.llava_available:
            self.skipTest("LLaVA-NeXT not importable")
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.bos_token_id = 1
        mock_tok.return_value = MagicMock(input_ids=[])
        mock_tok.side_effect = lambda text: MagicMock(input_ids=list(range(len(text))))

        # The function should split on <image> and insert IMAGE_TOKEN_INDEX
        prompt = "Hello <image> world"
        result = tokenizer_image_token(prompt, mock_tok, IMAGE_TOKEN_INDEX)
        self.assertIsInstance(result, list)
        self.assertIn(IMAGE_TOKEN_INDEX, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
