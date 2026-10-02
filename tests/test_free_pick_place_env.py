"""Terminal evidence must remain inspectable after the named success term fires."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ENV_FILE = (Path(__file__).resolve().parents[1] / "src/vla_isaaclab/envs"
            / "free_pick_place/env.py")


class FreePickPlaceEnvironmentTests(unittest.TestCase):
    def test_terminal_step_does_not_reset_scene(self):
        class FakeBase:
            def __init__(self, cfg, **kwargs):
                self.reset_count = 0

            def _reset_idx(self, env_ids):
                self.reset_count += 1

            def step(self, action):
                self._reset_idx([0])
                return None, None, [True], [False], {}

        fake = ModuleType("isaaclab.envs")
        fake.ManagerBasedRLEnv = FakeBase
        spec = importlib.util.spec_from_file_location("free_pick_place_env_test", ENV_FILE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab.envs": fake}):
            spec.loader.exec_module(module)
        env = module.FreePickPlaceEnv(SimpleNamespace(scene=SimpleNamespace(num_envs=1)))
        env._reset_idx([0])
        env.step(None)
        self.assertEqual(env.reset_count, 1)
        with self.assertRaises(RuntimeError):
            env.step(None)
        env._reset_idx([0])
        env.step(None)
        self.assertEqual(env.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
