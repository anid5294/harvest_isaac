"""Offline validation of canonical geometry and physical single-fruit layout."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

script = Path(__file__).resolve().parents[1] / "scripts/build_orchardbench_usd.py"
spec = importlib.util.spec_from_file_location("external_builder_test", script)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
adapter = builder.load_adapter()


class ExternalTreeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/"tree.json"
        self.tree = {"seed":42,"units":"m","up_axis":"Z", "branches":[
            {"id":0,"parent_id":-1,"start":[0,0,1.1],"end":[1,0,1.1],
             "radius_start":.01,"radius_end":.005}],
            "fruits":[{"id":8,"parent_branch_id":0,"anchor":[.5,0,1.1],
                       "center":[.5,0,1.04],"radius":.03,
                       "stem":{"mass_kg":.16,"break_force_n":17.0}}]}

    def load(self):
        self.path.write_text(json.dumps(self.tree))
        return adapter.load_tree(self.path,42)

    def test_native_position_and_single_dynamic_fruit(self):
        tree=self.load()
        layout,fruit=adapter.interaction_layout(tree)
        self.assertEqual(layout.apples, (tuple(fruit["center"]),))
        self.assertEqual(layout.radius,.03)
        self.assertEqual(layout.mass,.16)
        self.assertEqual(layout.stem_break_force,17.)
        self.assertEqual(tree["topology_validation"]["max_fruit_anchor_error_m"],0)

    def test_reject_floating_anchor(self):
        self.tree["fruits"][0]["anchor"][1]=.1
        with self.assertRaises(ValueError): self.load()

    def test_reject_wrong_seed(self):
        self.tree["seed"]=43
        with self.assertRaises(ValueError): self.load()

    def test_reject_cycle(self):
        self.tree["branches"][0]["parent_id"]=0
        with self.assertRaises(ValueError): self.load()

    def test_roundoff_taper_allowed(self):
        self.tree["branches"][0]["radius_end"]=.01+1e-17
        self.load()

    def test_unknown_fruit_rejected(self):
        with self.assertRaises(ValueError): adapter.interaction_layout(self.load(),99)

    def test_joint_frames_coincide(self):
        layout,_=adapter.interaction_layout(self.load())
        geometry=sys.modules["_offline_orchard.layout"].stem_geometry(layout,0)
        surface=tuple(a+b for a,b in zip(layout.target,geometry["fruit_joint_position"]))
        self.assertEqual(surface,geometry["bottom"])

    def test_checked_in_asset_hashes_and_layout(self):
        directory=script.parents[1]/"assets/orchard/orchardbench_seed42/isaac"
        layout,path,manifest=adapter.load_asset(directory,42)
        self.assertEqual(manifest["selected_fruit_id"],28)
        self.assertEqual(len(layout.apples),1)
        self.assertGreater(manifest["selected_fruit_wood_clearance_m"],.007)
        self.assertEqual(manifest["branch_count"],168)

    def test_new_camera_profile_has_exactly_three_views(self):
        path=script.with_name("preview_views.py")
        spec=importlib.util.spec_from_file_location("external_preview_views",path)
        preview=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preview)
        self.assertEqual([v[1] for v in preview.camera_views("orchardbench_single_tree_three_view_v1")],
                         ["external","left_wrist","right_wrist"])


if __name__ == "__main__": unittest.main()
