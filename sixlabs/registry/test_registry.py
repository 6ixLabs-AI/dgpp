#!/usr/bin/env python3
"""Tests for the model registry: the shipped models.json is consistent with the tree, resolve
builds what it should and refuses what it should, and check notices a broken registry.

  python3 -m unittest sixlabs/registry/test_registry.py
"""
import copy, json, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as R


class ShippedRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = R.load()

    def test_consistent_with_the_tree(self):
        self.assertEqual(R.check(self.reg), [])

    def test_every_served_family_of_the_binary_is_listed(self):
        served = {n for n, f in self.reg["families"].items() if f["served"]}
        self.assertEqual(served, R.server_families(R.ROOT))

    def test_table_has_the_working_models_and_their_baselines(self):
        t = R.table(self.reg, ["working"])
        self.assertIn("Qwen3-Next-80B-A3B Instruct", t)
        self.assertIn("Qwen3.6-35B-A3B", t)
        self.assertIn("| vllm | baseline |", t)
        self.assertNotIn("Nemotron", t)


class Resolve(unittest.TestCase):
    def setUp(self):
        self.reg = R.load()

    def test_working_model_gets_family_defaults_only_where_the_template_is_silent(self):
        reg = copy.deepcopy(self.reg)
        reg["families"]["qwen3_5"]["engine_defaults"] = {"prefill_order": "shortest", "kv_capacity": 1}
        cfg, prov = R.resolve(reg, "qwen3.6-35b-a3b")
        with open(os.path.join(R.ROOT, prov["template"])) as f:
            template = json.load(f)
        for k, v in template["engine"].items():
            self.assertEqual(cfg["engine"][k], v, k)          # the template wins (kv_capacity too)
        self.assertNotIn("prefill_order", template["engine"])
        self.assertEqual(cfg["engine"]["prefill_order"], "shortest")   # filled where the template is silent
        self.assertEqual(cfg["model"], "nvidia/Qwen3.6-35B-A3B-NVFP4")
        self.assertEqual(prov["world"], 1)
        self.assertEqual(set(cfg) - set(template), set())      # no key the engine does not know

    def test_the_35b_resolves_to_the_settings_it_was_verified_with(self):
        # The 35B was verified and measured at the automatic prefill budget. The serve binary
        # refused an explicit one for qwen3_5 until cecb0cd (the 35B exited at start on DGXone on
        # 2026-10-04 with one); it accepts it now, but no checkpoint of the family has been checked
        # on a GPU with a larger budget, so the shipped registry does not hand one out yet.
        cfg, _ = R.resolve(self.reg, "qwen3.6-35b-a3b")
        self.assertNotIn("prefill_budget_tokens", cfg["engine"])

    def test_the_sehyo_122b_has_its_own_template_and_the_small_pool(self):
        cfg, prov = R.resolve(self.reg, "qwen3.5-122b-a10b")      # the verified checkpoint is the default
        self.assertEqual(cfg["model"], "Sehyo/Qwen3.5-122B-A10B-NVFP4")
        self.assertEqual(cfg["engine"]["kv_capacity"], 65536)
        self.assertEqual(prov["status"], "working")

    def test_set_overrides_everything_and_parses_json(self):
        cfg, prov = R.resolve(self.reg, "qwen3-next-80b", sets=["kv_capacity=262144", "prefill_order=fair"])
        self.assertEqual(cfg["engine"]["kv_capacity"], 262144)
        self.assertEqual(cfg["engine"]["prefill_order"], "fair")
        self.assertEqual(prov["from_set"], {"kv_capacity": 262144, "prefill_order": "fair"})

    def test_variant_and_world(self):
        cfg, prov = R.resolve(self.reg, "qwen3-next-80b", variant="yarn512k")
        self.assertTrue(prov["template"].endswith("yarn512k.example.json"))
        cfg, prov = R.resolve(self.reg, "qwen3.8-flash-next", world=2)
        self.assertEqual(cfg["world_size"], 2)
        with self.assertRaises(R.RegistryError):
            R.resolve(self.reg, "qwen3-next-80b", world=2)
        with self.assertRaises(R.RegistryError):
            R.resolve(self.reg, "qwen3-next-80b", variant="nope")

    def test_compiled_is_refused_unless_allowed(self):
        with self.assertRaises(R.RegistryError) as e:
            R.resolve(self.reg, "qwen3.5-0.8b")
        self.assertIn("gpu-steps", str(e.exception))
        cfg, prov = R.resolve(self.reg, "qwen3.5-0.8b", allow_unverified=True)
        self.assertEqual(prov["status"], "compiled")

    def test_groundwork_and_refused_are_never_served(self):
        with self.assertRaises(R.RegistryError):
            R.resolve(self.reg, "nemotron-3-nano", allow_unverified=True)
        with self.assertRaises(R.RegistryError):
            R.resolve(self.reg, "qwen3-coder-next", checkpoint="Intel/Qwen3-Coder-Next-int4-AutoRound", allow_unverified=True)
        with self.assertRaises(R.RegistryError):
            R.resolve(self.reg, "no-such-model")


class CheckNotices(unittest.TestCase):
    def broken(self, edit):
        reg = copy.deepcopy(R.load())
        edit(reg)
        return R.check(reg)

    def test_unknown_family(self):
        self.assertTrue(self.broken(lambda r: r["models"]["qwen3-next-80b"].update(family="nope")))

    def test_served_family_missing_from_the_binary(self):
        self.assertTrue(any("no family of that name" in b for b in self.broken(lambda r: r["families"]["nemotron_h"].update(served=True))))

    def test_a_family_of_the_binary_missing_from_the_registry(self):
        self.assertTrue(any("does not list as served" in b for b in self.broken(lambda r: r["families"]["mimo_v2"].update(served=False))))

    def test_working_without_a_record(self):
        def edit(r):
            r["models"]["qwen3.6-35b-a3b"]["checkpoints"][0]["verified"]["record"] = "sixlabs/ports/nope.md"
        self.assertTrue(any("does not exist" in b for b in self.broken(edit)))

    def test_template_for_another_model(self):
        def edit(r):
            r["models"]["qwen3.5-0.8b"]["checkpoints"][0]["worlds"]["1"]["template"] = "deploy/cluster_qwen3.8-27b_fp8_w1.example.json"
        self.assertTrue(any("names model" in b for b in self.broken(edit)))

    def test_engine_default_the_engine_does_not_parse(self):
        self.assertTrue(any("not a key" in b for b in self.broken(lambda r: r["families"]["qwen3_5"]["engine_defaults"].update(no_such_key=1))))

    def test_prefill_budget_default_on_a_family_the_binary_refuses_it_for(self):
        accepted = R.prefill_budget_families(R.ROOT)
        self.assertIn("qwen3_next", accepted)
        self.assertNotIn("glm4_moe", accepted)
        self.assertTrue(any("refused by apps/dgpp_serve.cpp" in b for b in
                            self.broken(lambda r: r["families"]["glm4_moe"]["engine_defaults"].update(prefill_budget_tokens=1024))))
        self.assertFalse(any("refused by apps/dgpp_serve.cpp" in b for b in
                             self.broken(lambda r: r["families"]["qwen3_next"]["engine_defaults"].update(prefill_budget_tokens=512))))

    def test_branch_model_marked_as_runnable(self):
        def edit(r):
            r["models"]["gemma-4-31b"]["branch"] = "ports/gemma-4"       # as it was before it landed
            r["models"]["gemma-4-31b"]["checkpoints"][0]["status"] = "compiled"
        self.assertTrue(any("not on main" in b for b in self.broken(edit)))


if __name__ == "__main__":
    unittest.main()
