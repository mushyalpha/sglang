import unittest
from types import SimpleNamespace

from sglang.srt.speculative.spec_utils import check_token_map_compatible_with_lm_head
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSpecTokenMapTPGuard(CustomTestCase):
    def test_sharded_lm_head_is_rejected(self):
        lm_head = SimpleNamespace(tp_size=2)
        with self.assertRaisesRegex(ValueError, "--speculative-token-map"):
            check_token_map_compatible_with_lm_head(lm_head)

    def test_unsharded_lm_head_is_accepted(self):
        check_token_map_compatible_with_lm_head(SimpleNamespace(tp_size=1))

    def test_missing_lm_head_or_tp_size_is_accepted(self):
        check_token_map_compatible_with_lm_head(None)
        check_token_map_compatible_with_lm_head(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
