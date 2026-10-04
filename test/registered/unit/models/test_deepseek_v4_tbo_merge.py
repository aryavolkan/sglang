"""DeepSeek-V4 runs its own two-batch-overlap layer loop. Its layers carry no
residual, so the loop merges the two microbatches' hidden states only. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import operations
from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeepseekV4TboMerge(CustomTestCase):
    def test_microbatches_without_residual_merge_by_token_range(self):
        hc_mult, hidden_size = 2, 3
        hidden_states = torch.arange(5 * hc_mult * hidden_size, dtype=torch.float32)
        hidden_states = hidden_states.view(5, hc_mult, hidden_size)
        children = [
            SimpleNamespace(tbo_parent_token_range=(0, 3), tbo_padded_len=3),
            SimpleNamespace(tbo_parent_token_range=(3, 5), tbo_padded_len=3),
        ]
        forward_batch = SimpleNamespace(tbo_children=children, global_forward_mode=None)
        model = SimpleNamespace(layers=[None, None], start_layer=0, end_layer=2)

        def execute(inputs_arr, operations_arr, delta_stages):
            outputs = []
            for part in inputs_arr:
                self.assertIsNone(part["residual"])
                self.assertEqual(part["hidden_states"].shape, (3, hc_mult, hidden_size))
                outputs.append(
                    dict(
                        positions=part["positions"],
                        hidden_states=part["hidden_states"] * 2,
                        residual=None,
                        forward_batch=part["forward_batch"],
                        tbo_subbatch_index=part["tbo_subbatch_index"],
                    )
                )
            return outputs

        with (
            get_parallel().override(attn_dp_size=1),
            patch.object(
                OperationsStrategy,
                "init_new_tbo",
                return_value=SimpleNamespace(operations=[], tbo_delta_stages=0),
            ),
            patch.object(operations, "execute_overlapped_operations", execute),
        ):
            merged = DeepseekV4Model._forward_layers_tbo(
                model, torch.arange(5), hidden_states, forward_batch
            )

        torch.testing.assert_close(merged, hidden_states * 2)


if __name__ == "__main__":
    unittest.main()
