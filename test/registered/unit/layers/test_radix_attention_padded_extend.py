import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import radix_attention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingBackend:
    """Stand-in that checks the rows it is given against the batch's cache
    locations and returns one output row per query row."""

    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs):
        assert forward_batch.out_cache_loc.shape[0] == q.shape[0]
        self.calls.append(
            dict(
                q=q.shape[0],
                k=None if k is None else k.shape[0],
                positions=forward_batch.positions.shape[0],
                **{name: value.shape[0] for name, value in kwargs.items()},
            )
        )
        return torch.ones(q.shape[0], 6), torch.ones(q.shape[0], 2)


def _batch(mode, real, rows=8):
    return SimpleNamespace(
        forward_mode=mode,
        global_num_token_non_padded_cpu=real,
        out_cache_loc=torch.arange(rows),
        positions=torch.arange(rows),
    )


class TestPaddedExtendAttention(CustomTestCase):
    """MLP sync pads an extend batch's rows to a multiple of attention TP while
    its attention metadata covers only the real rows."""

    def test_only_padded_extend_batches_are_narrowed(self):
        q = torch.zeros(8, 2)
        cases = [
            (ForwardMode.EXTEND, 5, 5),
            (ForwardMode.EXTEND, 8, None),  # nothing padded
            (ForwardMode.EXTEND, 0, None),  # an idle rank's batch keeps its path
            (ForwardMode.TARGET_VERIFY, 5, None),
            (ForwardMode.DECODE, 5, None),
        ]
        for mode, real, expected in cases:
            with self.subTest(mode=mode, real=real):
                self.assertEqual(
                    radix_attention._padded_extend_real_tokens(q, _batch(mode, real)),
                    expected,
                )

    def test_the_backend_sees_the_real_rows_and_the_tail_is_zero(self):
        backend = _RecordingBackend()
        batch = _batch(ForwardMode.EXTEND, 5)
        out_cache_loc, positions = batch.out_cache_loc, batch.positions
        with patch.object(radix_attention, "get_attn_backend", return_value=backend):
            output, lse = radix_attention._attention_on_real_rows(
                None,
                5,
                torch.zeros(8, 2),
                torch.zeros(8, 2),
                torch.zeros(8, 2),
                batch,
                True,
                q_rope=torch.zeros(8, 1),
                k_rope=torch.zeros(8, 1),
            )
        self.assertEqual(
            backend.calls, [dict(q=5, k=5, positions=5, q_rope=5, k_rope=5)]
        )
        # The batch's own tensors come back for the rest of the layer.
        self.assertIs(batch.out_cache_loc, out_cache_loc)
        self.assertIs(batch.positions, positions)
        self.assertEqual(tuple(output.shape), (8, 6))
        torch.testing.assert_close(output[5:], torch.zeros(3, 6))
        torch.testing.assert_close(output[:5], torch.ones(5, 6))
        self.assertEqual(tuple(lse.shape), (8, 2))

    def test_keys_with_their_own_extent_are_left_whole(self):
        for key_value_num_tokens, key_rows in ((12, 12), (None, 12)):
            with self.subTest(key_value_num_tokens=key_value_num_tokens):
                backend = _RecordingBackend()
                with patch.object(
                    radix_attention, "get_attn_backend", return_value=backend
                ):
                    radix_attention._attention_on_real_rows(
                        None,
                        5,
                        torch.zeros(8, 2),
                        torch.zeros(key_rows, 2),
                        torch.zeros(key_rows, 2),
                        _batch(ForwardMode.EXTEND, 5),
                        False,
                        key_value_num_tokens=key_value_num_tokens,
                    )
                self.assertEqual(backend.calls[0]["k"], key_rows)


if __name__ == "__main__":
    unittest.main()
