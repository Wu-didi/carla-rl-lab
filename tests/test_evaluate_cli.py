from __future__ import annotations

import unittest

from scripts import evaluate


class EvaluateCliTest(unittest.TestCase):
    def test_full_and_limited_evaluations_have_distinct_scopes(self):
        parser = evaluate.build_argparser()
        full = parser.parse_args(["--checkpoint", "model.pt"])
        limited = parser.parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--routes",
                "5",
                "--weathers",
                "2",
            ]
        )
        self.assertEqual(evaluate.evaluation_scope(full), "full")
        self.assertEqual(
            evaluate.evaluation_scope(limited), "routes5_weathers2"
        )

    def test_output_tag_must_be_filesystem_safe(self):
        args = evaluate.build_argparser().parse_args(
            ["--checkpoint", "model.pt", "--output-tag", "bad/tag"]
        )
        with self.assertRaisesRegex(ValueError, "filesystem safe"):
            evaluate.evaluation_scope(args)


if __name__ == "__main__":
    unittest.main()
