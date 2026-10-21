import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from force_vla.stage_annotations import ROOT, SCHEMA, build_index, canonical_name, frame_labels, import_labels, validate_document, window_review


class StageAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.source = {"frame_count": 120, "episode_index": 37, "split": "train", "data_sha256": "source-a"}
        self.document = {"schema": SCHEMA, "source": self.source, "annotator": "test",
                         "complete": False, "segments": [], "keyframes": []}

    def segment(self, start, end, decision="keep", stage="aligned_press"):
        return {"start_frame": start, "end_frame": end, "stage": stage,
                "imitation": decision, "confidence": "confirmed", "note": "test"}

    def test_adjacent_intervals_are_valid_and_end_is_exclusive(self):
        self.document["segments"] = [self.segment(0, 60), self.segment(60, 120, "exclude")]
        self.document["complete"] = True
        self.assertEqual(validate_document(self.document, self.source)["covered_frames"], 120)
        _, decisions = frame_labels(self.document)
        self.assertEqual(decisions[59], "keep")
        self.assertEqual(decisions[60], "exclude")

    def test_wrong_source_rejected(self):
        changed = copy.deepcopy(self.document)
        changed["source"]["data_sha256"] = "different"
        with self.assertRaises(ValueError):
            validate_document(changed, self.source)

    def test_overlap_out_of_range_and_false_completion_rejected(self):
        for segments, complete in [([self.segment(0, 61), self.segment(60, 120)], False),
                                   ([self.segment(-1, 30)], False),
                                   ([self.segment(0, 121)], False),
                                   ([self.segment(1, 120)], True)]:
            with self.subTest(segments=segments, complete=complete):
                self.document.update(segments=segments, complete=complete)
                with self.assertRaises(ValueError):
                    validate_document(self.document, self.source)

    def test_uncertain_cannot_be_approved(self):
        segment = self.segment(0, 60)
        segment["confidence"] = "uncertain"
        self.document["segments"] = [segment]
        with self.assertRaises(ValueError):
            validate_document(self.document, self.source)
        segment["imitation"] = "review"
        self.assertEqual(validate_document(self.document, self.source)["review_frames"], 60)

    def test_mask_checks_all_future_actions_not_only_anchor(self):
        self.document["segments"] = [self.segment(0, 60), self.segment(60, 80, "exclude"),
                                     self.segment(80, 120)]
        stages, decisions = frame_labels(self.document)
        clean = window_review(10, list(range(11, 61)), stages, decisions)
        self.assertTrue(clean["all_future_actions_approved"])
        crossing = window_review(11, list(range(12, 62)), stages, decisions)
        self.assertFalse(crossing["all_future_actions_approved"])
        self.assertTrue(crossing["action_target_mask"][0])
        self.assertFalse(crossing["action_target_mask"][-1])
        self.assertEqual(crossing["suggested_sampling_weight"], 0)

    def test_bad_history_can_precede_good_recovery_targets(self):
        self.document["segments"] = [self.segment(0, 60, "exclude", "misaligned_contact"),
                                     self.segment(60, 120, "keep", "correction")]
        stages, decisions = frame_labels(self.document)
        result = window_review(60, list(range(61, 111)), stages, decisions)
        self.assertTrue(result["all_future_actions_approved"])
        self.assertEqual(result["anchor_stage"], "correction")

    def test_unmarked_is_not_approved(self):
        stages, decisions = frame_labels(self.document)
        result = window_review(0, list(range(1, 51)), stages, decisions)
        self.assertFalse(any(result["action_target_mask"]))

    def test_invalid_action_alignment_rejected(self):
        stages, decisions = frame_labels(self.document)
        with self.assertRaises(ValueError):
            window_review(0, list(range(2, 52)), stages, decisions)

    def test_keyframes_do_not_approve_actions(self):
        self.document["keyframes"] = [{"frame": 60, "event": "seated", "note": "visible"}]
        self.assertEqual(validate_document(self.document, self.source)["covered_frames"], 0)
        stages, decisions = frame_labels(self.document)
        self.assertFalse(window_review(0, list(range(1, 51)), stages, decisions)["all_future_actions_approved"])

    def test_import_archive_and_index_exclude_validation(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory(prefix="annotation-test-", dir=ROOT / "outputs") as temporary_output, tempfile.TemporaryDirectory(prefix="annotation-test-", dir=ROOT / "annotations") as temporary_annotation:
            root = Path(temporary_output)
            annotation_dir = Path(temporary_annotation)
            source = {**self.source, "dataset": "data/SYNTHETIC_TEST_ONLY"}
            document = {**self.document, "source": source,
                        "segments": [self.segment(0, 60, "exclude"), self.segment(60, 120, "keep", "correction")]}
            incoming = root / "incoming.json"
            incoming.write_text(json.dumps(document))
            windows = root / "windows.parquet"
            parquet.write_table(arrow.Table.from_pylist([
                {"episode_index": episode, "anchor_frame": anchor, "future_frame_indices": list(range(anchor + 1, anchor + 51))}
                for episode, anchor in [(37, 0), (37, 60), (39, 60)]]), windows)
            context = {"windows": windows, "episodes": {37: {"length": 120}, 39: {"length": 120}},
                       "contract": {"train_episodes": [37], "validation_episodes": [39], "sha256": "synthetic"}}
            args = SimpleNamespace(contract="unused", files=[str(incoming)], annotations=str(annotation_dir), replace=False)
            with patch("force_vla.stage_annotations.load_context", return_value=context), patch("force_vla.stage_annotations.episode_source", return_value=(source, None, {})):
                import_labels(args)
                target = annotation_dir / canonical_name(source)
                self.assertTrue(target.exists())
                document["segments"][1]["note"] = "reviewed again"
                incoming.write_text(json.dumps(document))
                with self.assertRaises(FileExistsError):
                    import_labels(args)
                args.replace = True
                import_labels(args)
                archives = list((annotation_dir / "history").glob("*.json"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(json.loads(archives[0].read_text())["segments"][1]["note"], "test")
                destination = root / "index"
                build_index(SimpleNamespace(contract="unused", annotations=str(annotation_dir), output=str(destination)))
                eligible = json.loads((destination / "eligible_train_rows.json").read_text())
                self.assertEqual(eligible["rows"], [1])
                report = json.loads((destination / "summary.json").read_text())
                self.assertEqual(report["validation_episodes_unchanged"], [39])
                self.assertFalse(report["training_ready"])

    def test_invalid_second_import_leaves_first_unwritten(self):
        with tempfile.TemporaryDirectory(prefix="annotation-test-", dir=ROOT / "outputs") as temporary_output, tempfile.TemporaryDirectory(prefix="annotation-test-", dir=ROOT / "annotations") as temporary_annotation:
            root = Path(temporary_output)
            source = {**self.source, "dataset": "data/SYNTHETIC_TEST_ONLY"}
            valid = {**self.document, "source": source}
            invalid = {**valid, "annotator": ""}
            paths = [root / "valid.json", root / "invalid.json"]
            for path, document in zip(paths, [valid, invalid]):
                path.write_text(json.dumps(document))
            with patch("force_vla.stage_annotations.load_context", return_value={}), patch("force_vla.stage_annotations.episode_source", return_value=(source, None, {})):
                with self.assertRaises(ValueError):
                    import_labels(SimpleNamespace(contract="unused", files=paths, annotations=temporary_annotation, replace=False))
            self.assertEqual(list(Path(temporary_annotation).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
