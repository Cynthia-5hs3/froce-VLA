import json
import math
from pathlib import Path
import tempfile
import time
import unittest

from force_vla.pi05.telemetry_dashboard import TelemetryTail, pose_channels, project_record, publish


def record(step=1, timestamp=None):
    return {"step": step, "status": "policy", "control_at_ns": timestamp or time.monotonic_ns(),
            "model_action": [float(step)] * 10, "action": [float(step) + .1] * 10,
            "measured_state": [.2] * 9 + [.067], "measured_gripper_open_fraction": .8,
            "measured_tau_J": [float(step)] * 7, "predicted_tau_J": [float(step) + .2] * 7,
            "measured_tau_ext": [.5] * 7, "inference_ms": 250.}


class TelemetryDashboardTests(unittest.TestCase):
    def test_pose_units_and_rotation_convention(self):
        for rotation, expected in (
            ([1, 0, 0, 0, 1, 0], [0, 0, 0]),
            ([1, 0, 0, 0, 0, 1], [90, 0, 0]),
            ([0, 1, 0, -1, 0, 0], [0, 0, 90]),
            ([0, 0, -1, 0, 1, 0], [0, 90, 0]),
            ([0, 0, 1, 0, 1, 0], [0, -90, 0]),
        ):
            with self.subTest(rotation=rotation):
                position, angles = pose_channels([.4, -.1, .3, *rotation, 1])
                self.assertEqual(position, [400, -100, 300])
                for angle, target in zip(angles, expected):
                    self.assertAlmostEqual(angle, target)
        self.assertIsNone(pose_channels([0.] * 10)[1])
        self.assertEqual(pose_channels(None), (None, None))

    def test_joint_angles_and_manual_observation_without_model_prediction(self):
        source = record()
        source["model_action"] = None
        source["teleop_active"] = True
        source["measured_joint_positions"] = [math.pi / 2] * 7
        source["measured_state"] = [.4, 0, .3, 1, 0, 0, 0, 1, 0, .8]
        projected = project_record(source)
        self.assertIsNone(projected["raw_xyz_mm"])
        self.assertEqual(projected["measured_xyz_mm"], [400, 0, 300])
        self.assertEqual(projected["measured_rpy_deg"], [0, 0, 0])
        self.assertEqual(projected["joint_degrees"], [90] * 7)
        self.assertTrue(projected["teleop_active"])
        source.pop("measured_joint_positions")
        self.assertIsNone(project_record(source)["joint_degrees"])

    def test_keeps_raw_requested_and_measured_distinct_and_gripper_units_consistent(self):
        source = record()
        projected = project_record(source)
        self.assertEqual(projected["raw"], [1.] * 10)
        self.assertEqual(projected["requested"], [1.1] * 10)
        self.assertEqual(projected["measured_state"][-1], .8)
        self.assertEqual(source["measured_state"][-1], .067)
        self.assertEqual(projected["tau"], [1.] * 7)
        self.assertEqual(projected["external_tau"], [.5] * 7)

    def test_legacy_logs_leave_missing_channels_empty(self):
        source = record()
        source.pop("model_action")
        source.pop("measured_state")
        source["model_xyz"] = [1., 2., 3.]
        source["command_debug"] = {"measured_xyz": [4., 5., 6.]}
        projected = project_record(source)
        self.assertEqual(projected["raw"][:3], [1., 2., 3.])
        self.assertEqual(projected["raw"][3:9], [None] * 6)
        self.assertEqual(projected["measured_state"][:3], [4., 5., 6.])
        source["measured_tau_J"][0] = float("nan")
        self.assertIsNone(project_record(source))

    def test_incremental_partial_lines_restart_and_file_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            tail = TelemetryTail(path, 3)
            try:
                self.assertFalse(tail.poll())
                self.assertEqual(tail.payload(False)["status"], "waiting_file")
                path.write_text(json.dumps(record())[:-2])
                self.assertTrue(tail.poll())
                self.assertEqual(len(tail.rows), 0)
                with path.open("a") as stream:
                    stream.write(json.dumps(record())[-2:] + "\n")
                tail.poll()
                self.assertEqual(len(tail.rows), 1)
                tail.poll()
                self.assertEqual(len(tail.rows), 1)
                with path.open("a") as stream:
                    stream.write('bad JSON\n[]\n{"event":"inference_restart"}\n')
                tail.poll()
                self.assertEqual(len(tail.rows), 0)
                self.assertEqual(tail.latest["status"], "inference_restart")
                with path.open("a") as stream:
                    stream.write(json.dumps(record()) + '\n{"event":"deployment_session"}\n')
                tail.poll()
                self.assertEqual(len(tail.rows), 0)
                self.assertEqual(tail.latest["status"], "deployment_session")
                path.write_text(json.dumps(record(0)) + "\n")
                tail.poll()
                self.assertEqual(tail.latest["step"], 0)
                replacement = Path(directory) / "replacement"
                replacement.write_text(json.dumps(record(7)) + "\n")
                replacement.replace(path)
                tail.poll()
                self.assertEqual(len(tail.rows), 1)
                self.assertEqual(tail.latest["step"], 7)
                self.assertGreaterEqual(tail.invalid_lines, 2)
            finally:
                tail.close()

    def test_bounded_history_staleness_and_safe_json_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            path = destination / "log.jsonl"
            timestamp = time.monotonic_ns() - 10_000_000_000
            path.write_text("".join(json.dumps(record(step, timestamp + step)) + "\n" for step in range(10)))
            tail = TelemetryTail(path, 3)
            try:
                tail.poll()
                self.assertEqual(len(tail.rows), 3)
                payload = tail.payload(True)
                self.assertEqual(payload["status"], "stale")
                payload["log"] = "</script>"
                publish(destination, payload)
                script = (destination / "telemetry.js").read_text()
                self.assertNotIn("</script>", script)
                restored = json.loads(script[len("window.receiveTelemetry("):-3])
                self.assertEqual(restored["log"], "</script>")
                self.assertEqual(len(restored["rows"]), 3)
                source = record(11)
                source["inference_ms"] = float("nan")
                with path.open("a") as stream:
                    stream.write(json.dumps(source) + "\n")
                tail.poll()
                publish(destination, tail.payload(True))
            finally:
                tail.close()


if __name__ == "__main__":
    unittest.main()
