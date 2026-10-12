"""Actual HTTP smoke checks shared by Docker CI and clean-wheel validation.

Run against a started server (no docker SDK or project imports required):
  CLINICAL_LAB_BASE_URL=http://127.0.0.1:8000 \
  CLINICAL_LAB_EXPECT_MODELS=logistic \
  CLINICAL_LAB_EXPECT_DEMO=1 python tests/test_container_smoke.py

Ordinary unit test collection skips these checks unless a target URL is supplied.
"""
from __future__ import annotations

import json
import math
import os
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = os.environ.get("CLINICAL_LAB_BASE_URL", "").rstrip("/")


def request(path: str, payload: dict | None = None, *, timeout: float = 45):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    with urlopen(Request(BASE + path, data=body, headers=headers), timeout=timeout) as response:
        return response.read(), response.headers


def read_json(path: str, payload: dict | None = None) -> dict:
    return json.loads(request(path, payload)[0])


@unittest.skipUnless(BASE, "Set CLINICAL_LAB_BASE_URL to exercise a running image or wheel")
class ContainerSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deadline = time.monotonic() + 240
        last_error = None
        while time.monotonic() < deadline:
            try:
                cls.status = read_json("/api/status")
                return
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(2)
        raise AssertionError(f"API did not become ready: {last_error}")

    def test_health_and_model_initialization(self):
        expected = set(os.environ.get("CLINICAL_LAB_EXPECT_MODELS", "logistic").split(","))
        self.assertEqual(set(self.status["available_models"]), expected)
        self.assertEqual(self.status["replay_source"], "synthetic")
        self.assertFalse(self.status["patient_uploads"])
        self.assertFalse(self.status["clinical_use"])
        self.assertEqual(self.status["demo"], os.environ.get("CLINICAL_LAB_EXPECT_DEMO", "1") == "1")

    def test_packaged_frontend_and_headers(self):
        home, headers = request("/")
        self.assertIn(b"Clinical AI Reliability Lab", home)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for path in ("/static/app.js", "/static/style.css"):
            body, _ = request(path)
            self.assertGreater(len(body), 100)

    def test_report_and_every_available_model_replay(self):
        report = read_json("/api/report")
        self.assertIn("metadata", report["report"])
        patients = read_json("/api/patients")
        self.assertEqual(patients["source"], "synthetic")
        patient = patients["patients"][0]
        for family in self.status["available_models"]:
            for stress in ("none", "missingness", "delay", "noise"):
                with self.subTest(family=family, stress=stress):
                    payload = {"patient_id": patient["id"], "family": family,
                               "hour": 12, "stress": stress, "severity": "medium"}
                    replay = read_json("/api/replay", payload)
                    self.assertEqual(replay["source"], "synthetic")
                    self.assertEqual(replay["hours"], list(range(13)))
                    self.assertEqual(len(replay["scores"]), 13)
                    self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in replay["scores"]))
                    self.assertEqual(replay["family"], family)
                    self.assertTrue(replay["measurements"])

    def test_invalid_replay_returns_client_error(self):
        with self.assertRaises(HTTPError) as result:
            request("/api/replay", {"patient_id": "../../untrusted", "hour": 12})
        self.assertEqual(result.exception.code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
