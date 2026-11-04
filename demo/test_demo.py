"""Run with: python demo/test_demo.py"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
import llm_client
from http.server import BaseHTTPRequestHandler
from templates import EXTRA_CASES, TEMPLATES


class Templates(unittest.TestCase):
    def test_covers_every_class(self):
        self.assertEqual(set(TEMPLATES), set(engine.CLASSES))

    def test_each_template_classifies_as_intended(self):
        for label, text in TEMPLATES.items():
            with self.subTest(label):
                self.assertEqual(engine.run_nlp(text)["label"], label)

    def test_each_template_has_highlights(self):
        for label, text in TEMPLATES.items():
            with self.subTest(label):
                self.assertTrue(engine.highlight_spans(text))


class Rendering(unittest.TestCase):
    def test_html_is_escaped(self):
        out = engine.render_report("Delayed enhancement <script>alert(1)</script>")
        self.assertNotIn("<script>", out)

    def test_skipped_sections_are_not_highlighted(self):
        text = "HISTORY: myocarditis\nFINDINGS:\nLate gadolinium enhancement: mid-wall scar."
        marked = [text[a:b].lower() for a, b, _ in engine.highlight_spans(text)]
        self.assertNotIn("myocarditis", marked)
        self.assertIn("mid-wall", marked)

    def test_findings_mask_matches_rules_output(self):
        for text in TEMPLATES.values():
            mask = engine.findings_mask(text)
            kept = "".join(c for c, keep in zip(text, mask) if keep)
            rules_kept = engine.nlp._strip_non_findings_sections(text)
            self.assertEqual("".join(kept.split()), "".join(rules_kept.split()))

    def test_three_figure_colors_used(self):
        labels = {lbl for _, _, lbl in engine.highlight_spans(TEMPLATES["ischemic"])}
        self.assertLessEqual({"anatomy", "pattern", "disease"}, labels)

    def test_no_scar_shows_negative_statement(self):
        labels = {lbl for _, _, lbl in engine.highlight_spans(TEMPLATES["no-scar"])}
        self.assertIn("negative", labels)


class ExtraCases(unittest.TestCase):
    def test_negated_wording_is_no_scar(self):
        for text in EXTRA_CASES.values():
            self.assertEqual(engine.run_nlp(text)["label"], "no-scar")


class Upload(unittest.TestCase):
    def test_txt_is_read_and_temp_copy_removed(self):
        import app
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Report body")
        self.assertEqual(app.read_uploaded_report(f.name), "Report body")
        self.assertFalse(os.path.exists(f.name))

    def test_bad_docx_returns_empty_and_is_removed(self):
        import app
        with tempfile.NamedTemporaryFile("w", suffix=".docx", delete=False) as f:
            f.write("not a docx")
        self.assertEqual(app.read_uploaded_report(f.name), "")
        self.assertFalse(os.path.exists(f.name))


class Exclusions(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(engine.run_nlp("  ")["reason"], "empty")

    def test_device(self):
        result = engine.run_nlp("Pacemaker in place. Late gadolinium enhancement: mid-wall scar.")
        self.assertIsNone(result["label"])
        self.assertEqual(result["reason"], "device")

    def test_no_lge_section(self):
        result = engine.run_nlp("Normal biventricular function.")
        self.assertIsNone(result["label"])
        self.assertEqual(result["reason"], "no_lge")


class Fallback(unittest.TestCase):
    def test_default_url_is_local_ollama(self):
        self.assertEqual(llm_client.get_backend({}).url, llm_client.DEFAULT_URL)

    def test_empty_url_means_nlp_only(self):
        self.assertFalse(llm_client.get_backend({"CMR_LLM_URL": ""}).configured)

    def test_remote_url_blocked_by_default(self):
        self.assertFalse(llm_client.get_backend({"CMR_LLM_URL": "http://gpu.example.org:11434"}).configured)

    def test_unreachable_server_returns_no_label(self):
        result = llm_client.get_backend({"CMR_LLM_URL": "http://127.0.0.1:9"}).classify("text")
        self.assertIsNone(result.label)
        self.assertTrue(result.message)


class PaperPrompt(unittest.TestCase):
    def test_prompt_is_the_papers_zero_shot_prompt(self):
        system, user = llm_client.load_prompt()
        self.assertEqual(len(system), 2702)   # the length given in the paper
        self.assertIn("{report_text}", user)

    def test_parser(self):
        self.assertEqual(llm_client.parse_answer('```json\n{"classification": "Mixed"}\n```'), "mixed")
        self.assertEqual(llm_client.parse_answer('{"classification": "no-scar", "x": [}'), "no-scar")
        self.assertIsNone(llm_client.parse_answer('{"classification": "scar"}'))


class OllamaProtocol(unittest.TestCase):
    """Client against a fake Ollama server."""

    def _serve(self, answer):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body):
                data = json.dumps(body).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def do_GET(self):
                self._send({"models": []})

            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self._send({"message": {"content": answer}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        backend = llm_client.get_backend({"CMR_LLM_URL": f"http://127.0.0.1:{server.server_address[1]}", "CMR_LLM_MODEL": "fake"})
        return backend, seen

    def test_label_and_paper_settings(self):
        backend, seen = self._serve('{"classification": "mixed"}')
        result = backend.classify("some report")
        self.assertEqual((result.label, result.model), ("mixed", "fake"))
        opts = seen[0]["options"]
        self.assertEqual((opts["temperature"], opts["seed"], opts["num_ctx"], seen[0]["think"]), (0.0, 42, 8192, False))
        self.assertIn("some report", seen[0]["messages"][1]["content"])

    def test_unparseable_answer_is_missing(self):
        backend, _ = self._serve("I think it is scar")
        self.assertIsNone(backend.classify("some report").label)

    def test_fewshot_off_by_default(self):
        self.assertFalse(llm_client.get_fewshot_backend({}).configured)

    def test_fewshot_prompt_has_five_examples(self):
        import json as _j
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for i, (lab, t) in enumerate([(k, v) for k, v in TEMPLATES.items()] * 2):
                f.write(_j.dumps({"report_id": f"S{i}", "text": t, "label": lab}) + "\n")
        self.addCleanup(os.remove, f.name)
        backend = llm_client.get_fewshot_backend({"CMR_DEMO_FEWSHOT": "1", "CMR_FEWSHOT_POOL": f.name})
        msgs = backend._messages(TEMPLATES["mixed"])
        self.assertEqual(msgs[1]["content"].count("=== EXAMPLE (label:"), 5)
        self.assertEqual(len(msgs[0]["content"]), 2241)   # the paper's few-shot prompt

    def test_agreement_message(self):
        ok = llm_client.LLMResult("mixed", model="fake")
        self.assertIn("97.8%", engine.llm_card(ok, "mixed"))
        self.assertIn("expert review", engine.llm_card(ok, "ischemic"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
