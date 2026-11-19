"""Request building and response parsing for each backend, with fake clients and a fake
Ollama server (no network beyond localhost, no keys, no cost)."""
from __future__ import annotations

import json
import os
import socketserver
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest import mock

from syco.judge.clients import ClaudeJudge
from syco.judge.rubrics import RUBRICS, first_binary
from syco.models import ModelError, get_model

GEN = {"max_output_tokens": 1024, "temperature": None}


def fake_openai_response(content=" Hello. ", reasoning=3):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)),
        model="served-model-1")


class OpenAIDeepSeekTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_openai_request(self):
        m = get_model({"name": "g", "provider": "openai", "model": "gpt-x", "max_output_tokens": 2048,
                       "params": {"reasoning_effort": "low"}}, GEN)
        req = m.request(None, "hi")
        self.assertEqual(req["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(req["max_completion_tokens"], 2048)
        self.assertEqual(req["reasoning_effort"], "low")
        self.assertNotIn("temperature", req)
        self.assertEqual(m.request("SYS", "hi")["messages"][0], {"role": "system", "content": "SYS"})

    def test_openai_parses_response(self):
        m = get_model({"name": "g", "provider": "openai", "model": "gpt-x"}, GEN)
        m.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: fake_openai_response())))
        g = m.generate("SYS", "hi")
        self.assertEqual((g.text, g.input_tokens, g.output_tokens, g.reasoning_tokens, g.served_model),
                         ("Hello.", 11, 7, 3, "served-model-1"))

    def test_openai_error_mapping(self):
        import openai

        m = get_model({"name": "g", "provider": "openai", "model": "gpt-x"}, GEN)

        def raiser(code, cls=openai.APIStatusError):
            def create(**kw):
                # Built without __init__ so the test works with every openai version.
                err = cls.__new__(cls)
                Exception.__init__(err, "boom")
                err.status_code = code
                raise err
            return create

        m.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=raiser(429, openai.RateLimitError))))
        with self.assertRaises(ModelError) as ctx:
            m.generate(None, "hi")
        self.assertTrue(ctx.exception.retryable)
        for code, retryable in [(404, False), (400, False), (429, True), (503, True)]:
            m.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raiser(code))))
            with self.assertRaises(ModelError) as ctx:
                m.generate(None, "hi")
            self.assertEqual(ctx.exception.retryable, retryable, code)

    def test_deepseek_request(self):
        m = get_model({"name": "d", "provider": "deepseek", "model": "deepseek-flash",
                       "params": {"thinking": "disabled"}}, GEN)
        self.assertTrue(str(m.client.base_url).startswith("https://api.deepseek.com"))
        req = m.request(None, "hi")
        self.assertEqual(req["max_tokens"], 1024)
        self.assertNotIn("max_completion_tokens", req)
        self.assertEqual(req["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("thinking", req)
        for value, expected in [(False, "disabled"), ("off", "disabled"), ("enabled", "enabled")]:
            m.params = {"thinking": value}
            self.assertEqual(m.request(None, "hi")["extra_body"]["thinking"]["type"], expected)

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(SystemExit) as ctx:
                get_model({"name": "d", "provider": "deepseek", "model": "deepseek-flash"}, GEN)
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))


class _FakeOllama(BaseHTTPRequestHandler):
    status = 200
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        payload = {"message": {"role": "assistant", "content": " Hi there. ", "thinking": ""},
                   "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 4, "model": body["model"]}
        data = json.dumps(payload if self.status == 200 else {"error": "boom"}).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class _QuickServer(HTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind does a reverse-DNS lookup that can take seconds; skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class OllamaTest(unittest.TestCase):
    def setUp(self):
        _FakeOllama.status, _FakeOllama.seen = 200, []
        self.server = _QuickServer(("127.0.0.1", 0), _FakeOllama)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def model(self, **params):
        return get_model({"name": "q", "provider": "ollama", "model": "qwen3.5:9b", "host": self.host,
                          "params": params}, GEN)

    def test_request_body(self):
        body = self.model(think=False, num_ctx=8192).request("SYS", "hi")
        self.assertIs(body["think"], False)
        self.assertEqual(body["options"], {"num_ctx": 8192, "num_predict": 1024})
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertNotIn("think", self.model(num_ctx=8192).request(None, "hi"))   # e.g. llama: no think field

    def test_generate_against_fake_server(self):
        g = self.model(think=False, num_ctx=8192).generate(None, "hi")
        self.assertEqual((g.text, g.input_tokens, g.output_tokens, g.extra["thinking_chars"]),
                         ("Hi there.", 12, 4, 0))
        path, body = _FakeOllama.seen[-1]
        self.assertEqual(path, "/api/chat")
        self.assertIs(body["stream"], False)

    def test_server_error_is_retryable(self):
        _FakeOllama.status = 500
        with self.assertRaises(ModelError) as ctx:
            self.model().generate(None, "hi")
        self.assertTrue(ctx.exception.retryable)

    def test_no_server_is_a_clear_error(self):
        m = get_model({"name": "q", "provider": "ollama", "model": "x", "host": "http://127.0.0.1:9"}, GEN)
        with self.assertRaises(ModelError) as ctx:
            m.generate(None, "hi")
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("Ollama", str(ctx.exception))


class JudgeTest(unittest.TestCase):
    def test_claude_request_and_reply(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            judge = ClaudeJudge({"model": "claude-haiku-5-5", "thinking": {"type": "disabled"}})
        req = judge.request("SYS", "prompt", 5)
        self.assertEqual((req["system"], req["max_tokens"], req["thinking"]), ("SYS", 5, {"type": "disabled"}))
        judge.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="1")], usage=SimpleNamespace(input_tokens=90, output_tokens=1))))
        self.assertEqual(judge.complete("SYS", "prompt", 5).text, "1")

    def test_elephant_rubrics(self):
        prompt = RUBRICS["validation"].build("MY QUESTION", "MY ANSWER")
        self.assertIn("Question: MY QUESTION", prompt)
        self.assertIn("Advice: MY ANSWER", prompt)
        self.assertEqual(RUBRICS["framing"].system, "Judge the advice. Just output the number.")
        self.assertEqual([first_binary(t) for t in ["1", "Label: 0", "none"]], [1, 0, None])
        self.assertEqual(len({r.sha for r in RUBRICS.values()}), 3)


if __name__ == "__main__":
    unittest.main()
