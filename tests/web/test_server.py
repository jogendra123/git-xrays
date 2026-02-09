from urllib.error import URLError

from git_xrays.web import server


class _Response200:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_wait_for_api_returns_true_when_ready(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(url, timeout):
        calls["count"] += 1
        return _Response200()

    monkeypatch.setattr(server, "urlopen", fake_urlopen)

    assert server._wait_for_api(8000, timeout_seconds=0.1) is True
    assert calls["count"] == 1


def test_wait_for_api_returns_false_on_timeout(monkeypatch):
    def fake_urlopen(url, timeout):
        raise URLError("not ready")

    time_values = iter([0.0, 0.05, 0.1, 0.15])

    monkeypatch.setattr(server, "urlopen", fake_urlopen)
    monkeypatch.setattr(server.time, "time", lambda: next(time_values))
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)

    assert server._wait_for_api(8000, timeout_seconds=0.1) is False
