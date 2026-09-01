"""Параметры штатного non-Giga маршрута."""

from types import SimpleNamespace

import main as assessor


def _config(temperature=0.001, top_p=0.001):
    return SimpleNamespace(
        contour="sds",
        contour_configs={"base_url": "https://gateway.example/api/v1"},
        llm_params={
            "temperature": temperature,
            "top_p": top_p,
            "timeout": 30,
        },
        verify_ssl_certs=True,
    )


def test_non_giga_omits_default_sampling_parameters(monkeypatch):
    captured = {}
    monkeypatch.delenv("TEMPERATURE", raising=False)
    monkeypatch.delenv("TOP_P", raising=False)
    monkeypatch.setattr(
        assessor,
        "SdsChatModel",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    assessor._build_judge_model("reasoning-model", _config(), "unused")

    assert captured["temperature"] is None
    assert captured["top_p"] is None


def test_non_giga_keeps_explicit_sampling_parameters(monkeypatch):
    captured = {}
    monkeypatch.setenv("TEMPERATURE", "0.2")
    monkeypatch.setenv("TOP_P", "0.3")
    monkeypatch.setattr(
        assessor,
        "SdsChatModel",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    assessor._build_judge_model(
        "reasoning-model",
        _config(temperature=0.2, top_p=0.3),
        "unused",
    )

    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.3
