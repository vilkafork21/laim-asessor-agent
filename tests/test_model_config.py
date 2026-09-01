"""Нормализация URL штатного AI Gateway."""

import pytest

from agent import config as config_module


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://gateway.example", "https://gateway.example/api/v1"),
        ("https://gateway.example/api/v1", "https://gateway.example/api/v1"),
        ("https://gateway.example/api/v1/", "https://gateway.example/api/v1"),
    ],
)
def test_ai_gateway_url_has_exactly_one_api_suffix(monkeypatch, configured, expected):
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("AI_GATEWAY_URL", configured)

    config = config_module.ModelsConfig()

    assert config.contour == "sds"
    assert config.contour_configs["base_url"] == expected
