"""OpenAI-compatible параметры non-Giga judge."""

from agent.sds_chat_model import SdsChatModel


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{
                "message": {"content": '{"score":1}'},
                "finish_reason": "stop",
            }]
        }


def test_none_sampling_parameters_are_not_sent(monkeypatch):
    captured = {}

    def fake_post(*_args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr("agent.sds_chat_model.requests.post", fake_post)
    model = SdsChatModel(
        base_url="https://gateway.example/api/v1",
        model_id="reasoning-model",
        temperature=None,
        top_p=None,
        timeout=1,
    )

    model.invoke("test")

    assert "temperature" not in captured
    assert "top_p" not in captured
