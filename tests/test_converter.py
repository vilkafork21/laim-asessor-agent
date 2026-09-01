"""
Юнит-тесты laim-trace-converter (самодостаточная версия: извлечение Q/A внутри).

Синтетика по трём семействам трейсов:
  • сырой AEF exit-спан (update.messages) + телеметрия-шум;
  • state_json / langgraph chain (external+exit мульти-спан, message_to_user);
  • fipa_acl-конверт {outgoing, new_state}.
"""
import io
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as m  # noqa: E402


def _exit_output(question: str, answer: str, scenario: str = "rag") -> str:
    return json.dumps({
        "graph": None,
        "update": {
            "stage": "__end__",
            "scenario": scenario,
            "messages": [
                {"type": "human", "content": question},
                {"type": "ai", "content": answer},
            ],
            "message_to_user": answer,
        },
    }, ensure_ascii=False)


def _make_spans(n_dialogs: int = 3, n_telemetry: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n_telemetry):
        rows.append({"trace_id": f"tlm{i}", "session_id": f"s{i}",
                     "aef_kind": "output_request", "span_name": "OutgoingWebRequest",
                     "input_text": "[]", "output_text": "",
                     "start_time": "2026-06-01 05:40:01", "end_time": "2026-06-01 05:40:01"})
    for d in range(n_dialogs):
        tid = f"dlg{d}"
        rows.append({"trace_id": tid, "session_id": f"S{d}",
                     "aef_kind": "input_request", "span_name": "POST /invoke",
                     "input_text": "{}", "output_text": "",
                     "start_time": f"2026-06-0{d + 2} 10:00:00", "end_time": f"2026-06-0{d + 2} 10:00:05"})
        rows.append({"trace_id": tid, "session_id": f"S{d}",
                     "aef_kind": "chain", "span_name": "exit",
                     "input_text": "{}",
                     "output_text": _exit_output(f"вопрос {d}", f"ответ {d}",
                                                 scenario="rag" if d % 2 else "picker"),
                     "start_time": f"2026-06-0{d + 2} 10:00:00", "end_time": f"2026-06-0{d + 2} 10:00:05"})
    return pd.DataFrame(rows)


def _make_state_json_spans(n: int = 4) -> pd.DataFrame:
    """langgraph chain: external (без ответа) + exit (с ответом) на каждый trace."""
    rows = []
    for i in range(n):
        q, a = f"вопрос {i}", f"ответ {i}"
        base = {"trace_id": f"t{i}", "session_id": f"s{i}", "aef_kind": "chain",
                "span_name": "Deposelector Agent Graph",
                "output_text": "",
                "start_time": "2026-06-01 10:00:00", "end_time": "2026-06-01 10:00:05"}
        rows.append({**base, "input_text": json.dumps({
            "messages": [{"type": "human", "content": q}],
            "current_phrase": q, "stage": "external", "scenario": "picker",
            "message_to_user": None}, ensure_ascii=False)})
        rows.append({**base, "input_text": json.dumps({
            "messages": [{"type": "human", "content": q}, {"type": "ai", "content": a}],
            "current_phrase": q, "stage": "exit", "scenario": "picker",
            "message_to_user": a}, ensure_ascii=False)})
    return pd.DataFrame(rows)


def _make_fipa_spans(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        q, a = f"вопрос {i}", f"ответ {i}"
        rows.append({"trace_id": f"f{i}", "session_id": f"fs{i}",
                     "aef_kind": "input_request", "span_name": "fipa",
                     "input_text": "{}",
                     "output_text": json.dumps({
                         "outgoing": {
                             "receiver": "d-credit-helper",
                             "content": {"message": [{"type": "text", "value": a}]},
                         },
                         "new_state": {"messages": [
                             {"type": "human", "content": q}]},
                     }, ensure_ascii=False),
                     "start_time": "2026-06-01 10:00:00", "end_time": "2026-06-01 10:00:05"})
    return pd.DataFrame(rows)


# ── выбор спанов и канон ────────────────────────────────────────────────────
def test_selects_exit_span_only():
    out = m.convert(_make_spans(3, 50))
    assert len(out) == 3
    assert set(out["trace_id"]) == {"dlg0", "dlg1", "dlg2"}


def test_canonical_columns():
    out = m.convert(_make_spans(2))
    assert list(out.columns) == list(m.CANONICAL_COLUMNS)


def test_carries_ids_and_timestamps():
    out = m.convert(_make_spans(1))
    row = out.iloc[0]
    assert row["trace_id"] == "dlg0"
    assert row["query_id"] == "dlg0"
    assert row["session_id"] == "S0"
    assert row["starttime"] == "2026-06-02 10:00:00"
    assert row["endtime"] == "2026-06-02 10:00:05"


# ── извлечение Q/A (самодостаточность: судья получает текст, не блоб) ───────
def test_extracts_question_and_answer_from_aef_exit():
    out = m.convert(_make_spans(1))
    row = out.iloc[0]
    assert row["input_query"] == "вопрос 0"
    assert row["output_answer"] == "ответ 0"


def test_extracts_state_json_family():
    out = m.convert(_make_state_json_spans(4),
                    span_names=("Deposelector Agent Graph",),
                    input_field="input_text", answer_field="input_text")
    assert len(out) == 4                       # dedup external+exit
    assert list(out["input_query"]) == [f"вопрос {i}" for i in range(4)]
    assert list(out["output_answer"]) == [f"ответ {i}" for i in range(4)]


def test_extracts_fipa_acl_family():
    out = m.convert(_make_fipa_spans(3),
                    span_names=(), aef_kinds=("input_request",))
    assert list(out["input_query"]) == [f"вопрос {i}" for i in range(3)]
    assert list(out["output_answer"]) == [f"ответ {i}" for i in range(3)]


def test_classification_output_is_route():
    out = m.convert(_make_spans(2), postulation="classification",
                    route_keys=("scenario",))
    assert set(out["output_answer"]) == {"rag", "picker"}
    assert list(out["route"]) == list(out["output_answer"])


# ── dedup мульти-спанов ──────────────────────────────────────────────────────
def test_dedup_prefers_nonempty_answer():
    out = m.convert(_make_state_json_spans(3),
                    span_names=("Deposelector Agent Graph",),
                    input_field="input_text", answer_field="input_text")
    assert out["query_id"].duplicated().sum() == 0
    assert (out["output_answer"].astype(str).str.strip() != "").all()


def test_dedup_can_be_disabled():
    out = m.convert(_make_state_json_spans(3),
                    span_names=("Deposelector Agent Graph",),
                    input_field="input_text", answer_field="input_text",
                    dedup=False)
    assert len(out) == 6                       # external+exit, без схлопывания


# ── guard'ы: громкий отказ вместо молчаливого мусора ────────────────────────
def test_empty_input_raises_unless_allowed():
    with pytest.raises(m.ConversionError, match="0 спанов"):
        m.convert(pd.DataFrame())
    out = m.convert(pd.DataFrame(), allow_empty=True)
    assert list(out.columns) == list(m.CANONICAL_COLUMNS)
    assert len(out) == 0


def test_filter_mismatch_raises_with_available_values():
    spans = _make_state_json_spans(2)          # span_name = имя графа, не 'exit'
    with pytest.raises(m.ConversionError) as e:
        m.convert(spans)                       # дефолт span_name=('exit',)
    assert "Deposelector Agent Graph" in str(e.value)


def test_all_empty_extraction_raises():
    spans = _make_spans(2)
    spans["output_text"] = '{"noise": 1}'
    with pytest.raises(m.ConversionError, match="извлечение не нашло диалог"):
        m.convert(spans)


def test_single_route_class_raises():
    spans = _make_spans(2)
    spans["output_text"] = spans["output_text"].str.replace('"picker"', '"rag"')
    with pytest.raises(m.ConversionError, match="класс"):
        m.convert(spans, postulation="classification", route_keys=("scenario",))


def test_basket_coverage_gate():
    spans = _make_spans(4)
    with pytest.raises(m.ConversionError, match="словар"):
        m.convert(spans, postulation="classification", route_keys=("scenario",),
                  basket_answers=("common_credits", "decline"))
    out = m.convert(spans, postulation="classification", route_keys=("scenario",),
                    basket_answers=("rag", "picker"))
    assert len(out) == 4


# ── точка входа узла ────────────────────────────────────────────────────────
def test_main_returns_result_port():
    res = m.main(traces=_make_spans(2))
    assert set(res.keys()) == {"result"}
    assert len(res["result"]) == 2


def test_main_accepts_parquet_bytes():
    buf = io.BytesIO()
    _make_spans(2).to_parquet(buf)
    res = m.main(traces=buf.getvalue())
    assert len(res["result"]) == 2


def test_main_string_params_parsed():
    res = m.main(traces=_make_spans(2), span_name="('exit',)", aef_kind="('chain',)",
                 route_keys="('scenario',)", postulation="classification",
                 dedup="true", allow_empty="false")
    assert set(res["result"]["output_answer"]) == {"rag", "picker"}
