"""Tests for DefectClassifier DefectResult, MockDefectClassifier, and VLLMDefectClassifier."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult
from skillet.perception.inspection.mock_defect_classifier import MockDefectClassifier
from skillet.perception.inspection.vllm_defect_classifier import VLLMDefectClassifier

_BLANK_IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
_FAKE_BUF = np.array([0, 1, 2], dtype=np.uint8)


# ---------------------------------------------------------------------------
# MockDefectClassifier
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_classifier() -> MockDefectClassifier:
    return MockDefectClassifier(
        {
            "block_ok": DefectResult(defective=False, confidence=0.95),
            "block_bad": DefectResult(defective=True, confidence=0.88),
        }
    )


def test_mock_returns_correct_result(mock_classifier: MockDefectClassifier) -> None:
    v = mock_classifier.classify(_BLANK_IMAGE, "block_ok")
    assert v.defective is False
    assert v.confidence == pytest.approx(0.95)


def test_mock_returns_defective_result(mock_classifier: MockDefectClassifier) -> None:
    v = mock_classifier.classify(_BLANK_IMAGE, "block_bad")
    assert v.defective is True
    assert v.confidence == pytest.approx(0.88)


def test_mock_raises_for_unknown_object(mock_classifier: MockDefectClassifier) -> None:
    with pytest.raises(KeyError, match="no_such_block"):
        mock_classifier.classify(_BLANK_IMAGE, "no_such_block")


def test_mock_is_defect_classifier_subclass(mock_classifier: MockDefectClassifier) -> None:
    assert isinstance(mock_classifier, DefectClassifier)


# ---------------------------------------------------------------------------
# VLLMDefectClassifier — _parse_result_from
# ---------------------------------------------------------------------------


def test_parse_result_yes() -> None:
    v = VLLMDefectClassifier._parse_result_from("YES", logprob=math.log(0.92))
    assert v.defective is True
    assert v.confidence == pytest.approx(0.92, rel=1e-5)


def test_parse_result_no() -> None:
    v = VLLMDefectClassifier._parse_result_from("NO", logprob=math.log(0.80))
    assert v.defective is False
    assert v.confidence == pytest.approx(0.80, rel=1e-5)


def test_parse_result_lowercase() -> None:
    v = VLLMDefectClassifier._parse_result_from("yes", logprob=math.log(0.75))
    assert v.defective is True


# ---------------------------------------------------------------------------
# VLLMDefectClassifier — full classify() call (OpenAI client mocked)
# ---------------------------------------------------------------------------


def _make_api_response(token_text: str, logprob: float) -> MagicMock:
    token_logprob = MagicMock()
    token_logprob.logprob = logprob

    choice = MagicMock()
    choice.message.content = token_text
    choice.logprobs.content = [token_logprob]

    response = MagicMock()
    response.choices = [choice]
    return response


@patch("skillet.perception.inspection.vllm_defect_classifier.cv2.imencode", return_value=(True, _FAKE_BUF))
@patch("openai.OpenAI")
def test_vllm_classify_defective(mock_openai_cls: MagicMock, _mock_encode: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_api_response("YES", logprob=math.log(0.91))

    classifier = VLLMDefectClassifier(base_url="http://localhost:8000/v1", model="llava-1.5")
    result = classifier.classify(_BLANK_IMAGE, "block_a")

    assert result.defective is True
    assert result.confidence == pytest.approx(0.91, rel=1e-4)


@patch("skillet.perception.inspection.vllm_defect_classifier.cv2.imencode", return_value=(True, _FAKE_BUF))
@patch("openai.OpenAI")
def test_vllm_classify_non_defective(mock_openai_cls: MagicMock, _mock_encode: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_api_response("NO", logprob=math.log(0.85))

    classifier = VLLMDefectClassifier(base_url="http://localhost:8000/v1", model="llava-1.5")
    result = classifier.classify(_BLANK_IMAGE, "block_b")

    assert result.defective is False
    assert result.confidence == pytest.approx(0.85, rel=1e-4)


@patch("skillet.perception.inspection.vllm_defect_classifier.cv2.imencode", return_value=(True, _FAKE_BUF))
@patch("openai.OpenAI")
def test_vllm_logprobs_param_sent(mock_openai_cls: MagicMock, _mock_encode: MagicMock) -> None:
    """The API call must request logprobs so confidence is derived from token probability."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_api_response("YES", logprob=math.log(0.7))

    classifier = VLLMDefectClassifier(base_url="http://localhost:8000/v1", model="m")
    classifier.classify(_BLANK_IMAGE, "b")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs.get("logprobs") is True
    assert kwargs.get("top_logprobs") == 1
