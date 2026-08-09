"""The TFT output columns are named by the encoder, never by position.

`NaNLabelEncoder` numbers its classes by sorting the labels, so it yields
`{'nan': 0, 'long': 1, 'short': 2}` — alphabetical. `DIRECTION_CLASSES` is
`['short', 'long']`, because the sign of the forward return defines it. Reading
the model's output positionally after dropping the leading `nan` column
therefore lined `short` up with the `long` logit: every TFT run written before
this was fixed carried the two probabilities exchanged, and its edge was the
negation of what the model had learned.

The defect is invisible to every aggregate the project reports. A swapped
two-class forecast keeps its entropy, its confidence and its Brier score
against a balanced outcome; only accuracy moves, and it moves to 1 - accuracy,
which near a coin flip is indistinguishable from noise. Trained on a synthetic
series whose direction is a deterministic function of the previous bar, the
assembled labels agreed with the truth on 0.013 of samples instead of 0.987.
"""

from __future__ import annotations

import numpy as np
import pytest
from prosper.domain import DIRECTION_CLASSES
from prosper.predict.tft import _class_columns, _logits_to_probs


class _Dataset:
    """Minimal stand-in exposing what `_class_columns` is allowed to read."""

    def __init__(self, classes: dict[str, int]) -> None:
        self.target_normalizer = type("N", (), {"classes_": classes})()


def test_columns_follow_the_encoder_not_the_declaration_order() -> None:
    columns = _class_columns(_Dataset({"nan": 0, "long": 1, "short": 2}))

    assert columns == [2, 1], "short is column 2 and long is column 1"
    assert [DIRECTION_CLASSES[i] for i in range(len(columns))] == ["short", "long"]


def test_an_encoder_without_a_nan_class_still_maps_correctly() -> None:
    assert _class_columns(_Dataset({"long": 0, "short": 1})) == [1, 0]


def test_a_declaration_ordered_encoder_is_the_identity() -> None:
    """Nothing here assumes the encoder disagrees — only that it is asked."""
    assert _class_columns(_Dataset({"nan": 0, "short": 1, "long": 2})) == [1, 2]


def test_numpy_string_keys_are_accepted() -> None:
    """`NaNLabelEncoder` stores `np.str_` keys once fitted on a numpy array."""
    classes = {"nan": 0, np.str_("long"): 1, np.str_("short"): 2}
    assert _class_columns(_Dataset(classes)) == [2, 1]


def test_a_dataset_with_no_encoder_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="target_normalizer"):
        _class_columns(_Dataset({}))


def test_a_missing_direction_class_is_an_error() -> None:
    with pytest.raises(ValueError, match="absent from the encoder"):
        _class_columns(_Dataset({"nan": 0, "long": 1}))


def test_the_probability_follows_the_named_column() -> None:
    """The logit that is largest must end up on the class the encoder names."""
    # Column 1 is `long`, column 2 is `short`; the model shouts `short`.
    logits = np.array([-9.0, 0.0, 5.0])
    probs = _logits_to_probs(logits, _class_columns(_Dataset({"nan": 0, "long": 1, "short": 2})))
    named = dict(zip(DIRECTION_CLASSES, probs, strict=True))

    assert named["short"] > named["long"]
    assert sum(probs) == pytest.approx(1.0)


def test_the_nan_column_carries_no_mass() -> None:
    """Selecting after the softmax and renormalising drops it exactly."""
    logits = np.array([100.0, 1.0, 2.0])
    probs = _logits_to_probs(logits, [2, 1])

    assert sum(probs) == pytest.approx(1.0)
    # Unaffected by the huge nan logit: still softmax([2.0, 1.0]).
    assert probs[0] == pytest.approx(np.exp(1.0) / (np.exp(1.0) + 1.0))


def test_a_degenerate_row_falls_back_to_uniform() -> None:
    probs = _logits_to_probs(np.array([np.nan, np.nan, np.nan]), [2, 1])
    assert probs == pytest.approx([0.5, 0.5])


def test_the_encoder_really_does_sort_its_classes() -> None:
    """Pins the upstream behaviour the fix exists for.

    If pytorch-forecasting ever stops sorting, this fails and the comment above
    stops being true — which is worth knowing immediately.
    """
    pytest.importorskip("pytorch_forecasting")
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    encoder = NaNLabelEncoder(add_nan=True)
    encoder.fit(np.array(["short", "long", "long", "short"]))

    assert {str(k): v for k, v in encoder.classes_.items()} == {"nan": 0, "long": 1, "short": 2}
