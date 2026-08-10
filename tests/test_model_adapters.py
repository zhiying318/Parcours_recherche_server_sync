import pytest

from geometry_teacher.model_adapters import normalize_grounding_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("person", "person."),
        (" person. ", "person."),
        ("car...", "car."),
        ("fire hydrant", "fire hydrant."),
    ],
)
def test_normalize_grounding_query(query, expected):
    assert normalize_grounding_query(query) == expected


@pytest.mark.parametrize("query", ["", "   ", "..."])
def test_normalize_grounding_query_rejects_empty(query):
    with pytest.raises(ValueError, match="must be non-empty"):
        normalize_grounding_query(query)
