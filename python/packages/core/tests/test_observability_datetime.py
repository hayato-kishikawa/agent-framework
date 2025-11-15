# Copyright (c) Microsoft. All rights reserved.

"""Tests for datetime serialization in observability telemetry.

This module tests that tool call results containing datetime objects are properly
serialized to JSON for OpenTelemetry tracing without raising TypeErrors.

Fixes issue #2219: Function returns with datetime key/value fail due to JSON serialization.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from agent_framework._types import FunctionResultContent
from agent_framework.observability import _to_otel_part


class TestDatetimeSerialization:
    """Test datetime and other non-JSON-serializable types in tool results."""

    def test_tool_result_with_datetime_value(self) -> None:
        """Test that tool results with datetime values are serialized for telemetry.

        Reproduces Issue #2219 scenario 2:
        Tool returns {"Seattle": datetime.today()}
        """
        content = FunctionResultContent(
            call_id="test-call-1",
            result={"Seattle": datetime(2025, 11, 16, 10, 30, 0)},
        )

        result = _to_otel_part(content)

        assert result is not None
        assert result["type"] == "tool_call_response"
        assert result["id"] == "test-call-1"
        # Should not raise TypeError
        assert isinstance(result["response"], str)
        # Response should be valid JSON
        parsed = json.loads(result["response"])
        assert "Seattle" in parsed
        # Datetime becomes ISO 8601 string representation
        assert parsed["Seattle"] == "2025-11-16T10:30:00"

    def test_tool_result_with_datetime_key_raises(self) -> None:
        """Test that datetime keys in tool results still raise TypeError.

        Reproduces Issue #2219 scenario 1:
        Tool returns {datetime.today(): "Seattle"}

        Note: JSON spec does not allow non-string keys, and Python's json.dumps()
        does not support datetime objects as dictionary keys even with default=str.
        This is a known limitation documented in Issue #2219.

        Users should convert datetime keys to strings before returning from tools.
        """
        dt_key = datetime(2025, 11, 16, 10, 30, 0)
        content = FunctionResultContent(
            call_id="test-call-2",
            result={dt_key: "Seattle"},
        )

        # This still raises TypeError - datetime keys are not supported
        with pytest.raises(TypeError, match="keys must be str, int, float, bool or None"):
            _to_otel_part(content)

    def test_tool_result_with_date_object(self) -> None:
        """Test that date objects (not datetime) are also serialized."""
        content = FunctionResultContent(
            call_id="test-call-3",
            result={"today": date(2025, 11, 16)},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert "today" in parsed
        # Date becomes ISO 8601 string representation
        assert parsed["today"] == "2025-11-16"

    def test_tool_result_with_decimal(self) -> None:
        """Test that Decimal objects are serialized preserving precision."""
        content = FunctionResultContent(
            call_id="test-call-4",
            result={"price": Decimal("19.99")},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert "price" in parsed
        # Decimal becomes string representation to preserve precision
        assert parsed["price"] == "19.99"

    def test_tool_result_with_uuid(self) -> None:
        """Test that UUID objects are serialized."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        content = FunctionResultContent(
            call_id="test-call-5",
            result={"id": test_uuid},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert "id" in parsed
        assert parsed["id"] == str(test_uuid)

    def test_tool_result_with_nested_datetime(self) -> None:
        """Test that nested structures with datetime are serialized."""
        content = FunctionResultContent(
            call_id="test-call-6",
            result={
                "data": {
                    "timestamp": datetime(2025, 11, 16, 10, 30, 0),
                    "metadata": {"created_at": datetime(2025, 11, 15, 8, 0, 0)},
                }
            },
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert "data" in parsed
        assert "timestamp" in parsed["data"]
        assert "metadata" in parsed["data"]
        assert "created_at" in parsed["data"]["metadata"]

    def test_tool_result_list_with_datetime(self) -> None:
        """Test that list results containing datetime are serialized."""

        content = FunctionResultContent(
            call_id="test-call-7",
            result=[
                datetime(2025, 11, 16, 10, 0, 0),
                "text",
                123,
                {"date": datetime(2025, 11, 15, 8, 0, 0)},
            ],
        )

        result = _to_otel_part(content)

        assert result is not None
        # Should not raise TypeError
        parsed = json.loads(result["response"])
        assert isinstance(parsed, list)
        assert len(parsed) == 4

    def test_regular_json_serializable_unchanged(self) -> None:
        """Test that regular JSON-serializable results still work correctly."""
        content = FunctionResultContent(
            call_id="test-call-8",
            result={"str": "value", "int": 42, "float": 3.14, "bool": True, "none": None},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert parsed == {"str": "value", "int": 42, "float": 3.14, "bool": True, "none": None}

    def test_datetime_with_timezone(self) -> None:
        """Test that datetime with timezone information is serialized correctly."""
        from datetime import timezone

        dt_with_tz = datetime(2025, 11, 16, 10, 30, 0, tzinfo=timezone.utc)
        content = FunctionResultContent(
            call_id="test-call-9",
            result={"timestamp": dt_with_tz},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert "timestamp" in parsed
        # ISO 8601 format with timezone
        assert parsed["timestamp"] == "2025-11-16T10:30:00+00:00"

    def test_deeply_nested_datetime(self) -> None:
        """Test deeply nested structures (5+ levels) with datetime objects."""
        content = FunctionResultContent(
            call_id="test-call-10",
            result={
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "timestamp": datetime(2025, 11, 16, 10, 30, 0),
                                    "id": UUID("12345678-1234-5678-1234-567812345678"),
                                }
                            }
                        }
                    }
                }
            },
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        level5 = parsed["level1"]["level2"]["level3"]["level4"]["level5"]
        assert level5["timestamp"] == "2025-11-16T10:30:00"
        assert level5["id"] == "12345678-1234-5678-1234-567812345678"

    def test_mixed_types_in_list(self) -> None:
        """Test list containing multiple different types including datetime."""
        content = FunctionResultContent(
            call_id="test-call-11",
            result=[
                datetime(2025, 11, 16, 10, 0, 0),
                date(2025, 11, 15),
                Decimal("99.99"),
                UUID("12345678-1234-5678-1234-567812345678"),
                "string",
                42,
                3.14,
                True,
                None,
                {"nested": datetime(2025, 11, 14, 8, 0, 0)},
            ],
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        assert isinstance(parsed, list)
        assert len(parsed) == 10
        # Verify each type is correctly serialized
        assert parsed[0] == "2025-11-16T10:00:00"  # datetime
        assert parsed[1] == "2025-11-15"  # date
        assert parsed[2] == "99.99"  # Decimal
        assert parsed[3] == "12345678-1234-5678-1234-567812345678"  # UUID
        assert parsed[4] == "string"
        assert parsed[5] == 42
        assert parsed[6] == 3.14
        assert parsed[7] is True
        assert parsed[8] is None
        assert parsed[9]["nested"] == "2025-11-14T08:00:00"

    def test_unsupported_type_raises_helpful_error(self) -> None:
        """Test that unsupported types raise TypeError with helpful message."""

        class CustomClass:
            """Custom class that is not JSON serializable."""

            pass

        content = FunctionResultContent(
            call_id="test-call-12",
            result={"custom": CustomClass()},
        )

        # Should raise TypeError with helpful message
        with pytest.raises(
            TypeError,
            match=r"Object of type CustomClass is not JSON serializable\. "
            r"Supported types: datetime, date, Decimal, UUID\. "
            r"Please convert to a supported type before returning from tool functions\.",
        ):
            _to_otel_part(content)

    def test_decimal_precision_preserved(self) -> None:
        """Test that Decimal precision is preserved in serialization."""
        content = FunctionResultContent(
            call_id="test-call-13",
            result={
                "high_precision": Decimal("0.123456789123456789"),
                "scientific": Decimal("1.23E+10"),
                "small": Decimal("0.00000001"),
            },
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        # Precision is preserved as string
        # Note: str(Decimal) may use scientific notation for very small/large numbers
        assert parsed["high_precision"] == "0.123456789123456789"
        assert parsed["scientific"] == "1.23E+10"
        assert parsed["small"] in ("0.00000001", "1E-8")  # Both are valid representations

    def test_empty_result(self) -> None:
        """Test that empty results are handled correctly."""
        # Empty dict
        content1 = FunctionResultContent(call_id="test-call-14", result={})
        result1 = _to_otel_part(content1)
        assert result1 is not None
        assert json.loads(result1["response"]) == {}

        # Empty list
        content2 = FunctionResultContent(call_id="test-call-15", result=[])
        result2 = _to_otel_part(content2)
        assert result2 is not None
        assert json.loads(result2["response"]) == []

    def test_datetime_microseconds(self) -> None:
        """Test that datetime with microseconds is serialized correctly."""
        dt_with_micro = datetime(2025, 11, 16, 10, 30, 0, 123456)
        content = FunctionResultContent(
            call_id="test-call-16",
            result={"timestamp": dt_with_micro},
        )

        result = _to_otel_part(content)

        assert result is not None
        parsed = json.loads(result["response"])
        # ISO 8601 format with microseconds
        assert parsed["timestamp"] == "2025-11-16T10:30:00.123456"
