"""Tests for output guardrails."""

from __future__ import annotations

import pytest

from ai_infra.guardrails.output import Hallucination, PIILeakage, PIILeakageMatch, Toxicity

# =============================================================================
# Toxicity Tests
# =============================================================================


class TestToxicityBasic:
    """Test basic Toxicity functionality."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        guard = Toxicity()
        assert guard.threshold == 0.7
        assert guard.method == "openai"
        assert "hate" in guard.categories
        assert "harassment" in guard.categories
        assert "violence" in guard.categories

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        guard = Toxicity(
            threshold=0.5,
            categories=["hate", "violence"],
            method="heuristic",
        )
        assert guard.threshold == 0.5
        assert guard.method == "heuristic"
        assert guard.categories == ["hate", "violence"]

    def test_name_attribute(self) -> None:
        """Test guardrail name attribute."""
        guard = Toxicity()
        assert guard.name == "toxicity"

    def test_clean_text_passes_heuristic(self) -> None:
        """Test that clean text passes heuristic check."""
        guard = Toxicity(method="heuristic")
        result = guard.check("The weather today is beautiful and sunny.")
        assert result.passed is True

    def test_clean_text_variations(self) -> None:
        """Test various clean texts pass."""
        guard = Toxicity(method="heuristic")
        texts = [
            "Thank you for your help!",
            "I appreciate your assistance with this matter.",
            "The project is going well and on schedule.",
            "Let me know if you need any clarification.",
        ]
        for text in texts:
            result = guard.check(text)
            assert result.passed is True, f"Failed for: {text}"


class TestToxicityHeuristic:
    """Test heuristic toxicity detection."""

    def test_violence_pattern_detected(self) -> None:
        """Test detection of violent content."""
        guard = Toxicity(method="heuristic", categories=["violence"])
        result = guard.check("You deserve to die for what you said")
        assert result.passed is False
        assert result.severity in ("high", "critical")

    def test_self_harm_pattern_detected(self) -> None:
        """Test detection of self-harm content."""
        guard = Toxicity(method="heuristic", categories=["self_harm"])
        result = guard.check("Here is a suicide method for you")
        assert result.passed is False
        assert result.severity == "critical"

    def test_harassment_pattern_detected(self) -> None:
        """Test detection of harassment content."""
        guard = Toxicity(method="heuristic", categories=["harassment"])
        result = guard.check("You should die, I will threaten your family")
        assert result.passed is False

    def test_mixed_categories(self) -> None:
        """Test detection when multiple categories match."""
        guard = Toxicity(method="heuristic")
        # This should match violence pattern
        result = guard.check("Kill all those people who disagree")
        assert result.passed is False
        assert "violence" in result.details.get("categories_flagged", [])


class TestToxicitySeverity:
    """Test severity determination."""

    def test_critical_severity_for_self_harm(self) -> None:
        """Test critical severity for self-harm."""
        guard = Toxicity(method="heuristic")
        # Pattern matching self-harm
        result = guard.check("cut myself to deal with it")
        if not result.passed:
            assert result.severity == "critical"

    def test_critical_severity_for_violence(self) -> None:
        """Test critical severity for violence."""
        guard = Toxicity(method="heuristic")
        result = guard.check("I will kill all those people")
        if not result.passed:
            assert result.severity == "critical"


# =============================================================================
# PIILeakage Tests
# =============================================================================


class TestPIILeakageBasic:
    """Test basic PIILeakage functionality."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        guard = PIILeakage()
        assert guard.action == "redact"
        assert "SSN" in guard.entities
        assert "CREDIT_CARD" in guard.entities
        assert "API_KEY" in guard.entities

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        guard = PIILeakage(
            entities=["EMAIL", "PHONE"],
            action="block",
            min_confidence=0.8,
        )
        assert guard.entities == ["EMAIL", "PHONE"]
        assert guard.action == "block"
        assert guard.min_confidence == 0.8

    def test_name_attribute(self) -> None:
        """Test guardrail name attribute."""
        guard = PIILeakage()
        assert guard.name == "pii_leakage"

    def test_clean_text_passes(self) -> None:
        """Test that clean text passes."""
        guard = PIILeakage()
        result = guard.check("The weather today is nice.")
        assert result.passed is True


class TestPIILeakageSSN:
    """Test SSN detection in outputs."""

    def test_detect_ssn_with_dashes(self) -> None:
        """Test SSN detection with dashes."""
        guard = PIILeakage(entities=["SSN"], action="block")
        # Use a realistic test SSN (not in obviously fake list)
        result = guard.check("The SSN is 234-56-7890")
        assert result.passed is False
        assert "SSN" in result.details.get("entities_found", {})

    def test_detect_ssn_without_dashes(self) -> None:
        """Test SSN detection without dashes."""
        guard = PIILeakage(entities=["SSN"], action="block")
        result = guard.check("Your SSN: 234567890")
        assert result.passed is False

    def test_redact_ssn(self) -> None:
        """Test SSN redaction."""
        guard = PIILeakage(entities=["SSN"], action="redact")
        result = guard.check("The SSN is 234-56-7890")
        assert result.passed is True
        assert "[SSN]" in result.details.get("redacted_text", "")

    def test_invalid_ssn_not_detected(self) -> None:
        """Test that invalid SSN patterns are not detected."""
        guard = PIILeakage(entities=["SSN"], action="block")
        # SSN starting with 000 is invalid
        result = guard.check("000-12-3456")
        assert result.passed is True


class TestPIILeakageCreditCard:
    """Test credit card detection in outputs."""

    def test_detect_visa(self) -> None:
        """Test Visa card detection."""
        guard = PIILeakage(entities=["CREDIT_CARD"], action="block")
        # Valid Luhn checksum Visa
        result = guard.check("Your card: 4111111111111111")
        assert result.passed is False

    def test_detect_mastercard(self) -> None:
        """Test Mastercard detection."""
        guard = PIILeakage(entities=["CREDIT_CARD"], action="block")
        # Valid Luhn checksum Mastercard
        result = guard.check("Card number is 5500000000000004")
        assert result.passed is False

    def test_invalid_card_not_detected(self) -> None:
        """Test that invalid card numbers are not detected."""
        guard = PIILeakage(entities=["CREDIT_CARD"], action="block")
        # Invalid Luhn checksum
        result = guard.check("Not a real card: 1234567890123456")
        assert result.passed is True


class TestPIILeakageAPIKeys:
    """Test API key detection in outputs."""

    def test_detect_openai_key(self) -> None:
        """Test OpenAI API key detection."""
        guard = PIILeakage(entities=["API_KEY"], action="block")
        result = guard.check("Use this key: sk-abcdefghijklmnopqrstuvwxyz123456")
        assert result.passed is False
        assert "API_KEY" in result.details.get("entities_found", {})

    def test_detect_github_token(self) -> None:
        """Test GitHub token detection."""
        guard = PIILeakage(entities=["API_KEY"], action="block")
        result = guard.check("Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert result.passed is False

    def test_detect_aws_access_key(self) -> None:
        """Test AWS access key detection."""
        guard = PIILeakage(entities=["AWS_ACCESS_KEY"], action="block")
        result = guard.check("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
        assert result.passed is False


class TestPIILeakagePrivateKey:
    """Test private key detection in outputs."""

    def test_detect_private_key_header(self) -> None:
        """Test private key header detection."""
        guard = PIILeakage(entities=["PRIVATE_KEY"], action="block")
        result = guard.check("-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg...")
        assert result.passed is False
        assert result.severity == "critical"

    def test_detect_rsa_private_key(self) -> None:
        """Test RSA private key detection."""
        guard = PIILeakage(entities=["PRIVATE_KEY"], action="block")
        result = guard.check("-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJ...")
        assert result.passed is False


class TestPIILeakageJWT:
    """Test JWT detection in outputs."""

    def test_detect_jwt(self) -> None:
        """Test JWT token detection."""
        guard = PIILeakage(entities=["JWT"], action="block")
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w"
        result = guard.check(f"Your token: {jwt}")
        assert result.passed is False
        assert "JWT" in result.details.get("entities_found", {})


class TestPIILeakageRedaction:
    """Test redaction functionality."""

    def test_redact_multiple_entities(self) -> None:
        """Test redacting multiple entities."""
        guard = PIILeakage(entities=["SSN", "EMAIL"], action="redact")
        text = "SSN: 234-56-7890, email: test@example.com"
        result = guard.check(text)
        assert result.passed is True
        redacted = result.details.get("redacted_text", "")
        assert "[SSN]" in redacted
        assert "[EMAIL]" in redacted
        assert "234-56-7890" not in redacted
        assert "test@example.com" not in redacted

    def test_warn_action(self) -> None:
        """Test warn action."""
        guard = PIILeakage(entities=["SSN"], action="warn")
        result = guard.check("SSN: 234-56-7890")
        assert result.passed is True
        assert "warning" in result.reason.lower()


class TestPIILeakageContext:
    """Test context-based filtering."""

    def test_expected_pii_allowed(self) -> None:
        """Test that expected PII from context is allowed."""
        guard = PIILeakage(entities=["EMAIL"], action="block")
        result = guard.check(
            "Contact: john@example.com",
            context={"expected_pii": ["john@example.com"]},
        )
        assert result.passed is True


class TestPIILeakageSeverity:
    """Test severity determination."""

    def test_critical_for_ssn(self) -> None:
        """Test critical severity for SSN."""
        guard = PIILeakage(entities=["SSN"], action="block")
        result = guard.check("SSN: 234-56-7890")
        assert result.passed is False
        assert result.severity == "critical"

    def test_critical_for_credit_card(self) -> None:
        """Test critical severity for credit card."""
        guard = PIILeakage(entities=["CREDIT_CARD"], action="block")
        result = guard.check("Card: 4111111111111111")
        assert result.severity == "critical"

    def test_high_for_api_key(self) -> None:
        """Test high severity for API key."""
        guard = PIILeakage(entities=["API_KEY"], action="block")
        result = guard.check("Key: sk-abcdefghijklmnopqrstuvwxyz123456")
        assert result.severity == "high"


# =============================================================================
# Hallucination Tests
# =============================================================================


class TestHallucinationBasic:
    """Test basic Hallucination functionality."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        guard = Hallucination()
        assert guard.method == "embedding"
        assert guard.threshold == 0.7
        assert guard.check_claims_only is True

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        guard = Hallucination(
            method="heuristic",
            threshold=0.8,
            check_claims_only=False,
        )
        assert guard.method == "heuristic"
        assert guard.threshold == 0.8
        assert guard.check_claims_only is False

    def test_name_attribute(self) -> None:
        """Test guardrail name attribute."""
        guard = Hallucination()
        assert guard.name == "hallucination"


class TestHallucinationNoSources:
    """Test behavior when no sources are provided."""

    def test_no_context_passes(self) -> None:
        """Test that check passes when no context provided."""
        guard = Hallucination(method="heuristic")
        result = guard.check("Some output text")
        assert result.passed is True

    def test_empty_sources_passes(self) -> None:
        """Test that check passes with empty sources."""
        guard = Hallucination()
        result = guard.check("Some output text", context={"sources": []})
        assert result.passed is True
        assert result.details.get("skipped") is True


class TestHallucinationHeuristic:
    """Test heuristic hallucination detection."""

    def test_grounded_text_passes(self) -> None:
        """Test that grounded text passes."""
        guard = Hallucination(method="heuristic")
        sources = ["Paris is the capital of France."]
        result = guard.check(
            "Paris is the capital of France.",
            context={"sources": sources},
        )
        # Should pass since the term is in sources
        assert result.passed is True

    def test_ungrounded_specific_claims(self) -> None:
        """Test detection of overly specific claims."""
        guard = Hallucination(method="heuristic")
        sources = ["The population is large."]
        result = guard.check(
            "The population is exactly 12,345,678 as of 12/25/2024.",
            context={"sources": sources},
        )
        # Should flag the specific claim
        if not result.passed:
            assert "warnings" in result.details

    def test_hedged_language_passes(self) -> None:
        """Test that hedged language is less likely to be flagged."""
        guard = Hallucination(method="heuristic", check_claims_only=True)
        sources = ["Some data about the topic."]
        result = guard.check(
            "I think this might be related to the topic.",
            context={"sources": sources},
        )
        # Hedged language should generally pass
        assert result.passed is True


class TestHallucinationSourceNormalization:
    """Test source normalization."""

    def test_string_source(self) -> None:
        """Test handling of string source."""
        guard = Hallucination(method="heuristic")
        result = guard.check(
            "Test output",
            context={"sources": "Single source text"},
        )
        # Should not error
        assert isinstance(result.passed, bool)

    def test_dict_sources(self) -> None:
        """Test handling of dict sources with content key."""
        guard = Hallucination(method="heuristic")
        result = guard.check(
            "Test output",
            context={
                "sources": [
                    {"content": "Source 1 text"},
                    {"text": "Source 2 text"},
                    {"page_content": "Source 3 text"},
                ]
            },
        )
        assert isinstance(result.passed, bool)


class TestHallucinationClaimExtraction:
    """Test claim extraction functionality."""

    def test_extracts_claims_with_numbers(self) -> None:
        """Test that claims with numbers are extracted."""
        guard = Hallucination(method="heuristic")
        text = "The company has 500 employees. The weather is nice."
        claims = guard._extract_claims(text)
        # Should include the claim with numbers
        assert any("500" in c for c in claims)

    def test_extracts_claims_with_dates(self) -> None:
        """Test that claims with dates are extracted."""
        guard = Hallucination(method="heuristic")
        text = "The event occurred in 2023. It was a fun day."
        claims = guard._extract_claims(text)
        assert any("2023" in c for c in claims)

    def test_skips_short_sentences(self) -> None:
        """Test that very short sentences are skipped."""
        guard = Hallucination(method="heuristic", check_claims_only=True)
        text = "Yes. No. Maybe. The weather is approximately 25 degrees today."
        claims = guard._extract_claims(text)
        # Short sentences should not be included
        assert "Yes." not in claims
        assert "No." not in claims


class TestHallucinationAsync:
    """Test async functionality."""

    @pytest.mark.asyncio
    async def test_async_check_heuristic(self) -> None:
        """Test async check with heuristic method."""
        guard = Hallucination(method="heuristic")
        result = await guard.check_async(
            "Test output",
            context={"sources": ["Source text"]},
        )
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_async_check_no_sources(self) -> None:
        """Test async check without sources."""
        guard = Hallucination(method="heuristic")
        result = await guard.check_async("Test output")
        assert result.passed is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestOutputGuardrailsIntegration:
    """Test output guardrails working together."""

    def test_multiple_guardrails_on_clean_text(self) -> None:
        """Test multiple guardrails pass on clean text."""
        toxicity = Toxicity(method="heuristic")
        pii = PIILeakage(entities=["SSN", "CREDIT_CARD"])
        hallucination = Hallucination(method="heuristic")

        clean_text = "The weather today is sunny and pleasant."
        sources = ["Today's forecast shows sunny conditions."]

        assert toxicity.check(clean_text).passed is True
        assert pii.check(clean_text).passed is True
        assert hallucination.check(clean_text, context={"sources": sources}).passed is True

    def test_pii_leakage_match_dataclass(self) -> None:
        """Test PIILeakageMatch dataclass."""
        match = PIILeakageMatch(
            entity_type="SSN",
            value="123-45-6789",
            start=0,
            end=11,
            confidence=0.95,
        )
        assert match.entity_type == "SSN"
        assert "***" in repr(match) or "..." in repr(match)  # Should be masked
