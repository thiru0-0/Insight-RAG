"""Tests for query engine: synonym expansion, coreference resolution, chat memory."""

import time
import pytest
from src.query_engine import (
    rewrite_query,
    ChatMemory,
    _extract_content_words,
    _extract_topic_from_history,
    _has_pronoun_reference,
    _PRONOUNS,
)


# ── Pronoun Set ──────────────────────────────────────────────────────

class TestPronounSet:
    def test_common_pronouns_present(self):
        for p in ["it", "they", "this", "that", "them", "its"]:
            assert p in _PRONOUNS

    def test_there_here_removed(self):
        """'there' and 'here' should NOT be in the pronoun set (they cause garbling)."""
        assert "there" not in _PRONOUNS
        assert "here" not in _PRONOUNS


# ── Pronoun Detection ────────────────────────────────────────────────

class TestHasPronounReference:
    def test_short_pronoun_query(self):
        assert _has_pronoun_reference("Tell me more about it")

    def test_no_pronoun(self):
        assert not _has_pronoun_reference("What is machine learning?")

    def test_there_not_detected(self):
        """'there' should not trigger coreference rewriting."""
        assert not _has_pronoun_reference("Is there a refund policy?")

    def test_this_detected(self):
        assert _has_pronoun_reference("What is this?")


# ── Topic Extraction ─────────────────────────────────────────────────

class TestTopicExtraction:
    def test_empty_history(self):
        assert _extract_topic_from_history([]) == ""

    def test_extracts_from_question(self):
        history = [{"q": "What is machine learning?", "a": "Machine learning is a subset of AI."}]
        topic = _extract_topic_from_history(history)
        assert len(topic) > 0
        assert "machine" in topic.lower()

    def test_limits_word_count(self):
        history = [
            {"q": "Tell me about artificial intelligence and deep neural network architectures and transformers",
             "a": "AI encompasses many approaches including deep learning and transformer models."}
        ]
        topic = _extract_topic_from_history(history)
        words = topic.split()
        assert len(words) <= 4, f"Topic too long ({len(words)} words): {topic}"


# ── Coreference Resolution ──────────────────────────────────────────

class TestCoreferenceResolution:
    def test_replaces_pronoun_with_topic(self):
        history = [{"q": "What is machine learning?", "a": "It is a subset of AI."}]
        result = rewrite_query("Tell me more about it", history=history, expand_synonyms=False)
        assert result["was_rewritten"]
        # The pronoun "it" should be replaced with topic words
        assert "it" not in result["rewritten"].lower().split()

    def test_only_first_pronoun_replaced(self):
        """Only the FIRST pronoun should be replaced, not all of them."""
        history = [{"q": "What is the contract?", "a": "It covers services."}]
        result = rewrite_query("What is it and how does it work?", history=history, expand_synonyms=False)
        if result["was_rewritten"]:
            # Count how many times the topic appears — should be once
            topic = _extract_topic_from_history(history)
            if topic:
                count = result["rewritten"].lower().count(topic.lower())
                assert count == 1, f"Topic '{topic}' appears {count} times in: {result['rewritten']}"

    def test_no_rewrite_without_pronoun(self):
        history = [{"q": "What is AI?", "a": "Artificial intelligence."}]
        result = rewrite_query("What is machine learning?", history=history, expand_synonyms=False)
        assert not result["was_rewritten"] or result["reason"].startswith("Expanded")

    def test_no_rewrite_without_history(self):
        result = rewrite_query("Tell me about it", history=None, expand_synonyms=False)
        assert not result["was_rewritten"]


# ── Synonym Expansion ────────────────────────────────────────────────

class TestSynonymExpansion:
    def test_expands_known_synonym(self):
        result = rewrite_query("What is termination?", history=None, expand_synonyms=True)
        assert len(result["expanded_terms"]) > 0
        assert result["was_rewritten"]

    def test_no_expansion_for_unknown(self):
        result = rewrite_query("What is photosynthesis?", history=None, expand_synonyms=True)
        assert len(result["expanded_terms"]) == 0

    def test_nda_expansion(self):
        result = rewrite_query("Explain the NDA terms", history=None, expand_synonyms=True)
        assert any("non-disclosure" in t or "confidential" in t for t in result["expanded_terms"])

    def test_disabled_expansion(self):
        result = rewrite_query("What is termination?", history=None, expand_synonyms=False)
        assert len(result["expanded_terms"]) == 0

    def test_result_structure(self):
        result = rewrite_query("Hello world", history=None)
        assert "original" in result
        assert "rewritten" in result
        assert "display_query" in result
        assert "expanded_terms" in result
        assert "was_rewritten" in result
        assert "reason" in result


# ── Chat Memory ──────────────────────────────────────────────────────

class TestChatMemory:
    def test_create_session(self):
        mem = ChatMemory()
        sid = mem.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_add_and_get(self):
        mem = ChatMemory()
        sid = mem.create_session()
        mem.add_turn(sid, "What is AI?", "AI is artificial intelligence.")
        history = mem.get_history(sid)
        assert len(history) == 1
        assert history[0]["q"] == "What is AI?"
        assert history[0]["a"] == "AI is artificial intelligence."

    def test_max_turns_limit(self):
        mem = ChatMemory()
        sid = mem.create_session()
        for i in range(15):
            mem.add_turn(sid, f"Q{i}", f"A{i}")
        history = mem.get_history(sid)
        assert len(history) == ChatMemory.MAX_TURNS

    def test_clear_session(self):
        mem = ChatMemory()
        sid = mem.create_session()
        mem.add_turn(sid, "Q", "A")
        mem.clear_session(sid)
        assert mem.get_history(sid) == []

    def test_auto_create_on_add(self):
        mem = ChatMemory()
        mem.add_turn("nonexistent", "Q", "A")
        history = mem.get_history("nonexistent")
        assert len(history) == 1

    def test_max_sessions_eviction(self):
        mem = ChatMemory()
        mem.MAX_SESSIONS = 5  # lower for test
        sids = []
        for _ in range(7):
            sids.append(mem.create_session())
        # Oldest sessions should have been evicted
        assert len(mem._sessions) <= 5

    def test_empty_history_for_unknown_session(self):
        mem = ChatMemory()
        assert mem.get_history("unknown_id") == []
