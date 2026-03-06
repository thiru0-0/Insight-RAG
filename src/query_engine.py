"""
Query Rewriting & Chat Memory Module
- Rewrites ambiguous queries using conversation history (coreference resolution)
- Expands queries with synonyms for better retrieval recall
- Maintains per-session conversation memory
"""

import re
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  CHAT MEMORY
# ═══════════════════════════════════════════════════════════════════════

class ChatMemory:
    """
    Server-side conversation memory with session management.
    Stores the last N turns per session for context carryover.
    """

    MAX_TURNS = 10          # keep last 10 Q&A pairs per session
    MAX_SESSIONS = 200      # evict oldest when exceeded
    SESSION_TTL = 3600      # 1 hour time-to-live

    def __init__(self):
        # session_id → { "turns": [...], "last_access": float }
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def create_session(self) -> str:
        """Create a new chat session and return its ID."""
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = {"turns": [], "last_access": time.time()}
        self._evict()
        return sid

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Append a Q&A turn to the session."""
        session = self._sessions.get(session_id)
        if session is None:
            # Auto-create if missing
            self._sessions[session_id] = {"turns": [], "last_access": time.time()}
            session = self._sessions[session_id]

        session["turns"].append({"q": question, "a": answer})
        # Trim to MAX_TURNS
        if len(session["turns"]) > self.MAX_TURNS:
            session["turns"] = session["turns"][-self.MAX_TURNS:]
        session["last_access"] = time.time()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation turns for this session."""
        session = self._sessions.get(session_id)
        if session is None:
            return []
        session["last_access"] = time.time()
        return list(session["turns"])

    def clear_session(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def _evict(self) -> None:
        """Remove expired sessions and enforce MAX_SESSIONS."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["last_access"] > self.SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]

        while len(self._sessions) > self.MAX_SESSIONS:
            self._sessions.popitem(last=False)  # remove oldest


# ═══════════════════════════════════════════════════════════════════════
#  QUERY REWRITER
# ═══════════════════════════════════════════════════════════════════════

# Pronouns and demonstratives that likely refer to prior context
_PRONOUNS = frozenset({
    "it", "its", "they", "them", "their", "theirs",
    "he", "him", "his", "she", "her", "hers",
    "this", "that", "these", "those",
})

# Common question words that should not be treated as content
_QUESTION_WORDS = frozenset({
    "what", "which", "how", "when", "where", "who", "why",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might",
    "tell", "me", "about", "explain", "describe", "show",
})

# Synonym groups for query expansion
_SYNONYM_MAP = {
    "termination": ["terminate", "end", "cancel", "cancellation"],
    "terminate": ["termination", "end", "cancel"],
    "agreement": ["contract", "deal", "arrangement"],
    "contract": ["agreement", "deal", "arrangement"],
    "confidential": ["confidentiality", "secret", "proprietary", "nda"],
    "nda": ["non-disclosure", "confidentiality", "confidential"],
    "liability": ["liable", "responsibility", "obligation"],
    "indemnification": ["indemnify", "indemnity", "compensation"],
    "establish": ["established", "founded", "created", "started"],
    "founded": ["established", "created", "started", "founding"],
    "located": ["location", "situated", "based", "address"],
    "location": ["located", "situated", "based", "address", "place"],
    "affiliate": ["affiliated", "affiliation", "associated", "association"],
    "affiliation": ["affiliated", "affiliate", "associated", "association"],
    "college": ["university", "institution", "school", "institute"],
    "university": ["college", "institution", "school", "institute"],
}


def _extract_content_words(text: str) -> List[str]:
    """Extract meaningful content words from text."""
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    extra_stop = {
        "a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "to",
        "by", "at", "from", "into", "up", "out", "than", "then", "also", "just",
        "more", "most", "some", "such", "very", "much", "only", "even", "still",
        "study", "programs", "given", "task", "automatically", "performance",
        "several", "kinds", "based", "used", "using", "has", "have", "had",
        "been", "being", "its", "other", "new", "first", "second", "third",
    }
    return [w for w in words if w not in _QUESTION_WORDS and w not in extra_stop and len(w) > 2]


def _has_pronoun_reference(query: str) -> bool:
    """Check if query contains pronouns that likely refer to prior context."""
    words = set(re.sub(r"[^a-z\s]", " ", query.lower()).split())
    content_words = words - _QUESTION_WORDS - {"a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "to"}
    # If the query has very few content words and contains a pronoun, it's referential
    has_pronoun = bool(words & _PRONOUNS)
    if has_pronoun and len(content_words) <= 4:
        return True
    return False


def _extract_topic_from_history(history: List[Dict[str, str]]) -> str:
    """Extract the main topic/entity from recent conversation history."""
    if not history:
        return ""

    # Look at the last 3 turns, most recent first
    recent = history[-3:]

    # Collect nouns/entities from recent questions and answers
    topic_words = []
    for turn in reversed(recent):
        q_words = _extract_content_words(turn["q"])
        # Take content words from the question (most likely the subject)
        topic_words.extend(q_words[:5])
        # Also check the answer for entities
        a_words = _extract_content_words(turn["a"])
        topic_words.extend(a_words[:3])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in topic_words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return " ".join(unique[:4])


def rewrite_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    expand_synonyms: bool = True,
) -> Dict[str, Any]:
    """
    Rewrite a query for better retrieval.

    Returns:
        {
            "original": str,
            "rewritten": str,
            "expanded_terms": list[str],
            "was_rewritten": bool,
            "reason": str,
        }
    """
    original = query.strip()
    rewritten = original
    expanded_terms = []
    was_rewritten = False
    reason = ""

    # ── Step 1: Coreference resolution via chat history ──────────
    if history and _has_pronoun_reference(original):
        topic = _extract_topic_from_history(history)
        if topic:
            # Replace only the FIRST pronoun occurrence with the topic
            rewritten_parts = []
            replaced = False
            for word in original.split():
                w_lower = word.lower().strip(".,!?;:")
                if not replaced and w_lower in _PRONOUNS:
                    # Preserve trailing punctuation from the original word
                    trailing = word[len(w_lower):] if len(word) > len(w_lower) else ""
                    rewritten_parts.append(topic + trailing)
                    replaced = True
                else:
                    rewritten_parts.append(word)
            candidate = " ".join(rewritten_parts)

            # Only rewrite if it's actually different
            if candidate.lower() != original.lower():
                rewritten = candidate
                was_rewritten = True
                reason = f"Resolved pronoun reference using conversation context"

    # ── Step 2: Synonym expansion ────────────────────────────────
    if expand_synonyms:
        query_words = re.sub(r"[^a-z0-9\s]", " ", rewritten.lower()).split()
        for word in query_words:
            if word in _SYNONYM_MAP:
                synonyms = _SYNONYM_MAP[word]
                expanded_terms.extend(synonyms[:2])  # add top 2 synonyms

        # Deduplicate expanded terms and remove any already in query
        existing = set(re.sub(r"[^a-z0-9\s]", " ", rewritten.lower()).split())
        expanded_terms = list(dict.fromkeys(t for t in expanded_terms if t not in existing))

        if expanded_terms:
            if not was_rewritten:
                reason = "Expanded with synonym terms"
            else:
                reason += "; expanded with synonym terms"
            was_rewritten = True

    # ── Step 3: Build final search query ─────────────────────────
    # The expanded terms are appended to the rewritten query for embedding search
    if expanded_terms:
        search_query = rewritten + " " + " ".join(expanded_terms)
    else:
        search_query = rewritten

    return {
        "original": original,
        "rewritten": search_query.strip(),
        "display_query": rewritten,  # human-readable version (without synonym noise)
        "expanded_terms": expanded_terms,
        "was_rewritten": was_rewritten,
        "reason": reason if reason else "No rewriting needed",
    }
