"""Pinned stoplist for the deterministic lexical signal.

Frozen at construction time so re-ingest of the same corpus produces a
byte-identical event log. Do not extend casually — every change shifts
the lexical signal across the entire benchmark.
"""

from __future__ import annotations

# Standard English function words and conversational glue. Lowercased,
# length-agnostic (the lexical builder enforces min_token_length>=4 on top
# of this list, so 1-3 char tokens are already filtered).
STOPLIST: frozenset[str] = frozenset(
    {
        "about", "above", "after", "again", "against", "all", "also", "always",
        "another", "any", "anyone", "anything", "are", "aren", "around", "back",
        "because", "been", "before", "being", "below", "between", "both", "but",
        "came", "can", "cannot", "could", "couldn", "did", "didn", "does", "doesn",
        "doing", "don", "done", "down", "during", "each", "either", "else", "even",
        "ever", "every", "feel", "felt", "for", "from", "get", "gets", "getting",
        "give", "given", "gives", "going", "gone", "got", "had", "hadn", "has",
        "hasn", "have", "haven", "having", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "however", "into", "isn", "its", "itself", "just",
        "knew", "know", "known", "knows", "least", "less", "like", "liked", "likes",
        "look", "looked", "looking", "looks", "made", "make", "makes", "making",
        "many", "may", "maybe", "might", "more", "most", "much", "must", "myself",
        "need", "needed", "needs", "never", "next", "not", "nothing", "now", "off",
        "often", "okay", "once", "one", "only", "other", "others", "our", "ours",
        "ourselves", "out", "over", "own", "perhaps", "really", "right", "same",
        "saw", "say", "saying", "says", "see", "seeing", "seems", "seen", "shall",
        "shan", "she", "should", "shouldn", "since", "some", "someone", "something",
        "sometimes", "soon", "still", "such", "take", "taken", "takes", "taking",
        "tell", "telling", "tells", "than", "thank", "thanks", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "thing",
        "things", "think", "thinking", "thinks", "this", "those", "though", "through",
        "thus", "tried", "tries", "true", "try", "trying", "under", "until", "very",
        "want", "wanted", "wants", "was", "wasn", "way", "well", "went", "were",
        "weren", "what", "whatever", "when", "where", "whether", "which", "while",
        "who", "whom", "whose", "why", "will", "with", "within", "without", "won",
        "would", "wouldn", "yes", "yet", "you", "your", "yours", "yourself",
        "yourselves",
        # Common conversational shorthand
        "user", "assistant", "session",
    }
)
