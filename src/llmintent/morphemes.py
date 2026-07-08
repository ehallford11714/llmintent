"""Morpheme and lemma extraction backends from the notebook."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

Backend = Literal["stanza", "spacy", "polyglot", "lemma"]


class MorphemeExtractor:
    """Extract linguistic units from word lists using optional NLP backends."""

    def __init__(self, backend: Backend = "lemma") -> None:
        self.backend = backend
        self._stanza = None
        self._spacy = None

    def extract(self, words: Iterable[str]) -> list[str]:
        word_list = [w.strip().lower() for w in words if w.strip()]
        if not word_list:
            return []
        if self.backend == "stanza":
            return self._extract_stanza(word_list)
        if self.backend == "spacy":
            return self._extract_spacy(word_list)
        if self.backend == "polyglot":
            return self._extract_polyglot(word_list)
        return word_list

    def extract_support(self, words: Iterable[str]) -> list[str]:
        """spaCy POS/dependency support tokens (notebook: extract_spacy_support)."""
        if self.backend != "spacy":
            return self.extract(words)
        nlp = self._load_spacy()
        units: list[str] = []
        for word in words:
            doc = nlp(word)
            for token in doc:
                if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
                    units.append(token.lemma_.lower())
                if token.dep_ in {"nsubj", "dobj", "pobj"}:
                    units.append(token.text.lower())
        return units

    def top_themes(self, words: Iterable[str], top_k: int = 5) -> list[tuple[str, int]]:
        counts = Counter(self.extract(words))
        return counts.most_common(top_k)

    def _load_stanza(self):
        if self._stanza is None:
            import stanza

            try:
                self._stanza = stanza.Pipeline("en", processors="tokenize,pos,lemma", verbose=False)
            except Exception:
                stanza.download("en")
                self._stanza = stanza.Pipeline("en", processors="tokenize,pos,lemma", verbose=False)
        return self._stanza

    def _load_spacy(self):
        if self._spacy is None:
            import spacy

            try:
                self._spacy = spacy.load("en_core_web_sm")
            except OSError as exc:
                raise RuntimeError(
                    "spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm"
                ) from exc
        return self._spacy

    def _extract_stanza(self, words: list[str]) -> list[str]:
        nlp = self._load_stanza()
        lemmas: list[str] = []
        for word in words:
            doc = nlp(word)
            for sent in doc.sentences:
                for token in sent.words:
                    lemmas.append(token.lemma.lower())
        return lemmas

    def _extract_spacy(self, words: list[str]) -> list[str]:
        nlp = self._load_spacy()
        lemmas: list[str] = []
        for word in words:
            doc = nlp(word)
            lemmas.extend(token.lemma_.lower() for token in doc)
        return lemmas

    def _extract_polyglot(self, words: list[str]) -> list[str]:
        from polyglot.text import Text

        morphemes: list[str] = []
        for word in words:
            try:
                morphemes.extend(Text(word, hint_language_code="en").morphemes)
            except Exception:
                morphemes.append(word)
        return morphemes


def extract_morphemes_stanza(word_list: list[str]) -> list[str]:
    """Notebook-compatible helper."""
    return MorphemeExtractor("stanza").extract(word_list)


def extract_spacy_support(word_list: list[str]) -> list[str]:
    """Notebook-compatible helper."""
    return MorphemeExtractor("spacy").extract_support(word_list)
