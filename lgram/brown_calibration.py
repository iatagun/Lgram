"""
Brown Corpus calibration — run GenreCalibrator on NLTK Brown Corpus.
Produces empirically-derived genre thresholds with HIGH confidence (n=30/genre).
"""

from nltk.corpus import brown

from .genre_calibrator import GenreCalibrator

# Map Brown categories to our genres
GENRE_MAP = {
    "narrative": ["adventure", "fiction", "mystery", "romance", "science_fiction"],
    "expository": ["news", "government", "learned"],
    "essay": ["editorial", "reviews", "belles_lettres"],
}

SAMPLE_SIZE = 30  # per genre


def load_brown_corpus() -> dict:
    """Extract texts from Brown Corpus, mapped to our genre types."""
    corpus = {}
    for genre, cats in GENRE_MAP.items():
        texts = []
        for cat in cats:
            for fid in brown.fileids(cat):
                raw = " ".join(brown.words(fid))
                sents = raw.split(". ")
                chunk_size = 8
                for i in range(0, len(sents), chunk_size):
                    chunk = ". ".join(sents[i : i + chunk_size])
                    if len(chunk.split()) > 30:
                        texts.append(chunk.strip())
        corpus[genre] = texts[:SAMPLE_SIZE]
    return corpus


def run():
    corpus = load_brown_corpus()
    c = GenreCalibrator("en_core_web_md")
    profiles = c.calibrate(corpus)
    print(c.report(profiles))


if __name__ == "__main__":
    run()
