"""Agreement statistics and the sentiment classifier. No UI code here."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline

from db import LABELS, Record

TIE = "tie"
TEST_SIZE = 0.2
MIN_TRAINING_ROWS = 10


def resolve(labels: tuple[str, ...]) -> tuple[str, float]:
    """Majority label and agreement. With no majority (all three differ) the label is TIE."""
    counts = Counter(labels)
    label, votes = counts.most_common(1)[0]
    agreement = votes / len(labels)
    if votes == 1:
        return TIE, agreement
    return label, agreement


@dataclass(frozen=True)
class Summary:
    total: int
    unique_texts: int
    average_agreement: float | None
    ties: int


def summarize(records: list[Record]) -> Summary:
    resolved = [resolve(r.labels) for r in records]
    agreements = [a for _, a in resolved]
    return Summary(
        total=len(records),
        unique_texts=len({r.text for r in records}),
        average_agreement=sum(agreements) / len(agreements) if agreements else None,
        ties=sum(1 for label, _ in resolved if label == TIE),
    )


@dataclass(frozen=True)
class SentenceRow:
    text: str
    records: int
    votes: dict[str, int]
    majority: str
    average_agreement: float


def by_sentence(records: list[Record]) -> list[SentenceRow]:
    """Per-sentence totals: votes per label across every record of that sentence."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.text].append(r)
    rows = []
    for text, recs in groups.items():
        votes = Counter(label for r in recs for label in r.labels)
        top = votes.most_common()
        majority = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else TIE
        rows.append(
            SentenceRow(
                text=text,
                records=len(recs),
                votes={label: votes.get(label, 0) for label in LABELS},
                majority=majority,
                average_agreement=sum(resolve(r.labels)[1] for r in recs) / len(recs),
            )
        )
    return sorted(rows, key=lambda row: (-row.records, row.text))


@dataclass
class ClassifierResult:
    ok: bool
    message: str = ""
    accuracy: float | None = None
    report: str = ""
    unseen_accuracy: float | None = None
    ceiling: float | None = None
    train_size: int = 0
    test_size: int = 0
    excluded_ties: int = 0
    seed: int = 42
    predictions: dict[int, tuple[str, float]] = field(default_factory=dict)


def _model(seed: int):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=1000, random_state=seed),
    )


def label_ceiling(texts: list[str], labels: list[str]) -> float:
    """Best accuracy any model that sees only the text can reach on these rows.

    Rows with the same text but different labels can't all be right, so the best possible answer per
    sentence is its commonest label among these rows.
    """
    per_text: dict[str, Counter] = defaultdict(Counter)
    for text, label in zip(texts, labels):
        per_text[text][label] += 1
    return sum(c.most_common(1)[0][1] for c in per_text.values()) / len(labels)


def unseen_sentence_accuracy(texts: list[str], labels: list[str], seed: int) -> float | None:
    """GroupKFold by sentence: every test sentence is absent from training."""
    unique = len(set(texts))
    if unique < 2:
        return None
    hits = 0
    for train_idx, test_idx in GroupKFold(n_splits=min(5, unique)).split(texts, labels, groups=texts):
        y_train = [labels[i] for i in train_idx]
        if len(set(y_train)) < 2:
            continue
        model = _model(seed).fit([texts[i] for i in train_idx], y_train)
        predicted = model.predict([texts[i] for i in test_idx])
        hits += sum(p == labels[i] for p, i in zip(predicted, test_idx))
    return hits / len(texts)


def train_classifier(records: list[Record], seed: int = 42) -> ClassifierResult:
    """Train on every non-tie record with a stratified 80/20 split, then predict every record."""
    resolved = [(r, resolve(r.labels)[0]) for r in records]
    data = [(r.text, label) for r, label in resolved if label != TIE]
    excluded = len(resolved) - len(data)
    texts = [t for t, _ in data]
    labels = [label for _, label in data]
    classes = Counter(labels)

    if len(data) < MIN_TRAINING_ROWS or len(classes) < 2:
        return ClassifierResult(
            ok=False,
            excluded_ties=excluded,
            seed=seed,
            message=f"Need at least {MIN_TRAINING_ROWS} resolved records in 2 or more labels to train "
            f"(have {len(data)} in {len(classes)}).",
        )

    test_count = max(1, round(len(data) * TEST_SIZE))
    stratify = labels if min(classes.values()) >= 2 and test_count >= len(classes) else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=seed, stratify=stratify
    )
    if len(set(y_train)) < 2:
        return ClassifierResult(ok=False, excluded_ties=excluded, seed=seed,
                                message="The training split holds only one label; add more annotations.")

    model = _model(seed).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    present = [label for label in LABELS if label in set(y_test) | set(y_pred)]
    report = classification_report(y_test, y_pred, labels=present, digits=2, zero_division=0)

    probabilities = model.predict_proba([r.text for r in records])
    classes_ = list(model.classes_)
    predictions = {
        r.id: (classes_[row.argmax()], float(row.max())) for r, row in zip(records, probabilities)
    }

    return ClassifierResult(
        ok=True,
        accuracy=accuracy_score(y_test, y_pred),
        report=report,
        unseen_accuracy=unseen_sentence_accuracy(texts, labels, seed),
        ceiling=label_ceiling(x_test, y_test),
        train_size=len(x_train),
        test_size=len(x_test),
        excluded_ties=excluded,
        seed=seed,
        predictions=predictions,
    )


def describe(result: ClassifierResult) -> str:
    """The classification report followed by notes on what the numbers mean."""
    if not result.ok:
        return result.message
    lines = [
        f"Model: TF-IDF (1-2 grams) + logistic regression · seed {result.seed}",
        f"Train {result.train_size} / test {result.test_size} records, stratified 80/20 split; "
        f"{result.excluded_ties} tie records left out",
        "",
        result.report.rstrip(),
        "",
        f"Accuracy on the held-out split:            {result.accuracy:.2%}",
    ]
    if result.ceiling is not None:
        lines.append(f"Best possible on this split (label noise): {result.ceiling:.2%}")
    if result.unseen_accuracy is not None:
        lines.append(f"Accuracy on unseen sentences (GroupKFold): {result.unseen_accuracy:.2%}")
    lines += [
        "",
        "The test split shares its sentences with the training split, so its accuracy measures how well",
        "the model recalls each sentence's usual label. The same sentence carries different labels in",
        "different records, so no model can beat the 'best possible' line. GroupKFold keeps whole",
        "sentences out of training and shows how the model does on text it has never seen.",
    ]
    return "\n".join(lines)
