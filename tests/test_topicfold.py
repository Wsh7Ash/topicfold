from topicfold.topicfold import (
    agglomerative_cluster,
    cosine,
    feature_hash,
    fold,
    fold_dir,
    render_report,
    tokenize,
    vectorize,
)


def test_tokenize_drops_stopwords_and_short():
    toks = tokenize("The cat and the hat sat on a mat to the airport")
    assert "the" not in toks
    assert "cat" in toks and "airport" in toks
    assert all(len(t) > 1 for t in toks)


def test_feature_hash_is_deterministic_and_in_range():
    assert feature_hash("hello", 512) == feature_hash("hello", 512)
    assert 0 <= feature_hash("hello", 512) < 512


def test_identical_vectors_have_full_cosine():
    v = vectorize(tokenize("database migrations rollback forward database"))
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_related_notes_cluster_together():
    notes = [
        ("db1.txt", "database migration rollback strategy for postgres schema"),
        ("db2.txt", "postgres schema migration and rollback plan"),
        ("cake1.txt", "chocolate cake recipe with flour sugar butter and eggs"),
    ]
    clusters = fold(notes, threshold=0.15)
    # db1 and db2 must share a cluster; cake1 should be separate.
    groups = {frozenset(c.docs) for c in clusters}
    assert any({"db1.txt", "db2.txt"} <= set(g) for g in groups)
    assert not any({"db1.txt", "cake1.txt"} <= set(g) for g in groups)


def test_agglomerative_respects_threshold():
    # Two orthogonal one-hot vectors: cosine 0, never merge at threshold > 0.
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    groups = agglomerative_cluster([a, b], threshold=0.5)
    assert sorted(len(g) for g in groups) == [1, 1]


def test_fold_dir_and_report(tmp_path):
    (tmp_path / "a.txt").write_text("kubernetes pods deployment replicas", encoding="utf-8")
    (tmp_path / "b.txt").write_text("kubernetes deployment rolling update", encoding="utf-8")
    (tmp_path / "c.txt").write_text("watercolor painting brushes and canvas", encoding="utf-8")
    clusters = fold_dir(tmp_path, threshold=0.1)
    assert len(clusters) >= 1
    report = render_report(clusters)
    assert "topic cluster" in report
    assert "a.txt" in report


def test_fold_empty_input():
    assert fold([]) == []
