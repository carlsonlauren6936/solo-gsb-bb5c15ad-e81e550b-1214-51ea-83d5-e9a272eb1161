"""Cross-consistency tests for the shared SnapshotTimeline replay engine.

The same single-pass engine backs three endpoints:

* ``GET /participants/compare``                 (per-participant totals)
* ``GET /participants/{id}/threshold-intervals`` (per-participant totals)
* ``GET /leaderboard/changes``                  (full ranked snapshots)

These tests drive one ledger history — containing linear corrections, a
forking correction, a cross-participant re-attribution, a later chained
correction and negative points — and verify every engine-derived view
agrees with the independently-implemented ``GET /leaderboard`` snapshot at
every revision, plus direct engine-level invariants. Sub-interval queries
must report exactly the restriction of the full-range result.
"""

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
import app.ledger as ledger
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated SQLite file."""
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "timeline-test.db"))
    db_module.init_db()
    return TestClient(app)


def _post(client, event_id, participant_id, points, correction_of=None):
    payload = {
        "event_id": event_id,
        "participant_id": participant_id,
        "points": points,
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    r = client.post("/events", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture()
def rich_history(client):
    """A history exercising every engine concern.

    rev1  e1 alice +10          (linear chain start)
    rev2  e2 bob   +7
    rev3  e3 alice +20 -> e1    (linear correction; alice 10 -> 20)
    rev4  e4 carol +30 -> e1    (fork off e1: e3 and e4 both effective)
    rev5  e5 dave  -25 -> e3    (cross-participant + chain extension;
                                 e3 leaves, alice 20 -> 0, dave -25;
                                 e4 stays carol's leaf)
    rev6  e6 bob   -7
    rev7  e7 alice +5           (alice 0 -> 5; the 0-total gap is real)
    rev8  e8 bob   +20 -> e2    (same-owner correction: e2's +7 is replaced
                                 by +20; the standalone -7 event remains)
    """
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    _post(client, "e3", "alice", 20, correction_of="e1")
    _post(client, "e4", "carol", 30, correction_of="e1")
    _post(client, "e5", "dave", -25, correction_of="e3")
    _post(client, "e6", "bob", -7)
    _post(client, "e7", "alice", 5)
    _post(client, "e8", "bob", 20, correction_of="e2")
    return client


def _leaderboard_totals(client, revision):
    """Independently computed ``{participant_id: total_points}`` snapshot.

    Revision 0 is the empty leaderboard (not queryable via ``/leaderboard``).
    """
    if revision == 0:
        return {}
    body = client.get("/leaderboard", params={"revision": revision})
    assert body.status_code == 200
    return {e["participant_id"]: e["total_points"] for e in body.json()["entries"]}


def _leaderboard_ranks(client, revision):
    """Independently computed ``{participant_id: {rank,total}}`` snapshot."""
    if revision == 0:
        return {}
    body = client.get("/leaderboard", params={"revision": revision})
    assert body.status_code == 200
    return {
        e["participant_id"]: {"rank": e["rank"], "total_points": e["total_points"]}
        for e in body.json()["entries"]
    }


# ---------------------------------------------------------------------------
# Engine-level invariants (linear correction, fork, cross-participant,
# negative points)
# ---------------------------------------------------------------------------

def _timeline_at(to_revision, target_revision):
    """Build the engine with exactly one range read, replayed to ``target``."""
    with ledger.read_tx() as conn:
        timeline = ledger.SnapshotTimeline.load(conn, to_revision)
    timeline.advance_to(target_revision)
    return timeline


def test_linear_correction_tracks_effective_tip(rich_history):
    # rev1: e1 effective; rev2 still; rev3: e3 supersedes e1 (first correction)
    tl = _timeline_at(3, 3)
    assert tl.is_effective("e3")
    assert not tl.is_effective("e1")
    assert tl.chain_leaves("e1") == frozenset({"e3"})
    assert tl.fork_roots() == frozenset()
    assert tl.participant_total("alice") == 20
    first = tl.first_corrections
    assert [(c.event_id, c.from_participant, c.to_participant) for c in first][:1] == [
        ("e3", "alice", "alice")
    ]


def test_fork_keeps_both_leaves_and_supersedes_ancestor_once(rich_history):
    tl = _timeline_at(4, 4)
    assert not tl.is_effective("e1")  # shared ancestor superseded once
    assert tl.is_effective("e3")
    assert tl.is_effective("e4")
    assert tl.chain_leaves("e1") == frozenset({"e3", "e4"})
    assert tl.fork_roots() == frozenset({"e1"})
    # both leaves count: alice 20, carol 30; e1 is not double-subtracted
    assert tl.participant_total("alice") == 20
    assert tl.participant_total("carol") == 30


def test_cross_participant_correction_debits_and_credits_once(rich_history):
    tl = _timeline_at(5, 5)
    # e5 (dave, -25) corrects e3 (alice, 20): alice loses the leaf, dave gets it
    assert not tl.is_effective("e3")
    assert tl.is_effective("e4")  # the other fork branch is untouched
    assert tl.chain_leaves("e1") == frozenset({"e4", "e5"})
    assert tl.participant_total("alice") == 0   # 20 gone, no other alice event
    assert tl.participant_total("carol") == 30
    assert tl.participant_total("dave") == -25  # negative points leaf
    assert tl.participant_total("bob") == 7
    # e4 forks e1 but e1 is already superseded by e3 at rev4, so it is a
    # fork-branch event rather than a "first correction"; e5 is the only
    # cross-participant first correction (e3 alice -> e5 dave).
    cross = [
        c for c in tl.first_corrections
        if c.from_participant != c.to_participant
    ]
    assert (
        [(c.event_id, c.from_participant, c.to_participant) for c in cross]
        == [("e5", "alice", "dave")]
    )
    # the fork still records two leaves with different owners
    assert {
        tl._effective[eid].participant_id for eid in tl.chain_leaves("e1")
    } == {"carol", "dave"}


def test_negative_points_zero_total_participant_present_then_gone(rich_history):
    # rev6: bob 7 - 7 nets to exactly 0 but he still holds effective events;
    # rev8: bob's e2 is corrected away, but +20 replaces it.
    tl6 = _timeline_at(6, 6)
    assert tl6.participant_total("bob") == 0
    # a participant netting zero while holding events still appears in a full
    # snapshot (matches /leaderboard's _effective_rows aggregation)
    assert "bob" in tl6.rank_positions()
    assert tl6.rank_positions()["bob"].total_points == 0

    tl8 = _timeline_at(8, 8)
    assert not tl8.is_effective("e2")
    assert tl8.is_effective("e8")
    # e8 (+20) replaces e2 (+7); the standalone e6 (-7) stays effective,
    # so bob nets 20 - 7 = 13.
    assert tl8.participant_total("bob") == 13


def test_engine_snapshots_match_leaderboard_at_every_revision(rich_history):
    """Full ranked snapshot from the engine equals /leaderboard per revision."""
    with ledger.read_tx() as conn:
        timeline = ledger.SnapshotTimeline.load(conn, 8)
    for rev in range(0, 9):
        timeline.advance_to(rev)
        engine_totals = dict(timeline.ordered_totals())
        assert engine_totals == _leaderboard_totals(rich_history, rev), rev
        engine_ranks = {
            pid: {"rank": pos.rank, "total_points": pos.total_points}
            for pid, pos in timeline.rank_positions().items()
        }
        assert engine_ranks == _leaderboard_ranks(rich_history, rev), rev


def test_engine_uses_one_range_read_only(rich_history, monkeypatch):
    """The range read happens once; advancing revisions never re-queries."""
    import sqlite3

    executed_sql: list[str] = []

    class CountingConnection(sqlite3.Connection):
        def execute(self, sql, params=()):
            executed_sql.append(str(sql))
            return super().execute(sql, params)

    def counting_connect():
        conn = CountingConnection(
            db_module.DB_PATH, timeout=30, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # read_tx() resolves ``connect`` from the db module at call time
    monkeypatch.setattr(db_module, "connect", counting_connect)

    with ledger.read_tx() as conn:
        timeline = ledger.SnapshotTimeline.load(conn, 8)
        for rev in range(0, 9):
            timeline.advance_to(rev)
            timeline.participant_total("alice")
            timeline.rank_positions()

    # exactly one SELECT against the events table for the whole sweep
    event_reads = [sql for sql in executed_sql if " FROM events" in sql]
    assert len(event_reads) == 1, event_reads
    assert "revision <= ?" in event_reads[0]


# ---------------------------------------------------------------------------
# compare endpoint vs independent /leaderboard totals (every revision)
# ---------------------------------------------------------------------------

def _compare(client, left, right, frm, to):
    r = client.get(
        "/participants/compare",
        params={"left_id": left, "right_id": right,
                "from_revision": frm, "to_revision": to},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _expand_compare(body):
    """``{revision: (left_points, right_points, leader)}`` from intervals."""
    out = {}
    for iv in body["intervals"]:
        for rev in range(iv["start_revision"], iv["end_revision"] + 1):
            out[rev] = (iv["left_points"], iv["right_points"], iv["leader"])
    return out


def test_compare_matches_leaderboard_all_revisions(rich_history):
    body = _compare(rich_history, "alice", "bob", 0, 8)
    expanded = _expand_compare(body)
    assert set(expanded) == set(range(0, 9))
    for rev in range(0, 9):
        totals = _leaderboard_totals(rich_history, rev)
        lp, rp, leader = expanded[rev]
        assert lp == totals.get("alice", 0), rev
        assert rp == totals.get("bob", 0), rev
        expected = (
            "tie" if lp == rp else "left" if lp > rp else "right"
        )
        assert leader == expected, rev


def test_compare_subinterval_is_restriction_of_full_range(rich_history):
    full = _expand_compare(_compare(rich_history, "alice", "carol", 0, 8))
    for frm, to in [(2, 2), (3, 6), (4, 8), (0, 0), (7, 8)]:
        sub = _expand_compare(_compare(rich_history, "alice", "carol", frm, to))
        assert set(sub) == set(range(frm, to + 1))
        for rev in sub:
            assert sub[rev] == full[rev], (frm, to, rev)


def test_compare_sees_cross_participant_fork_branch(rich_history):
    # dave only appears at rev5 with the negative-points correction e5
    body = _compare(rich_history, "dave", "alice", 0, 8)
    expanded = _expand_compare(body)
    assert expanded[4] == (0, 20, "right")        # before dave exists
    assert expanded[5] == (-25, 0, "right")      # dave -25 loses to alice 0
    assert expanded[7] == (-25, 5, "right")      # alice returns with e7


# ---------------------------------------------------------------------------
# threshold intervals vs independent /leaderboard totals
# ---------------------------------------------------------------------------

def _threshold(client, participant, threshold, frm, to):
    r = client.get(
        f"/participants/{participant}/threshold-intervals",
        params={"threshold": threshold, "from_revision": frm, "to_revision": to},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _covered_revisions(body):
    covered = {}
    for iv in body["intervals"]:
        for rev in range(iv["start_revision"], iv["end_revision"] + 1):
            covered[rev] = iv
    return covered


def test_threshold_matches_leaderboard_all_revisions(rich_history):
    for participant, threshold in [
        ("alice", 10),
        ("alice", -1),
        ("bob", 13),
        ("carol", 30),
        ("dave", 0),
        ("ghost", 0),
        ("ghost", 1),
    ]:
        body = _threshold(rich_history, participant, threshold, 0, 8)
        covered = _covered_revisions(body)
        for rev in range(0, 9):
            total = _leaderboard_totals(rich_history, rev).get(participant, 0)
            if total >= threshold:
                assert rev in covered, (participant, threshold, rev)
            else:
                assert rev not in covered, (participant, threshold, rev)
        # endpoint totals are exact
        for iv in body["intervals"]:
            s_total = _leaderboard_totals(
                rich_history, iv["start_revision"]
            ).get(participant, 0)
            e_total = _leaderboard_totals(
                rich_history, iv["end_revision"]
            ).get(participant, 0)
            assert iv["start_points"] == s_total
            assert iv["end_points"] == e_total


def test_threshold_subinterval_is_restriction_of_full_range(rich_history):
    full_body = _threshold(rich_history, "alice", 10, 0, 8)
    full_covered = _covered_revisions(full_body)
    for frm, to in [(3, 7), (0, 0), (1, 1), (5, 8), (6, 7)]:
        sub = _threshold(rich_history, "alice", 10, frm, to)
        sub_covered = _covered_revisions(sub)
        expected = {rev: iv for rev, iv in full_covered.items() if frm <= rev <= to}
        # covered revisions match; endpoints re-derived for the cut interval
        assert set(sub_covered) == set(expected), (frm, to)
        for iv in sub["intervals"]:
            # every reported interval is maximal *within the subrange* and
            # agrees with the full-range totals
            for rev in range(iv["start_revision"], iv["end_revision"] + 1):
                assert rev in full_covered


def test_threshold_negative_points_and_zero_boundary(rich_history):
    # dave is absent (0 points) at rev0..4 and holds -25 from rev5 on:
    # threshold 0 qualifies only while absent (0 >= 0), then he is below.
    body = _threshold(rich_history, "dave", 0, 0, 8)
    assert [
        (iv["start_revision"], iv["end_revision"]) for iv in body["intervals"]
    ] == [(0, 4)]
    # threshold -25: both 0 (rev0..4) and -25 (rev5..8) qualify, so they
    # merge into one interval whose endpoints carry the different totals.
    body = _threshold(rich_history, "dave", -25, 0, 8)
    assert [
        (iv["start_revision"], iv["end_revision"], iv["start_points"], iv["end_points"])
        for iv in body["intervals"]
    ] == [(0, 8, 0, -25)]
    # a subrange starting inside the negative run reports just that run
    body = _threshold(rich_history, "dave", -25, 6, 8)
    assert [
        (iv["start_revision"], iv["end_revision"], iv["start_points"], iv["end_points"])
        for iv in body["intervals"]
    ] == [(6, 8, -25, -25)]
    # threshold strictly above -25 drops the negative run entirely
    body = _threshold(rich_history, "dave", -24, 0, 8)
    assert [
        (iv["start_revision"], iv["end_revision"], iv["start_points"], iv["end_points"])
        for iv in body["intervals"]
    ] == [(0, 4, 0, 0)]


# ---------------------------------------------------------------------------
# leaderboard/changes vs independent before/after /leaderboard snapshots
# ---------------------------------------------------------------------------

def _changes(client, frm, to):
    r = client.get(
        "/leaderboard/changes",
        params={"from_revision": frm, "to_revision": to},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_changes_match_independent_snapshots(rich_history):
    for frm, to in [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6), (0, 8), (6, 8)]:
        before = _leaderboard_ranks(rich_history, frm)
        after = _leaderboard_ranks(rich_history, to)
        body = _changes(rich_history, frm, to)
        by_pid = {c["participant_id"]: c for c in body["changes"]}
        for pid in sorted(set(before) | set(after)):
            b = before.get(pid)
            a = after.get(pid)
            change = by_pid.get(pid)
            if b == a:
                assert change is None, (frm, to, pid)
            else:
                assert change is not None, (frm, to, pid)
                assert change["before"] == b
                assert change["after"] == a
                assert change["points_delta"] == (
                    (a["total_points"] if a else 0)
                    - (b["total_points"] if b else 0)
                )
        # exactly the changed participants, ordered participant_id-ascending
        expected_pids = {
            pid
            for pid in set(before) | set(after)
            if before.get(pid) != after.get(pid)
        }
        assert set(by_pid) == expected_pids, (frm, to)
        pids = [c["participant_id"] for c in body["changes"]]
        assert pids == sorted(pids)


def test_changes_single_engine_drives_both_endpoints(rich_history, monkeypatch):
    """One request = one range read + one MAX read; snapshots share the engine."""
    calls: list[str] = []
    real_init = ledger.SnapshotTimeline.__init__

    def counting_init(self, events):
        real_init(self, events)
        # mark the event collection as observed by a single engine instance
        calls.append("engine")

    monkeypatch.setattr(ledger.SnapshotTimeline, "__init__", counting_init)
    body = _changes(rich_history, 3, 7)
    assert body["from_revision"] == 3 and body["to_revision"] == 7
    assert calls == ["engine"]


# ---------------------------------------------------------------------------
# revision=0 semantics and status codes stay unchanged across all three
# ---------------------------------------------------------------------------

def test_revision_zero_empty_snapshot_all_three(rich_history):
    # compare 0 -> 0 starts on the empty ledger
    body = _compare(rich_history, "alice", "bob", 0, 0)
    assert body["intervals"] == [
        {"start_revision": 0, "end_revision": 0,
         "left_points": 0, "right_points": 0, "leader": "tie"}
    ]
    # threshold with threshold 0 includes revision 0 with zero points
    body = _threshold(rich_history, "ghost", 0, 0, 0)
    assert body["intervals"] == [
        {"start_revision": 0, "end_revision": 0, "start_points": 0, "end_points": 0}
    ]
    # changes from revision 0 treats the empty board as "before"
    body = _changes(rich_history, 0, 1)
    assert body["changes"][0]["before"] is None


def test_validation_status_codes_unchanged(client):
    _post(client, "e1", "alice", 10)
    # ordering errors -> 422, out-of-range -> 404, ordering checked first
    assert client.get(
        "/participants/compare",
        params={"left_id": "a", "right_id": "b", "from_revision": 2, "to_revision": 1},
    ).status_code == 422
    assert client.get(
        "/participants/a/threshold-intervals",
        params={"threshold": 1, "from_revision": 2, "to_revision": 1},
    ).status_code == 422
    assert client.get(
        "/leaderboard/changes",
        params={"from_revision": 1, "to_revision": 1},
    ).status_code == 422
    assert client.get(
        "/participants/compare",
        params={"left_id": "a", "right_id": "b", "from_revision": 0, "to_revision": 9},
    ).status_code == 404
    assert client.get(
        "/participants/a/threshold-intervals",
        params={"threshold": 1, "from_revision": 0, "to_revision": 9},
    ).status_code == 404
    assert client.get(
        "/leaderboard/changes",
        params={"from_revision": 0, "to_revision": 9},
    ).status_code == 404
    # ordering (422) wins over range (404)
    assert client.get(
        "/participants/a/threshold-intervals",
        params={"threshold": 1, "from_revision": 9, "to_revision": 5},
    ).status_code == 422
