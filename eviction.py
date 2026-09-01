"""Algorithm 1 — recoverable context eviction with a tiered headline index.

The working view is bounded; nothing is lost. Evicted spans stay verbatim in
the Event Log under their `seq` addresses, and a compact tiered index keeps the
agent aware of history it can no longer see, anchored to the exact addresses it
can re-materialize from.
"""

import re

TOK = 4  # chars per token, rough


def est(text):
    return max(1, len(text) // TOK)


def format_headline(task=None, verified=None, next_action=None, status=None):
    """The paper's landmark shape: task, verified state, next action, status.

    Auto-derived headlines (truncated block text) only support recall by
    wording, which is the failure lexical search already has. A model-authored
    landmark supports POSITION-based navigation: it says what a span was for,
    so the agent can decide to go back to it without remembering its words.
    """
    parts = [
        ("task", task),
        ("verified", ", ".join(verified or [])),
        ("next", next_action),
        ("status", status),
    ]
    got = [f"{k}={str(v).strip()[:70]}" for k, v in parts if v]
    return " | ".join(got)


class Block:
    """One unit of the working view."""

    __slots__ = ("headline", "is_payload", "role", "seq", "text")

    def __init__(self, seq, role, text, headline=None, is_payload=False):
        self.seq = seq
        self.role = role
        self.text = text
        # Fall back to truncated text only when the model wrote no landmark.
        self.headline = headline or (text.replace("\n", " ")[:60])
        self.is_payload = is_payload

    def tokens(self):
        return est(self.text)

    def __repr__(self):
        return f"<blk {self.seq} {self.role} {self.tokens()}t>"


def fold_payloads(blocks, recovery_session=None):
    """Tool payloads collapse to a seq pointer — cheapest recovery, so folded
    first. The row keeps a bounded preview; the full bytes stay in the log."""
    for b in blocks:
        if b.is_payload and not b.text.startswith("[folded"):
            preview = b.text.replace("\n", " ")[:80]
            call = f"ms.expand({b.seq}"
            if recovery_session is not None:
                call += f", session_id={recovery_session!r}"
            b.text = f"[folded payload seq {b.seq}: {preview}… -> {call})]"
    return blocks


def rollup(index, k):
    """Tiered index. Tier 0 holds the k newest evicted headline blocks in full;
    when it overflows, the k-1 oldest collapse to one line each and merge into
    the next tier. After n evictions the index is O(k log_k n) blocks."""
    for t in range(len(index)):
        while len(index[t]) > k:
            n = k - 1 if k > 1 else 1
            old = index[t][:n]
            sessions = {e.get("session_id") for e in old}
            if len(sessions) > 1:
                raise ValueError("cannot roll up mixed-session index entries")
            del index[t][:n]
            lo = min(e["lo"] for e in old)
            hi = max(e["hi"] for e in old)
            # "collapse to one line each and merge into the next tier" — one
            # line still has to carry signal. A bare span count tells the agent
            # nothing about whether a region is worth expanding, which defeats
            # the purpose of a navigation anchor.
            gist = "; ".join(
                # drop any "N spans:" prefix a lower tier already added, or
                # re-merging nests it: "3 spans: 3 spans: 3 spans: ..."
                re.sub(r"^\d+ spans:\s*", "", e["headline"].split("|")[-1].strip())[:26]
                for e in old
            ).strip("; ")[:110]
            merged = {
                "lo": lo,
                "hi": hi,
                "headline": f"{len(old)} spans: {gist}"
                if gist
                else f"{len(old)} spans, seq {lo}-{hi}",
            }
            session_id = sessions.pop()
            if session_id is not None:
                merged["session_id"] = session_id
            if t + 1 == len(index):
                index.append([])
            index[t + 1].append(merged)
    return index


def render_index(index):
    """The eviction index as it appears in the working view."""
    if not any(index):
        return ""
    lines = ["--- evicted (recover with ms.expand(lo, hi)) ---"]
    for t in range(len(index) - 1, -1, -1):
        for e in index[t]:
            call = f"ms.expand({e['lo']}, {e['hi']}"
            if "session_id" in e:
                call += f", session_id={e['session_id']!r}"
            lines.append(f"  [{e['lo']}-{e['hi']}] {e['headline']} -> {call})")
    return "\n".join(lines)


def evict(view, index, budget, k=4, protect_tail=3, recovery_session=None):
    """Algorithm 1. Mutates and returns (view, index).

    view    — list of Block, oldest first
    index   — list of tiers (list of dicts); [] on first call
    budget  — token ceiling (rho * C) for the view
    """
    if sum(b.tokens() for b in view) <= budget:
        return view, index

    # PROTECTED: the recent tail always stays verbatim in the view.
    split = max(0, len(view) - protect_tail)
    older, protected = view[:split], view[split:]

    # FOLDPAYLOADS: cheapest reduction first — payloads become seq pointers.
    fold_payloads(older, recovery_session)

    # SELECTSPAN + EVICTTOINDEX: drop the oldest spans until under budget.
    # Their headlines enter the index, anchored to exact seq addresses.
    def total():
        return sum(b.tokens() for b in older + protected)

    evicted = []
    while older and total() > budget:
        evicted.append(older.pop(0))
    if evicted:
        if not index:
            index.append([])
        entry = {
            "lo": min(b.seq for b in evicted),
            "hi": max(b.seq for b in evicted),
            "headline": "; ".join(b.headline for b in evicted[:2])[:90],
        }
        if recovery_session is not None:
            entry["session_id"] = recovery_session
        index[0].append(entry)
        rollup(index, k)

    return older + protected, index


def demo():
    ms_seq = 0

    def blk(text, payload=False):
        nonlocal ms_seq
        ms_seq += 1
        return Block(
            ms_seq, "user", text, headline=f"turn {ms_seq}", is_payload=payload
        )

    # under budget: nothing happens
    view = [blk("a" * 40), blk("b" * 40)]
    v, idx = evict(list(view), [], budget=1000)
    assert len(v) == 2 and idx == []

    # payload folding shrinks the view without touching the log
    big = blk("R" * 8000, payload=True)
    view = [big] + [blk("x" * 40) for _ in range(3)]
    before = sum(b.tokens() for b in view)
    v, idx = evict(view, [], budget=200, recovery_session="run")
    assert sum(b.tokens() for b in v) < before
    assert any("ms.expand" in b.text for b in v), "payload not folded to a pointer"
    assert f"ms.expand({big.seq}, session_id='run')" in v[0].text

    # over budget: oldest evicted, tail protected, index anchors addresses
    view = [blk("y" * 4000) for _ in range(10)]
    lo_seq = view[0].seq
    v, idx = evict(view, [], budget=800, protect_tail=3)
    assert sum(b.tokens() for b in v) <= 800 or len(v) == 3, sum(b.tokens() for b in v)
    assert len(v) >= 3, "tail must stay"
    assert idx and idx[0], "index empty after eviction"
    assert idx[0][0]["lo"] == lo_seq, idx

    # every evicted seq is still addressable through the index
    kept = {b.seq for b in v}
    covered = set()
    for tier in idx:
        for e in tier:
            covered |= set(range(e["lo"], e["hi"] + 1))
    original = {b.seq for b in view} | kept
    assert original <= (kept | covered), "invariant broken: a seq is unreachable"

    # rollup keeps the index sublinear: 60 evictions must not mean 60 entries
    idx2 = []
    for i in range(60):
        idx2.append([]) if not idx2 else None
        idx2[0].append({"lo": i * 10, "hi": i * 10 + 9, "headline": f"span {i}"})
        rollup(idx2, k=4)
    total_entries = sum(len(t) for t in idx2)
    assert total_entries <= 16, total_entries
    assert len(idx2) >= 3, f"expected multiple tiers, got {len(idx2)}"

    gapped = []
    for i in range(60):
        gapped.append([]) if not gapped else None
        gapped[0].append(
            {"lo": i * 3, "hi": i * 3 + 1, "headline": f"trace {i}", "session_id": "run"}
        )
        rollup(gapped, k=4)
    rendered_gaps = render_index(gapped)
    assert rendered_gaps.count("ms.expand(") == sum(len(t) for t in gapped) + 1
    assert len(rendered_gaps) < 2000, len(rendered_gaps)
    assert all(e.get("session_id") == "run" for tier in gapped for e in tier)

    try:
        mixed = [[{"lo": 1, "hi": 1, "headline": "one", "session_id": "a"},
                  {"lo": 2, "hi": 2, "headline": "two", "session_id": "b"},
                  {"lo": 3, "hi": 3, "headline": "three"},
                  {"lo": 4, "hi": 4, "headline": "four"}]]
        rollup(mixed, k=3)
        raise AssertionError("mixed-session rollup made a bad recovery handle")
    except ValueError:
        pass
    assert [e["lo"] for e in mixed[0]] == [1, 2, 3, 4]

    # rendering is compact and points at the recovery call
    r = render_index(idx)
    assert "ms.expand" in r and len(r) < 500, r

    print(
        f"ok — eviction checks passed (60 evictions -> {total_entries} index entries, "
        f"{len(idx2)} tiers)"
    )


if __name__ == "__main__":
    demo()
