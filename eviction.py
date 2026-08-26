"""Algorithm 1 — recoverable context eviction with a tiered headline index.

The working view is bounded; nothing is lost. Evicted spans stay verbatim in
the Event Log under their `seq` addresses, and a compact tiered index keeps the
agent aware of history it can no longer see, anchored to the exact addresses it
can re-materialize from.
"""

TOK = 4  # chars per token, rough


def est(text):
    return max(1, len(text) // TOK)


class Block:
    """One unit of the working view."""

    __slots__ = ("headline", "is_payload", "role", "seq", "text")

    def __init__(self, seq, role, text, headline=None, is_payload=False):
        self.seq = seq
        self.role = role
        self.text = text
        self.headline = headline or (text.replace("\n", " ")[:60])
        self.is_payload = is_payload

    def tokens(self):
        return est(self.text)

    def __repr__(self):
        return f"<blk {self.seq} {self.role} {self.tokens()}t>"


def fold_payloads(blocks):
    """Tool payloads collapse to a seq pointer — cheapest recovery, so folded
    first. The row keeps a bounded preview; the full bytes stay in the log."""
    for b in blocks:
        if b.is_payload and not b.text.startswith("[folded"):
            preview = b.text.replace("\n", " ")[:80]
            b.text = f"[folded payload seq {b.seq}: {preview}… -> ms.expand({b.seq})]"
    return blocks


def rollup(index, k):
    """Tiered index. Tier 0 holds the k newest evicted headline blocks in full;
    when it overflows, the k-1 oldest collapse to one line each and merge into
    the next tier. After n evictions the index is O(k log_k n) blocks."""
    for t in range(len(index)):
        while len(index[t]) > k:
            old = (
                [index[t].pop(0) for _ in range(k - 1)] if k > 1 else [index[t].pop(0)]
            )
            lo = min(e["lo"] for e in old)
            hi = max(e["hi"] for e in old)
            merged = {
                "lo": lo,
                "hi": hi,
                "headline": f"{len(old)} spans, seq {lo}-{hi}",
            }
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
            lines.append(f"  [{e['lo']}-{e['hi']}] {e['headline']}")
    return "\n".join(lines)


def evict(view, index, budget, k=4, protect_tail=3):
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
    fold_payloads(older)

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
        index[0].append(
            {
                "lo": min(b.seq for b in evicted),
                "hi": max(b.seq for b in evicted),
                "headline": "; ".join(b.headline for b in evicted[:2])[:90],
            }
        )
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
    v, idx = evict(view, [], budget=200)
    assert sum(b.tokens() for b in v) < before
    assert any("ms.expand" in b.text for b in v), "payload not folded to a pointer"

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

    # rendering is compact and points at the recovery call
    r = render_index(idx)
    assert "ms.expand" in r and len(r) < 500, r

    print(
        f"ok — eviction checks passed (60 evictions -> {total_entries} index entries, "
        f"{len(idx2)} tiers)"
    )


if __name__ == "__main__":
    demo()
