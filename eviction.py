"""Compact, recoverable trace indexes."""

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


def demo():
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

    print(
        f"ok — eviction checks passed (60 evictions -> {total_entries} index entries, "
        f"{len(idx2)} tiers)"
    )


if __name__ == "__main__":
    demo()
