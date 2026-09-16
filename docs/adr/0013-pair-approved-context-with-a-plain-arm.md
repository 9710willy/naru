# 0013. Pair approved context with a plain arm

## Context

The benchmark measures programmatic recall over long histories. It does not
show whether a promoted Naru doc changes normal coding-agent behavior. Hook
delivery and claim counts prove that data moved. They do not prove that the
model used it or that unrelated work stayed safe.

## Decision

Run each ordinary task twice with the same model and settings. Give one arm the
approved Naru doc and give the other no Naru doc. Change nothing else.

Use literal required and forbidden answer text for the first harness. Mark a
case as `fact` when approved context should help. Mark it as `control` when the
context should not matter. Report paired wins and losses with an exact McNemar
test. Count a plain-only control pass as harmful carryover.

Keep arm names and score text out of the model prompt. Store full answers in an
ignored result artifact for review.

## Why

The Context as an Environment paper supports bounded context delivery. MemDelta
supports comparison against a retrieval control. Neither result proves this
repo's human-curated layer. A paired plain arm isolates that layer with the
smallest useful change. Literal checks keep the first result reproducible and
avoid adding a second model judge before the basic effect exists.
