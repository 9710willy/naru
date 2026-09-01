# 0008. Persist agent state and curated skills

Raw agent traces stay recoverable during normal retention, and evidence for an active item lasts as long as that item. Naru stores each valid `headline` as canonical `agent_state` JSON with its source run and sequence range, and it sends older trace through scoped Event Log pointers. Procedures reuse the claim inbox and reach `## Skills` only after promotion, which keeps one store and one human gate.
