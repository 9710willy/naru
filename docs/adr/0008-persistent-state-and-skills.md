# 0008. Persist agent state and curated skills

Raw agent traces stay recoverable during normal retention, and evidence for an active item lasts as long as that item. Naru stores each valid `headline` as canonical `agent_state` JSON with its source run and sequence range. The agent writes each completed trace span to a scoped Event Log pointer and rolls older pointers into a tiered index. It does not keep a second general view-eviction policy. Procedures reuse the claim inbox and reach `## Skills` only after promotion, which keeps one store and one human gate.

Sources: [WikiSkill](https://arxiv.org/abs/2608.27454) for keeping raw experience recoverable and promoting reviewed procedures, and [SKILL.state](https://arxiv.org/abs/2608.26263) for separating stable instructions from compact changing state.
