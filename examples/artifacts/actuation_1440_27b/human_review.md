# Human review of a balanced 12-case sample

Reviewed 2 frame + 2 policy items from each split. Benchmark was not edited.

No malformed JSON, missing choices, or broken letter keys. Answer letters match `choices` and `answer_semantic`. Letter-to-label maps are permuted across items as intended. Policy items keep STEP/WAIT crossed with mapping. Scene sentences match the object/motion fields.

Possible ambiguity (not edited; would be a new benchmark version):

- `calibration:t7:disk:toward:frame:commit`: Casey answers in first-person quotes ("My choice is to step aside."). Gold label is commit, not quote. Quote is defined as somebody else's words, so the gold is defensible, but the quotation marks can invite a quote classification.

- `confirmation:t10:bag:toward:frame:commit`: "The decision was settled yesterday" plus "on the next turn" is a past decision with pending execution. Gold is commit. Cross-tense items are intended; still a harder status call than t0.

No silent edits. Dataset SHA-256 remains `db709fbb000adc557278de0c64bbf84733101eefb6068504f066a132ab94e5d0`.
