# 1,440-case actuation study (Qwen 3.8-27B)

Measured on the real local 27B NF4 snapshot. No GPT-2, unsloth, or fake activations.

The model can do the tasks. On held-out wording and objects it still got about **91%** of the letter answers right. It almost never missed STEP/WAIT. It struggled more with whether a sentence is a commitment or a reflection (reflect ~46% on confirmation).

## What is sitting in the network, and where

Toward vs away is easy to read out **early**. That is mostly because those words are already in the sentence. A bag-of-words classifier is perfect at motion; the network probe is not doing extra hidden work there.

Whether the text is a commitment, a reflection, a quote, and so on, is **not** readable early. It becomes readable in the later layers and stays readable through the final normalization. STEP vs WAIT is the same: weak as a non-word signal early, strong late.

## What the model actually uses

Readable is not the same as used. Swapping one example's internal state into a paired example:

- Early or middle layers (blocks 4 and 31 of 64) did almost nothing.
- A late layer (block 60) moved the answers a lot for scene direction, STEP vs WAIT, and commit vs reflect.
- Random noise of the same size did not. Putting a state's own vector back left the output unchanged.

## One sentence

On this 27B model, actuation-related information is **available early as wording**, **organized as a decision late**, and **causally used late**. It is not "intention only appears at the last layer," and it is not "the early motion signal is what steers the choice."

## What this does not show

It does not show a specific neuron, a fly-like circuit, or that the model has a goal of its own. Those tests were not run. Confirmation is three sentence families and four objects, not 1,440 independent proofs. Results are for this quantized 27B checkpoint, at the assistant-prefix token.

Full measured tables: [FINDINGS.md](FINDINGS.md).
