# Critique-Driven Design

SakanaAI/Text-to-Lora demonstrates an important direction: generate LoRA
adapters directly from natural-language task descriptions. The weak points this
prototype addresses are:

1. **Ungrounded generation**: a hypernetwork can produce plausible-looking
   weights even when the prompt is outside the training distribution.
2. **No confidence surface**: downstream code cannot tell whether an adapter is
   generated from evidence or extrapolated.
3. **Adapter safety**: raw generated deltas can have unstable norms or layer
   imbalance.
4. **Poor inspectability**: a heavy end-to-end system is hard to debug before
   real task evaluation.

True-LoRA keeps the natural-language interface, but makes the adapter generator
evidence-aware. Retrieval supplies a strong baseline from known adapters,
generation supplies compositional generalization, and uncertainty controls how
much each side contributes.

## Intended Evaluation

Use three evaluation layers:

- adapter reconstruction loss on held-out adapter banks
- task metrics after injecting generated adapters into a frozen base model
- stress tests for paraphrases, underspecified prompts, adversarial prompts, and
  out-of-domain descriptions

The model should abstain or shrink deltas when uncertainty is high instead of
silently producing high-impact adapter weights.
