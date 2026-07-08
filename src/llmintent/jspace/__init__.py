"""J-space: verbalizable intent decoding inspired by Anthropic's Jacobian lens.

Reference: Gurnee et al., "Verbalizable Representations Form a Global Workspace
in Language Models" (Transformer Circuits, 2026).
https://transformer-circuits.pub/2026/workspace/
"""

from llmintent.jspace.decode import IntentToken, decode_intents, logit_lens_decode
from llmintent.jspace.decompose import SparseIntent, sparse_intent_decomposition
from llmintent.jspace.regimes import LayerRegime, classify_layer_regimes
from llmintent.jspace.trace import IntentTrace, build_intent_trace
from llmintent.jspace.transport import TransportMaps, fit_transport_maps

__all__ = [
    "IntentToken",
    "IntentTrace",
    "LayerRegime",
    "SparseIntent",
    "TransportMaps",
    "build_intent_trace",
    "classify_layer_regimes",
    "decode_intents",
    "fit_transport_maps",
    "logit_lens_decode",
    "sparse_intent_decomposition",
]
