"""Tests for J-space regime classification."""

import pandas as pd

from llmintent.jspace.regimes import LayerRegime, classify_layer_regimes, regime_bands


def test_classify_three_regimes():
    stats = pd.DataFrame(
        {
            "layer": list(range(9)),
            "entropy": [4.0, 3.8, 3.5, 2.5, 2.0, 1.8, 1.0, 0.5, 0.2],
            "motor_alignment": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95, 0.99],
        }
    )
    out = classify_layer_regimes(stats, num_layers=9)
    assert set(out["regime"]) == {LayerRegime.SENSORY.value, LayerRegime.WORKSPACE.value, LayerRegime.MOTOR.value}
    bands = regime_bands(out)
    assert bands.sensory[0] <= bands.workspace[0] <= bands.motor[0]
