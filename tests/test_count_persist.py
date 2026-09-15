"""Higher-count items stay aligned with the world file; lead scoring is first-line only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from count_persist_27b import ITEMS, assess, lead_nums, wrong_live
from llmintent.worldfile import apply_world


def test_items_match_world_file():
    assert [i["id"] for i in ITEMS] == [
        "shell_n3",
        "shell_n7",
        "shell_n11",
        "count_n5",
        "count_n9",
        "count_n13",
        "give_n3",
        "give_n9",
    ]
    for item in ITEMS:
        w = apply_world(item["text"])
        if item["family"] == "shell":
            assert w.loc["ball"] == item["expect"][0]
        elif item["family"] == "count":
            assert str(w.count["box"]) == item["expect"][0]
        else:
            assert str(w.count["Ada"]) == item["expect"][0]
            assert str(w.count["Bea"]) == item["expect"][1]


def test_stale_drops_last_update():
    shell = next(i for i in ITEMS if i["id"] == "shell_n3")
    assert apply_world(shell["text"]).loc["ball"] == "1"
    assert apply_world(shell["stale_text"]).loc["ball"] == "3"
    count = next(i for i in ITEMS if i["id"] == "count_n5")
    assert apply_world(count["stale_text"]).count["box"] == 8
    give = next(i for i in ITEMS if i["id"] == "give_n3")
    stale = apply_world(give["stale_text"])
    assert stale.count["Ada"] == 8
    assert stale.count["Bea"] == 5


def test_lead_ignores_walkthrough_start():
    item = next(i for i in ITEMS if i["id"] == "shell_n3")
    walk = "Step 1: Swap cup 1 with cup 2.\nThe ball ends under cup 1."
    assert lead_nums(walk) == []
    assert assess(walk, item)["wrote_result"] is False
    wrong_lead = "The ball is under cup 3.\nStart under cup 1."
    assert assess(wrong_lead, item)["wrote_result"] is False
    assert assess("1\nAfter the last swap.", item)["wrote_result"] is True
    assert assess("The ball is under cup 1.", item)["wrote_result"] is True


def test_give_needs_both_first_line_numbers():
    item = next(i for i in ITEMS if i["id"] == "give_n3")
    assert assess("7 6\nAda then Bea.", item)["wrote_result"] is True
    assert assess("Ada has 7.\nBea has 6.", item)["wrote_result"] is False


def test_wrong_live_replaces_expect():
    item = next(i for i in ITEMS if i["id"] == "shell_n3")
    assert "cup 2" in wrong_live(item, "the ball is under cup 1") or "3" in wrong_live(
        item, "the ball is under cup 1"
    )
