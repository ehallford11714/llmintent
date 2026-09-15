"""Entity state file: persistence is the current situation, not the cast list.

Name-lock stored who appeared (1, 2, 3). Reasoning-as-persistence needs
where the ball is *now* after the last swap. This file updates. Identity
payloads stay locked at birth; situation does not.

This is beside the transformer. It does not claim the base weights grew
object files. Decay is not trained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class WorldFile:
    """Current attributes. Keys are entities; values are the live state."""

    loc: dict[str, str] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)
    cup: dict[str, str | None] = field(default_factory=dict)
    bag: dict[str, str | None] = field(default_factory=dict)
    acc: int | None = None
    ignore: set[int] = field(default_factory=set)
    seen_cups: set[str] = field(default_factory=set)
    seen_places: set[str] = field(default_factory=set)
    seen_nums: set[str] = field(default_factory=set)

    def value_surfaces(self) -> list[str]:
        """Live attribute values only (cup 1, house) — not entity names."""
        out: list[str] = []
        seen: set[str] = set()

        def add(piece: str) -> None:
            p = str(piece).strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)

        for v in self.loc.values():
            add(v)
        for v in self.count.values():
            add(str(v))
        if self.acc is not None:
            add(str(self.acc))
        return out

    def rival_surfaces(self) -> list[str]:
        live = set(self.value_surfaces())
        out: list[str] = []
        for n in sorted(self.seen_cups | self.seen_places | self.seen_nums):
            if n not in live:
                out.append(n)
        return out

    def format_live(self) -> str:
        bits: list[str] = []
        if "ball" in self.loc:
            bits.append(f"the ball is under cup {self.loc['ball']}")
        for k, v in sorted(self.loc.items()):
            if k == "ball":
                continue
            bits.append(f"the {k} is at {v}")
        for k, v in sorted(self.count.items()):
            bits.append(f"{k} has {v}")
        if self.acc is not None and "box" not in self.count:
            bits.append(f"the running total is {self.acc}")
        return "; ".join(bits)

    def surfaces(self) -> list[str]:
        """Live values only — the persistend situation, not every mention."""
        out: list[str] = []
        seen: set[str] = set()

        def add(piece: str) -> None:
            p = str(piece).strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)

        for v in self.loc.values():
            add(v)
        for v in self.count.values():
            add(str(v))
        if self.acc is not None:
            add(str(self.acc))
        for v in self.cup.values():
            if v:
                add(v)
        for v in self.bag.values():
            if v:
                add(v)
        for k, v in self.cup.items():
            if v == "ball":
                add(k)
        return out


def _swap_map(d: dict[str, str | None], a: str, b: str) -> None:
    d.setdefault(a, None)
    d.setdefault(b, None)
    d[a], d[b] = d[b], d[a]


def apply_world(text: str) -> WorldFile:
    """Walk the prompt in order and rewrite the file."""
    w = WorldFile()
    t = " ".join(text.split())

    ign = re.search(r"ignore\s+(\d+)", t, re.I)
    if ign:
        w.ignore.add(int(ign.group(1)))

    start_acc = re.search(r"start with\s+(\d+)", t, re.I)
    if start_acc:
        w.acc = int(start_acc.group(1))
        w.seen_nums.add(str(int(start_acc.group(1))))
    if ign:
        w.seen_nums.add(str(int(ign.group(1))))

    # Token events in document order.
    events: list[tuple[int, str, tuple]] = []

    def mark(rx: str, kind: str, flags: int = re.I) -> None:
        for m in re.finditer(rx, t, flags):
            events.append((m.start(), kind, m.groups()))

    mark(r"ball starts under cup\s+(\d+)", "ball_start")
    mark(r"swap cup\s+(\d+)\s+with cup\s+(\d+)", "swap_cup")
    mark(r"the (\w+) is on the (\w+)", "loc")
    mark(r"the (\w+) is in the (\w+)", "loc")
    mark(r"the (\w+) moves to the (\w+)", "move")
    mark(r"(\w+) has (\d+) books", "has")
    mark(r"(\w+) gives (\d+) to (\w+)", "give")
    mark(r"a box has (\d+)", "box")
    mark(r"\badd (\d+)\b", "add")
    mark(r"take (\d+) out", "take")
    mark(r"subtract (\d+)", "sub")
    mark(r"a (\w+) is in the (\w+) bag", "in_bag")
    mark(r"they are swapped", "swap_bags")
    mark(r"the (\w+) is moved to the (\w+)", "moved")
    mark(r"actually moved to the (\w+)", "actual")
    mark(r"that is false", "false")
    mark(r"held by the (\w+)", "held")
    mark(r"gives it to the (\w+)", "give_it")

    events.sort(key=lambda e: e[0])
    skip_next_move = False
    for _pos, kind, g in events:
        if kind == "false":
            skip_next_move = True
        elif kind == "ball_start":
            cup = g[0]
            w.cup[cup] = "ball"
            w.loc["ball"] = cup
            w.seen_cups.add(cup)
        elif kind == "swap_cup":
            a, b = g[0], g[1]
            w.seen_cups.add(a)
            w.seen_cups.add(b)
            _swap_map(w.cup, a, b)
            if w.loc.get("ball") == a:
                w.loc["ball"] = b
            elif w.loc.get("ball") == b:
                w.loc["ball"] = a
        elif kind == "loc":
            w.loc[g[0].lower()] = g[1].lower()
            w.seen_places.add(g[1].lower())
        elif kind == "move":
            w.loc[g[0].lower()] = g[1].lower()
            w.seen_places.add(g[1].lower())
        elif kind == "has":
            w.count[g[0]] = int(g[1])
            w.seen_nums.add(str(int(g[1])))
        elif kind == "give":
            src, n, dst = g[0], int(g[1]), g[2]
            w.count[src] = w.count.get(src, 0) - n
            w.count[dst] = w.count.get(dst, 0) + n
            w.seen_nums.add(str(n))
        elif kind == "box":
            w.count["box"] = int(g[0])
            w.seen_nums.add(str(int(g[0])))
            if w.acc is None:
                w.acc = int(g[0])
        elif kind == "add":
            n = int(g[0])
            if n not in w.ignore:
                if "box" in w.count:
                    w.count["box"] += n
                if w.acc is not None:
                    w.acc += n
            w.seen_nums.add(str(n))
        elif kind == "take":
            n = int(g[0])
            if n not in w.ignore:
                if "box" in w.count:
                    w.count["box"] -= n
                if w.acc is not None:
                    w.acc -= n
            w.seen_nums.add(str(n))
        elif kind == "sub":
            n = int(g[0])
            if n not in w.ignore and w.acc is not None:
                w.acc -= n
            w.seen_nums.add(str(n))
        elif kind == "in_bag":
            w.bag[g[1].lower()] = g[0].lower()
            w.loc[g[0].lower()] = g[1].lower()
        elif kind == "swap_bags":
            if "red" in w.bag and "blue" in w.bag:
                _swap_map(w.bag, "red", "blue")
                for who, place in list(w.loc.items()):
                    if place == "red":
                        w.loc[who] = "blue"
                    elif place == "blue":
                        w.loc[who] = "red"
        elif kind == "moved":
            if skip_next_move:
                skip_next_move = False
                continue
            who, dest = g[0].lower(), g[1].lower()
            if who in ("they", "someone"):
                continue
            for k, v in list(w.bag.items()):
                if v == who:
                    w.bag[k] = None
            w.loc[who] = dest
            w.seen_places.add(dest)
        elif kind == "actual":
            # last object mentioned that has a loc, else egg
            who = "egg" if "egg" in w.loc or "egg" in t.lower() else next(iter(w.loc), "egg")
            w.loc[who] = g[0].lower()
            w.seen_places.add(g[0].lower())
        elif kind == "held":
            w.loc["book"] = g[0].lower()
            w.seen_places.add(g[0].lower())
        elif kind == "give_it":
            w.loc["book"] = g[0].lower()
            w.seen_places.add(g[0].lower())
    return w
