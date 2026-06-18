"""
gaffer/events/base.py
──────────────────────
FootballEvent — a single detected football occurrence with timestamp,
type, responsible team, pitch location, and type-specific payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Event type strings ────────────────────────────────────────────────────────
# Point events (happen at a moment in time)
POSSESSION_CHANGE   = "possession_change"
POSSESSION_RECOVERY = "possession_recovery"   # won back after just losing it
COUNTER_ATTACK      = "counter_attack"         # recovery + rapid forward movement
HIGH_PRESS          = "high_press"             # onset: intensity >= threshold
HIGH_PRESS_ENDED    = "high_press_ended"
LINE_BREAK          = "line_break"             # ball crosses defending team's backline
SPRINT_START        = "sprint_start"
SPRINT_END          = "sprint_end"
COMPACT_BLOCK       = "compact_block"          # team enters low-block shape
PROGRESSIVE_PASS    = "progressive_pass"       # ball advances >15m toward goal
OVERLOAD            = "overload"                # numerical superiority in a pitch zone
DOMINANCE           = "dominance"                # sustained territorial control of attacking third
PASS                = "pass"                      # completed pass between teammates


_LABELS: dict[str, str] = {
    POSSESSION_CHANGE:   "POSS CHANGE",
    POSSESSION_RECOVERY: "RECOVERY",
    COUNTER_ATTACK:      "COUNTER!",
    HIGH_PRESS:          "HIGH PRESS",
    HIGH_PRESS_ENDED:    "press off",
    LINE_BREAK:          "LINE BREAK",
    SPRINT_START:        "sprint",
    SPRINT_END:          "sprint end",
    COMPACT_BLOCK:       "BLOCK",
    PROGRESSIVE_PASS:    "PROG PASS",
    OVERLOAD:            "OVERLOAD",
    DOMINANCE:           "DOMINANCE",
    PASS:                "pass",
}

_HIGHLIGHT: set[str] = {
    COUNTER_ATTACK, HIGH_PRESS, LINE_BREAK, POSSESSION_RECOVERY, PROGRESSIVE_PASS,
    OVERLOAD, DOMINANCE,
}


@dataclass
class FootballEvent:
    frame_idx:   int
    time_s:      float
    event_type:  str
    team:        str | None = None                      # "teamA" | "teamB"
    location_m:  tuple[float, float] | None = None     # pitch coords
    data:        dict = field(default_factory=dict)

    def label(self) -> str:
        base  = _LABELS.get(self.event_type, self.event_type)
        tlbl  = {"teamA": " A", "teamB": " B"}.get(self.team or "", "")
        if self.event_type == OVERLOAD and self.data:
            lane = self.data.get("lane", "").replace("_", " ")
            cf   = self.data.get("count_for")
            ca   = self.data.get("count_against")
            sc   = self.data.get("space_control_pct")
            suffix = f" sc={sc:.0f}%" if sc is not None else ""
            return f"{base}{tlbl} {lane} {cf}v{ca}{suffix}"
        if self.event_type == DOMINANCE and self.data:
            pct = self.data.get("control_pct")
            dur = self.data.get("duration_s")
            return f"{base}{tlbl} {pct:.0f}% ({dur:.0f}s)"
        if self.event_type == PASS and self.data:
            s = self.data.get("sender_id")
            r = self.data.get("receiver_id")
            d = self.data.get("distance_m")
            return f"{base}{tlbl} {s}->{r} {d:.0f}m"
        if self.event_type == PROGRESSIVE_PASS and self.data:
            d = self.data.get("distance_m") or self.data.get("fwd_m")
            return f"{base}{tlbl} {d:.0f}m" if d is not None else f"{base}{tlbl}"
        return f"{base}{tlbl}"

    @property
    def is_highlight(self) -> bool:
        return self.event_type in _HIGHLIGHT
