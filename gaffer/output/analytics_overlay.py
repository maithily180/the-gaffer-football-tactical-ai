"""
gaffer/output/analytics_overlay.py
────────────────────────────────────
Renders an AnalyticsSnapshot onto the video frame:

  • Stats panel (top-left)  — per-team compactness, defensive line, pressing,
    plus a possession bar.
  • Voronoi pitch inset     — the 2D pitch filled with each team's controlled
    cells (space control), players dotted on top.

Pure rendering; consumes the dataclasses produced by PitchAnalyticsEngine.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from gaffer import config
from gaffer.analytics.engine import AnalyticsSnapshot
from gaffer.analytics.episodes import Episode
from gaffer.calibration.pitch_model import PitchModel
from gaffer.events.base import FootballEvent

_A_CLR  = config.TEAM_A_COLOR_BGR     # red
_B_CLR  = config.TEAM_B_COLOR_BGR     # blue
_HUD    = (0, 220, 60)                # green
_GREY   = (160, 160, 160)
_YELLOW = (0, 220, 220)               # highlight events


_MAX_TICKER = 6       # max events shown in the ticker
_TICKER_TTL = 120    # frames an event stays in the ticker (~5s @ 25fps)
_EPISODE_TTL = 175    # frames a closed-episode banner stays up (~7s @ 25fps) — rarer, bigger than a single event


class AnalyticsOverlay:
    def __init__(self, pitch_model: PitchModel | None = None):
        self.pm = pitch_model or PitchModel()
        self._base_pitch = self.pm.draw_pitch()
        # (event, expire_frame) — newest first
        self._ticker: deque[tuple[FootballEvent, int]] = deque(maxlen=_MAX_TICKER)
        self._episode_banner: deque[tuple[Episode, int]] = deque(maxlen=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        snap: AnalyticsSnapshot | None,
        *,
        voronoi_width: int = 360,
    ) -> np.ndarray:
        """Return a copy of `frame` with the stats panel, Voronoi inset, and event ticker."""
        out = frame.copy()
        if snap is None:
            cv2.putText(out, "ANALYTICS: no calibration", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2, cv2.LINE_AA)
            return out
        self._draw_panel(out, snap)
        self._draw_voronoi_inset(out, snap, voronoi_width)
        self._update_ticker(snap)
        self._draw_ticker(out, snap.frame_idx)
        self._update_episode_banner(snap)
        self._draw_episode_banner(out, snap.frame_idx)
        return out

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _draw_panel(self, frame: np.ndarray, snap: AnalyticsSnapshot) -> None:
        a, b = snap.team_a, snap.team_b
        ca, cb = a.compactness, b.compactness
        poss = snap.possession

        def _line(label: str, va, vb) -> str:
            return f"{label:<11}{va:>9}{vb:>9}"

        lines = [
            ("HDR", "                 TEAM A   TEAM B"),
            ("SEP", ""),
            ("TXT", _line("Players",   ca.n_players,                cb.n_players)),
            ("TXT", _line("Width m",   f"{ca.width_m:.0f}",         f"{cb.width_m:.0f}")),
            ("TXT", _line("Depth m",   f"{ca.length_m:.0f}",        f"{cb.length_m:.0f}")),
            ("TXT", _line("Area m2",   f"{ca.hull_area_m2:.0f}",    f"{cb.hull_area_m2:.0f}")),
            ("TXT", _line("Def line",  _fmt(a.def_line_m),          _fmt(b.def_line_m))),
            ("TXT", _line("Control%",  f"{snap.voronoi['teamA_pct']:.0f}",
                                       f"{snap.voronoi['teamB_pct']:.0f}")),
            ("TXT", _line("Atk3rd%",   *_attacking_third_pct(snap))),
            ("TXT", _line("Formation", _fmt_formation(snap.formation_a),
                                       _fmt_formation(snap.formation_b))),
            ("SEP", ""),
            ("TXT", _line("Possess%",  f"{poss.pct_a:.0f}",         f"{poss.pct_b:.0f}")),
        ]
        if snap.pressing is not None:
            owner_lbl = "A" if poss.owner == "teamA" else "B"
            lines.append(("TXT",
                f"Press: {snap.pressing['intensity']} on {owner_lbl}  "
                f"(near {snap.pressing['nearest_dist_m']}m)"))
        if snap.visibility is not None:
            v = snap.visibility
            lines.append(("SEP", ""))
            lines.append(("TXT", f"View : {_regions_short(v.regions_visible)} ({v.coverage_pct:.0f}%)"))
            if snap.ball_region is not None:
                lines.append(("TXT", f"Ball : {snap.ball_region.replace('_',' ')}"))

        pad, line_h = 10, 20
        box_w = 290
        box_h = pad * 2 + line_h * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (kind, text) in enumerate(lines):
            y = 10 + pad + (i + 1) * line_h - 4
            if kind == "SEP":
                continue
            clr = _HUD if kind == "HDR" else (230, 230, 230)
            cv2.putText(frame, text, (10 + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, clr, 1, cv2.LINE_AA)

        # Possession bar under the panel
        self._draw_possession_bar(frame, 10 + pad, 10 + box_h + 6,
                                  box_w - 2 * pad, poss.pct_a, poss.pct_b)

    def _draw_possession_bar(self, frame, x, y, w, pct_a, pct_b) -> None:
        h = 14
        wa = int(w * pct_a / 100.0)
        cv2.rectangle(frame, (x, y), (x + wa, y + h), _A_CLR, -1)
        cv2.rectangle(frame, (x + wa, y), (x + w, y + h), _B_CLR, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(frame, f"{pct_a:.0f}%", (x + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{pct_b:.0f}%", (x + w - 34, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Voronoi inset ─────────────────────────────────────────────────────────

    def _draw_voronoi_inset(self, frame: np.ndarray, snap: AnalyticsSnapshot,
                            width: int) -> None:
        pitch = self._render_voronoi_pitch(snap)
        h0, w0 = pitch.shape[:2]
        height = int(width * h0 / w0)
        pitch = cv2.resize(pitch, (width, height), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        margin = 12
        x2, y2 = W - margin, margin + height
        x1, y1 = x2 - width, margin
        if x1 < 0 or y2 > H:
            return
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(pitch, 0.85, roi, 0.15, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _render_voronoi_pitch(self, snap: AnalyticsSnapshot) -> np.ndarray:
        img = self._base_pitch.copy()
        cells = snap.voronoi.get("cells", [])

        if cells:
            fill = img.copy()
            for team, poly_m in cells:
                pts = np.array(
                    [self.pm.pitch_to_pixels(x, y) for (x, y) in poly_m],
                    dtype=np.int32,
                )
                clr = config.VORONOI_A_COLOR if team == "teamA" else config.VORONOI_B_COLOR
                cv2.fillPoly(fill, [pts], clr)
            cv2.addWeighted(fill, 0.5, img, 0.5, 0, img)

        # Player dots on top
        for team, positions in snap.positions.items():
            clr = _A_CLR if team == "teamA" else _B_CLR
            for (x, y) in positions:
                cx, cy = self.pm.pitch_to_pixels(x, y)
                cv2.circle(img, (cx, cy), 6, clr, -1)
                cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

        # Role labels (LB, DM, RW, ...) next to whichever dots have a track_id
        for tid, (x, y) in snap.player_positions_m.items():
            role = snap.roles.get(tid)
            if role is None:
                continue
            cx, cy = self.pm.pitch_to_pixels(x, y)
            cv2.putText(img, role.role, (cx + 7, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

        if snap.ball_xy is not None:
            bx, by = self.pm.pitch_to_pixels(*snap.ball_xy)
            cv2.circle(img, (bx, by), 5, config.BALL_COLOR_BGR, -1)
            cv2.circle(img, (bx, by), 5, (0, 0, 0), 1)

        return img

    # ── Event ticker ──────────────────────────────────────────────────────────

    def _update_ticker(self, snap: AnalyticsSnapshot) -> None:
        """Add new events from this snap and keep only unexpired entries."""
        for ev in snap.events:
            self._ticker.appendleft((ev, snap.frame_idx + _TICKER_TTL))

    def _draw_ticker(self, frame: np.ndarray, current_frame: int) -> None:
        """Draw a slim event log at the bottom-left of the frame."""
        active = [(ev, exp) for ev, exp in self._ticker if exp > current_frame]
        if not active:
            return

        H, W = frame.shape[:2]
        line_h = 22
        pad    = 8
        box_h  = pad * 2 + line_h * len(active)
        y_base = H - 10 - box_h
        box_w  = 260

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_base), (10 + box_w, y_base + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (ev, exp) in enumerate(active):
            fade   = min(1.0, (exp - current_frame) / 40)   # fade out in last 40 frames
            bright = int(230 * fade)
            t_str  = f"{ev.time_s:5.1f}s  {ev.label()}"
            colour = (
                (_YELLOW[0], _YELLOW[1], bright)  # yellow-ish highlight for big events
                if ev.is_highlight else
                (bright, bright, bright)
            )
            y = y_base + pad + (i + 1) * line_h - 4
            cv2.putText(frame, t_str, (10 + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)

    # ── Tactical episode banner ───────────────────────────────────────────────

    def _update_episode_banner(self, snap: AnalyticsSnapshot) -> None:
        for ep in snap.closed_episodes:
            self._episode_banner.appendleft((ep, snap.frame_idx + _EPISODE_TTL))

    def _draw_episode_banner(self, frame: np.ndarray, current_frame: int) -> None:
        """Centered banner for the most recently closed tactical episode —
        rarer and more significant than a single event, so it gets its own
        spot (top-center) instead of competing with the per-event ticker."""
        active = [(ep, exp) for ep, exp in self._episode_banner if exp > current_frame]
        if not active:
            return
        ep, exp = active[0]
        fade = min(1.0, (exp - current_frame) / 50)
        tlabel = "TEAM A" if ep.team == "teamA" else "TEAM B"
        text = f"EPISODE #{ep.episode_id}  {tlabel}  {ep.narrative()}  -> {ep.outcome}"

        H, W = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x, y = (W - tw) // 2, 40

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 12, y - th - 10), (x + tw + 12, y + 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        clr = (int(230 * fade), int(120 * fade), int(230 * fade))   # magenta — distinct from yellow event highlights
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)


def _attacking_third_pct(snap: AnalyticsSnapshot) -> tuple[str, str]:
    """Each team's % control of their OWN attacking third, '--' if unknown."""
    sc = snap.space_control
    if sc is None or not sc.by_third:
        return "--", "--"
    n_thirds = 3   # matches space_control.py / overload.py default grid
    a_third = (n_thirds - 1) if snap.team_a.attack_dir == +1 else 0
    b_third = (n_thirds - 1) if snap.team_b.attack_dir == +1 else 0
    a_pair = sc.by_third.get(a_third) if snap.team_a.attack_dir != 0 else None
    b_pair = sc.by_third.get(b_third) if snap.team_b.attack_dir != 0 else None
    a_str = f"{a_pair[0]:.0f}" if a_pair else "--"
    b_str = f"{b_pair[1]:.0f}" if b_pair else "--"
    return a_str, b_str


def _fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:.0f}"


def _fmt_formation(fm) -> str:
    return fm.formation_str if fm is not None else "--"


def _regions_short(regions: list[str]) -> str:
    """['left_third','middle_third'] → 'L+M'."""
    m = {"left_third": "L", "middle_third": "M", "right_third": "R"}
    return "+".join(m.get(r, "?") for r in regions) or "--"
