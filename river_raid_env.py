"""
River Raid Gym-Style Environment in Pure Python + NumPy.
Contains all river generation, islands, crossing jets, collision physics,
7-ray Lidar sensor casting, and reward shaping.
(No fuel, ships, helicopters, or shooting, matching your custom rules.)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RiverSlice:
    y: float
    center: float
    width: float
    island_width: float
    is_road: bool = False


@dataclass
class CrossingJet:
    x: float
    y: float
    w: float
    h: float
    vx: float
    direction: int
    points: int = 100


@dataclass
class SceneryItem:
    x: float
    y: float
    kind: str  # "tree" or "house"


@dataclass
class RayHit:
    dist: float
    hit_x: float
    hit_y: float
    hit_type: str  # "NONE", "LAND", "JET"


class RiverRaidEnv:
    """Complete River Raid simulation & RL environment."""

    W: int = 600
    H: int = 720
    HUD_H: int = 88
    PLAY_H: int = 632
    SLICE_H: int = 4

    def __init__(self) -> None:
        self.score: int = 0
        self.high_score: int = 0
        self.sector: int = 1
        self.distance_in_sector: float = 0.0
        self.sector_length: float = 2600.0
        self.scroll_pos: float = 0.0
        self.done: bool = False

        # Player state
        self.player_x: float = self.W / 2
        self.player_y: float = self.PLAY_H - 72.0
        self.player_w: float = 28.0
        self.player_h: float = 28.0
        self.min_speed: float = 1.8
        self.max_speed: float = 5.8
        self.cruise_speed: float = 3.2
        self.player_speed: float = self.cruise_speed
        self.player_bank: int = 0

        # River procedural generator state
        self.gen_center: float = self.W / 2
        self.gen_target_center: float = self.W / 2
        self.gen_width: float = 290.0
        self.gen_target_width: float = 290.0
        self.gen_island_width: float = 0.0
        self.gen_target_island_width: float = 0.0
        self.gen_island_timer: int = 140
        self.gen_bridge_pending: bool = False
        self.gen_bridge_countdown: int = 0
        self.gen_spawn_cooldown: int = 35

        self.slices: List[RiverSlice] = []
        self.jets: List[CrossingJet] = []
        self.scenery: List[SceneryItem] = []

        # Sensor telemetry for visualization
        self.last_rays: List[RayHit] = []
        self.last_target_x: float = self.W / 2
        self.last_nearest_jet: Optional[CrossingJet] = None

        self.reset()

    def reset(self) -> np.ndarray:
        self.score = 0
        self.sector = 1
        self.distance_in_sector = 0.0
        self.scroll_pos = 0.0
        self.done = False

        self.player_x = self.W / 2
        self.player_y = self.PLAY_H - 72.0
        self.player_speed = self.cruise_speed
        self.player_bank = 0

        self.gen_center = self.W / 2
        self.gen_target_center = self.W / 2
        self.gen_width = 290.0
        self.gen_target_width = 290.0
        self.gen_island_width = 0.0
        self.gen_target_island_width = 0.0
        self.gen_island_timer = 140
        self.gen_bridge_pending = False
        self.gen_bridge_countdown = 0
        self.gen_spawn_cooldown = 35

        self.slices.clear()
        self.jets.clear()
        self.scenery.clear()

        num_slices = (self.PLAY_H // self.SLICE_H) + 5
        for i in range(num_slices):
            y = float(self.PLAY_H - i * self.SLICE_H)
            self.slices.append(
                RiverSlice(
                    y=y,
                    center=self.W / 2,
                    width=290.0,
                    island_width=0.0,
                    is_road=False,
                )
            )

        return self.get_observation()

    def _generate_top_slice(self) -> None:
        self.distance_in_sector += self.SLICE_H

        if self.distance_in_sector >= self.sector_length and not self.gen_bridge_pending:
            self.gen_bridge_pending = True
            self.gen_bridge_countdown = 45
            self.gen_target_center = self.W / 2
            self.gen_target_width = 116.0
            self.gen_target_island_width = 0.0

        if self.gen_bridge_pending:
            self.gen_bridge_countdown -= 1
            self.gen_center += (self.W / 2 - self.gen_center) * 0.12
            self.gen_width += (116.0 - self.gen_width) * 0.14
            self.gen_island_width = max(0.0, self.gen_island_width - 8.0)
        else:
            if random.random() < 0.04:
                margin = 130.0
                self.gen_target_center = margin + random.random() * (self.W - margin * 2.0)

            if random.random() < 0.045:
                min_w = max(125.0, 210.0 - self.sector * 10.0)
                max_w = 420.0
                self.gen_target_width = min_w + random.random() * (max_w - min_w)

            self.gen_island_timer -= 1
            if self.gen_island_timer <= 0:
                if (
                    self.gen_target_island_width == 0.0
                    and self.gen_width > 300.0
                    and random.random() < 0.45
                ):
                    self.gen_target_island_width = 65.0 + random.random() * 65.0
                    self.gen_island_timer = 90 + random.randint(0, 90)
                else:
                    self.gen_target_island_width = 0.0
                    self.gen_island_timer = 110 + random.randint(0, 120)

            if self.gen_width < 270.0:
                self.gen_target_island_width = 0.0

            self.gen_center += (self.gen_target_center - self.gen_center) * 0.035
            self.gen_width += (self.gen_target_width - self.gen_width) * 0.04
            self.gen_island_width += (self.gen_target_island_width - self.gen_island_width) * 0.06
            if self.gen_island_width < 6.0 and self.gen_target_island_width == 0.0:
                self.gen_island_width = 0.0

        half_w = self.gen_width / 2.0
        if self.gen_center - half_w < 36.0:
            self.gen_center = 36.0 + half_w
        if self.gen_center + half_w > self.W - 36.0:
            self.gen_center = self.W - 36.0 - half_w

        top_y = self.slices[-1].y - self.SLICE_H
        is_road_band = self.gen_bridge_pending and (6 <= self.gen_bridge_countdown <= 18)

        new_slice = RiverSlice(
            y=top_y,
            center=self.gen_center,
            width=self.gen_width,
            island_width=self.gen_island_width,
            is_road=is_road_band,
        )
        self.slices.append(new_slice)

        if not is_road_band and random.random() < 0.14:
            left_edge = new_slice.center - new_slice.width / 2.0
            right_edge = new_slice.center + new_slice.width / 2.0
            if random.random() < 0.5:
                sx = max(14.0, left_edge - 22.0 - random.random() * max(10.0, left_edge - 36.0))
            else:
                sx = min(
                    self.W - 24.0,
                    right_edge + 16.0 + random.random() * max(10.0, self.W - right_edge - 40.0),
                )
            self.scenery.append(
                SceneryItem(
                    x=sx,
                    y=top_y,
                    kind="tree" if random.random() < 0.78 else "house",
                )
            )

        if self.gen_bridge_pending and self.gen_bridge_countdown <= 0:
            self.gen_bridge_pending = False
            self.distance_in_sector = 0.0
            self.gen_target_width = 280.0
            self.sector += 1
            self._add_score(500)
        elif not self.gen_bridge_pending:
            self._spawn_jets(new_slice)

    def _spawn_jets(self, s: RiverSlice) -> None:
        self.gen_spawn_cooldown -= 1
        if self.gen_spawn_cooldown > 0:
            return

        from_left = random.random() < 0.5
        direction = 1 if from_left else -1
        vx = direction * (3.2 + min(2.2, self.sector * 0.25))
        self.jets.append(
            CrossingJet(
                x=20.0 if from_left else self.W - 20.0,
                y=s.y,
                w=30.0,
                h=18.0,
                vx=vx,
                direction=direction,
            )
        )
        base_gap = max(22.0, 48.0 - self.sector * 2.5)
        self.gen_spawn_cooldown = int(base_gap + random.random() * 25.0)

    def _add_score(self, pts: int) -> None:
        self.score += pts
        if self.score > self.high_score:
            self.high_score = self.score

    def get_slice_at_y(self, y: float) -> RiverSlice:
        if not self.slices:
            return RiverSlice(y, self.W / 2, 290.0, 0.0, False)
        # Slices are ordered from bottom (highest y) to top (lowest y)
        best = self.slices[0]
        best_diff = abs(best.y - y)
        for s in self.slices:
            d = abs(s.y - y)
            if d < best_diff:
                best_diff = d
                best = s
        return best

    def is_touching_land(self, x: float, y: float, half_w: float) -> bool:
        s = self.get_slice_at_y(y)
        left_bank = s.center - s.width / 2.0
        right_bank = s.center + s.width / 2.0
        if x - half_w < left_bank or x + half_w > right_bank:
            return True
        if s.island_width > 6.0:
            isl_left = s.center - s.island_width / 2.0
            isl_right = s.center + s.island_width / 2.0
            if x + half_w > isl_left and x - half_w < isl_right:
                return True
        return False

    def get_safe_center(self, s: RiverSlice, ref_x: float) -> float:
        left_bank = s.center - s.width / 2.0
        right_bank = s.center + s.width / 2.0
        if s.island_width > 6.0:
            isl_left = s.center - s.island_width / 2.0
            isl_right = s.center + s.island_width / 2.0
            left_center = (left_bank + isl_left) / 2.0
            right_center = (isl_right + right_bank) / 2.0
            return (
                left_center
                if abs(ref_x - left_center) <= abs(ref_x - right_center)
                else right_center
            )
        return s.center

    def get_safe_margin(self, x: float, y: float) -> float:
        s = self.get_slice_at_y(y)
        left_bank = s.center - s.width / 2.0
        right_bank = s.center + s.width / 2.0
        if s.island_width > 6.0:
            isl_left = s.center - s.island_width / 2.0
            isl_right = s.center + s.island_width / 2.0
            if x < s.center:
                return min(x - left_bank, isl_left - x)
            return min(x - isl_right, right_bank - x)
        return min(x - left_bank, right_bank - x)

    def _cast_ray(self, angle_rad: float, max_dist: float = 260.0) -> RayHit:
        step = 10.0
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)
        d = step
        while d <= max_dist:
            rx = self.player_x + dx * d
            ry = self.player_y + dy * d
            if rx < 0 or rx > self.W or ry < 0 or ry > self.PLAY_H:
                return RayHit(d / max_dist, rx, ry, "LAND")
            if self.is_touching_land(rx, ry, 4.0):
                return RayHit(d / max_dist, rx, ry, "LAND")
            for j in self.jets:
                if abs(rx - j.x) < j.w * 0.6 and abs(ry - j.y) < j.h * 0.8:
                    return RayHit(d / max_dist, rx, ry, "JET")
            d += step
        return RayHit(
            1.0,
            self.player_x + dx * max_dist,
            self.player_y + dy * max_dist,
            "NONE",
        )

    def get_observation(self) -> np.ndarray:
        angles_deg = (-75, -50, -25, 0, 25, 50, 75)
        obs = np.zeros(13, dtype=np.float32)
        rays: List[RayHit] = []

        for i, deg in enumerate(angles_deg):
            r = self._cast_ray(math.radians(deg), 260.0)
            rays.append(r)
            obs[i] = r.dist
        self.last_rays = rays

        slice_now = self.get_slice_at_y(self.player_y)
        slice_near = self.get_slice_at_y(self.player_y - 70.0)
        slice_far = self.get_slice_at_y(self.player_y - 160.0)

        safe_now = self.get_safe_center(slice_now, self.player_x)
        safe_near = self.get_safe_center(slice_near, safe_now)
        safe_far = self.get_safe_center(slice_far, safe_near)
        self.last_target_x = safe_near

        obs[7] = (safe_now - self.player_x) / (self.W * 0.5)
        obs[8] = (safe_near - self.player_x) / (self.W * 0.5)
        obs[9] = (safe_far - self.player_x) / (self.W * 0.5)

        nearest_jet: Optional[CrossingJet] = None
        best_dy = 9999.0
        for j in self.jets:
            dy = self.player_y - j.y
            if -24.0 < dy < 280.0 and dy < best_dy:
                best_dy = dy
                nearest_jet = j
        self.last_nearest_jet = nearest_jet

        if nearest_jet is not None:
            obs[10] = (nearest_jet.x - self.player_x) / (self.W * 0.5)
            obs[11] = (self.player_y - nearest_jet.y) / 280.0
        else:
            obs[10] = 0.0
            obs[11] = 1.0

        obs[12] = (self.player_speed - self.min_speed) / (self.max_speed - self.min_speed)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one physics step.
        Actions: 0=COAST, 1=LEFT, 2=RIGHT, 3=FAST, 4=SLOW
        """
        self.player_bank = 0
        if action == 1:
            self.player_x -= 4.2
            self.player_bank = -1
        elif action == 2:
            self.player_x += 4.2
            self.player_bank = 1

        if action == 3:
            self.player_speed += (self.max_speed - self.player_speed) * 0.12
        elif action == 4:
            self.player_speed += (self.min_speed - self.player_speed) * 0.14
        else:
            self.player_speed += (self.cruise_speed - self.player_speed) * 0.10

        speed_ratio = (self.player_speed - self.min_speed) / (self.max_speed - self.min_speed)

        if int(self.scroll_pos // 20) < int((self.scroll_pos + self.player_speed) // 20):
            self._add_score(10)

        self.scroll_pos += self.player_speed
        for s in self.slices:
            s.y += self.player_speed
        while self.slices and self.slices[0].y > self.PLAY_H + self.SLICE_H * 2:
            self.slices.pop(0)
        while self.slices[-1].y > -self.SLICE_H * 2:
            self._generate_top_slice()

        for item in self.scenery:
            item.y += self.player_speed
        self.scenery = [item for item in self.scenery if item.y <= self.PLAY_H + 30]

        crashed = self.is_touching_land(self.player_x, self.player_y, self.player_w * 0.42)

        surviving_jets: List[CrossingJet] = []
        for j in self.jets:
            j.y += self.player_speed
            j.x += j.vx
            if j.x < -30.0:
                j.x = self.W + 30.0
            elif j.x > self.W + 30.0:
                j.x = -30.0

            if not crashed:
                if (
                    abs(self.player_x - j.x) * 2.0 < (self.player_w * 0.78 + j.w * 0.85)
                    and abs(self.player_y - j.y) * 2.0 < (self.player_h * 0.78 + j.h * 0.85)
                ):
                    crashed = True

            if j.y <= self.PLAY_H + 60.0:
                surviving_jets.append(j)
        self.jets = surviving_jets

        next_obs = self.get_observation()

        if crashed:
            self.done = True
            return next_obs, -12.0, True, {"score": self.score, "sector": self.sector}

        # Reward shaping: stay centered in safe channel + forward speed - bank/jet danger
        margin = self.get_safe_margin(self.player_x, self.player_y)
        slice_near = self.get_slice_at_y(self.player_y - 65.0)
        safe_center = self.get_safe_center(slice_near, self.player_x)
        center_error = abs(self.player_x - safe_center) / 140.0

        reward = 0.25 + (1.0 - min(1.2, center_error)) * 0.35 + speed_ratio * 0.1
        if margin < 32.0:
            reward -= (32.0 - margin) * 0.04

        if self.last_nearest_jet is not None:
            j = self.last_nearest_jet
            dy = self.player_y - j.y
            dx = abs(self.player_x - j.x)
            if 0.0 < dy < 75.0 and dx < 42.0:
                reward -= 0.45

        return next_obs, float(reward), False, {"score": self.score, "sector": self.sector}
