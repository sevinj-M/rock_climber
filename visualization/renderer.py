import pygame
import numpy as np
import time
from env.climbing_env import ClimbingEnv


# ── Visual constants ───────────────────────────────────────────
CELL        = 80          # pixels per grid cell
MARGIN      = 60          # border padding
FPS         = 6           # steps per second (lower = easier to follow)
PANEL_W     = 260         # right-side info panel width

# Colors
BG          = (245, 242, 235)
WALL_BG     = (225, 220, 210)
HOLD_COLOR  = (120, 100,  80)
HOLD_RING   = (160, 140, 110)
TARGET_COL  = (220, 160,  20)
LH_COLOR    = (60,  120, 200)   # left hand  — blue
RH_COLOR    = (200,  80,  60)   # right hand — red
LF_COLOR    = (60,  170, 100)   # left foot  — green
RF_COLOR    = (180,  80, 180)   # right foot — purple
COM_COLOR   = (255, 200,   0)
BODY_COLOR  = (60,   60,  60)
TEXT_DARK   = (40,   40,  40)
TEXT_MUTED  = (130, 120, 110)
PANEL_BG    = (235, 230, 220)
SUCCESS_COL = (60,  180,  80)
FAIL_COL    = (200,  60,  60)

LIMB_COLORS = [LH_COLOR, RH_COLOR, LF_COLOR, RF_COLOR]
LIMB_LABELS = ["LH", "RH", "LF", "RF"]


class ClimbingRenderer:
    """
    Pygame renderer for ClimbingEnv.

    Usage:
        renderer = ClimbingRenderer(env)
        renderer.render(agent, n_episodes=5, fps=6)
        renderer.close()
    """

    def __init__(self, env: ClimbingEnv):
        self.env = env

        # Compute window size from wall dimensions
        self.wall_px_w = env.W * CELL
        self.wall_px_h = env.H * CELL
        self.win_w     = self.wall_px_w + 2 * MARGIN + PANEL_W
        self.win_h     = self.wall_px_h + 2 * MARGIN

        pygame.init()
        pygame.display.set_caption("Rock Climber RL")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 16)
        self.font_sm  = pygame.font.SysFont("monospace", 13)

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, agent, n_episodes: int = 5, fps: int = FPS):
        """
        Run n_episodes using the agent (greedy, ε=0) and render each step.
        Press Q or close the window to exit early.
        """
        agent.epsilon = 0.0
        results = []

        for ep in range(1, n_episodes + 1):
            obs, _    = self.env.reset()
            done      = truncated = False
            ep_reward = 0.0
            steps     = 0
            success   = False
            route     = []   # list of (limb_id, hold_pos) for trail drawing

            while not (done or truncated):
                # ── Event pump ─────────────────────────────────────────────────
                if self._should_quit():
                    pygame.quit()
                    return results

                # ── Agent step ─────────────────────────────────────────────────
                mask   = self.env.get_valid_action_mask()
                action = agent.act(obs, mask)
                limb_id, hold_id = divmod(action, self.env.n_holds)
                route.append((limb_id, self.env.holds[hold_id]))

                obs, reward, done, truncated, info = self.env.step(action)
                ep_reward += reward
                steps     += 1
                if info.get("reached_target"):
                    success = True

                # ── Draw ───────────────────────────────────────────────────────
                self._draw(ep, n_episodes, steps, ep_reward,
                           success, done or truncated, route)
                self.clock.tick(fps)

            # ── Episode end flash ──────────────────────────────────────────────
            self._flash_result(success)
            results.append({
                "episode": ep,
                "reward":  ep_reward,
                "steps":   steps,
                "success": success,
            })
            time.sleep(0.4)

        return results

    def close(self):
        pygame.quit()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self, ep, n_ep, steps, reward, success, done, route):
        self.screen.fill(BG)
        self._draw_wall()
        self._draw_route_trail(route)
        self._draw_holds()
        self._draw_agent()
        self._draw_panel(ep, n_ep, steps, reward, done, success)
        pygame.display.flip()

    def _draw_wall(self):
        """Filled background rect for the wall area."""
        rect = pygame.Rect(
            MARGIN, MARGIN,
            self.wall_px_w, self.wall_px_h
        )
        pygame.draw.rect(self.screen, WALL_BG, rect, border_radius=8)

    def _draw_holds(self):
        """Draw every hold as a circle. Target hold gets a star ring."""
        for i, (x, y) in enumerate(self.env.holds):
            px, py = self._to_px(x, y)
            is_target = (i == self.env.target_hold)
            color     = TARGET_COL if is_target else HOLD_COLOR
            ring      = TARGET_COL if is_target else HOLD_RING

            pygame.draw.circle(self.screen, ring,  (px, py), 14)
            pygame.draw.circle(self.screen, color, (px, py), 10)

            if is_target:
                self._draw_star(px, py, outer=20, inner=9, color=TARGET_COL)

    def _draw_agent(self):
        """Draw limbs as colored dots, body as lines between them, COM as diamond."""
        limb_positions = []
        for limb, hold_idx in enumerate(self.env.limb_holds):
            if hold_idx is None:
                continue
            x, y   = self.env.holds[hold_idx]
            px, py = self._to_px(x, y)
            limb_positions.append((px, py))
            color = LIMB_COLORS[limb]

            # Outer glow ring
            pygame.draw.circle(self.screen, color, (px, py), 16, 3)
            # Filled dot
            pygame.draw.circle(self.screen, color, (px, py), 11)
            # Label
            lbl = self.font_sm.render(LIMB_LABELS[limb], True, (255, 255, 255))
            self.screen.blit(lbl, (px - lbl.get_width() // 2,
                                   py - lbl.get_height() // 2))

        # Body lines between hands and feet
        self._draw_body_lines()

        # Center of mass diamond
        active = [self.env.holds[h] for h in self.env.limb_holds if h is not None]
        if active:
            com  = np.mean(active, axis=0)
            cpx, cpy = self._to_px(*com)
            self._draw_diamond(cpx, cpy, size=8, color=COM_COLOR)

    def _draw_body_lines(self):
        """Connect LH–LF and RH–RF with thin body lines."""
        pairs = [(self.env.LH, self.env.LF), (self.env.RH, self.env.RF),
                 (self.env.LH, self.env.RH), (self.env.LF, self.env.RF)]
        for a, b in pairs:
            ha, hb = self.env.limb_holds[a], self.env.limb_holds[b]
            if ha is None or hb is None:
                continue
            pxa, pya = self._to_px(*self.env.holds[ha])
            pxb, pyb = self._to_px(*self.env.holds[hb])
            pygame.draw.line(self.screen, BODY_COLOR,
                             (pxa, pya), (pxb, pyb), 2)

    def _draw_route_trail(self, route: list):
        """Faint trail showing where each limb has been."""
        for limb_id, (x, y) in route:
            px, py = self._to_px(x, y)
            color  = (*LIMB_COLORS[limb_id][:3], 40)
            surf   = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*LIMB_COLORS[limb_id], 40), (5, 5), 5)
            self.screen.blit(surf, (px - 5, py - 5))

    def _draw_panel(self, ep, n_ep, steps, reward, done, success):
        """Right-side info panel."""
        px = self.wall_px_w + 2 * MARGIN
        py = MARGIN

        # Panel background
        panel_rect = pygame.Rect(px - 10, py - 10, PANEL_W, self.wall_px_h + 20)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=8)

        def write(text, y_off, color=TEXT_DARK, font=None):
            font = font or self.font_md
            surf = font.render(text, True, color)
            self.screen.blit(surf, (px + 10, py + y_off))

        write(f"Episode {ep} / {n_ep}",        0,   font=self.font_lg)
        write(f"Steps  : {steps}",             36)
        write(f"Reward : {reward:>7.3f}",      58)

        # Limb hold info
        write("Limbs:",                        95,  color=TEXT_MUTED)
        for i, (label, color) in enumerate(zip(LIMB_LABELS, LIMB_COLORS)):
            h = self.env.limb_holds[i]
            pos = f"{self.env.holds[h]}" if h is not None else "—"
            write(f"  {label}: {pos}", 115 + i * 20, color=color)

        # Target
        tx, ty = self.env.holds[self.env.target_hold]
        write(f"Target : ({tx:.0f}, {ty:.0f})", 200, color=TARGET_COL)

        # Status
        if done:
            if success:
                write("✓ REACHED TARGET", 240, color=SUCCESS_COL, font=self.font_lg)
            else:
                write("✗ FELL",           240, color=FAIL_COL,    font=self.font_lg)

        # Legend
        write("Legend:",                   290, color=TEXT_MUTED)
        write("● LH — left hand",          310, color=LH_COLOR)
        write("● RH — right hand",         330, color=RH_COLOR)
        write("● LF — left foot",          350, color=LF_COLOR)
        write("● RF — right foot",         370, color=RF_COLOR)
        write("◆ COM — center of mass",    390, color=COM_COLOR)
        write("★ Target hold",             410, color=TARGET_COL)

    def _flash_result(self, success: bool):
        """Brief full-screen color flash on episode end."""
        color  = (*SUCCESS_COL, 80) if success else (*FAIL_COL, 80)
        flash  = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        flash.fill(color)
        self.screen.blit(flash, (0, 0))
        pygame.display.flip()
        time.sleep(0.25)

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        """Convert grid coordinates to screen pixels. y=0 at bottom."""
        px = int(MARGIN + x * CELL + CELL // 2)
        py = int(MARGIN + (self.env.H - 1 - y) * CELL + CELL // 2)
        return px, py

    def _draw_star(self, cx, cy, outer, inner, color, points=5):
        angles = [
            (i * 2 * np.pi / points - np.pi / 2) + (np.pi / points) * (i % 2)
            for i in range(points * 2)
        ]
        radii  = [outer if i % 2 == 0 else inner for i in range(points * 2)]
        verts  = [(cx + r * np.cos(a), cy + r * np.sin(a))
                  for r, a in zip(radii, angles)]
        pygame.draw.polygon(self.screen, color, verts)

    def _draw_diamond(self, cx, cy, size, color):
        points = [(cx, cy - size), (cx + size, cy),
                  (cx, cy + size), (cx - size, cy)]
        pygame.draw.polygon(self.screen, color, points)

    def _should_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return True
        return False