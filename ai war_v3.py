import pygame
import random
import math
import pickle
import time

WIDTH, HEIGHT = 1100, 750
WORLD_SIZE = 1000
START_AGENTS = 100
TREE_COUNT = 25
MAX_AGENTS = 1000
SEED = 1454
random.seed(SEED)
FPS = 30

BASE_METABOLISM = 0.05
MOVE_COST = 0.0009
REPRO_THRESHOLD = 90
REPRO_COST = 45
EAT_RADIUS = 12
EAT_RATE = 0.8
TREE_MAX_FOOD = 30
TREE_MIN_FOOD = 10
TREE_REGEN_RATE = 0.02
ATTACK_RADIUS = 14
ATTACK_COOLDOWN_TICKS = 16
ATTACK_DAMAGE = 9.0
ATTACK_COST = 0.2
ATTACK_STEAL = 0.35
PUSH_FORCE = 0.9

KILL_REWARD = 25  # <-- added kill reward

WATER_DRAG = 0.55
MOUNTAIN_DRAG = 0.80
LAND_DRAG = 0.88
WATER_ENERGY_MULT = 1.30
MOUNTAIN_ENERGY_MULT = 1.45
START_ZOOM = 1.0
MIN_ZOOM, MAX_ZOOM = 0.25, 3.5
CAMERA_SPEED_BASE = 12.0
CHUNK_STEP = 60
GRID_CELL = 70
TREE_GRID_CELL = 120
GENOME_LEN = 3
ENEMY_THRESHOLD = 0.35
NN_INPUTS = 12
NN_HIDDEN = 24
NN_OUTPUTS = 6

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolving Creatures - Fast Mode")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

### KILL PARTICLES ADDED
kill_particles = []


def hash2d(x, y, seed=SEED):
    n = int(x) * 374761393 + int(y) * 668265263 + seed * 1442695040888963407
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return n & 0xffffffff


def value_noise(x, y, scale=0.0025, octaves=3, lacunarity=2.0, gain=0.5):
    amp = 1.0
    freq = scale
    total = 0.0
    norm = 0.0
    for _ in range(octaves):
        xi = int(math.floor(x * freq))
        yi = int(math.floor(y * freq))
        tx = x * freq - xi
        ty = y * freq - yi

        def rnd(ix, iy):
            return (hash2d(ix, iy) / 0xffffffff)

        v00 = rnd(xi, yi)
        v10 = rnd(xi + 1, yi)
        v01 = rnd(xi, yi + 1)
        v11 = rnd(xi + 1, yi + 1)
        vx0 = v00 * (1 - tx) + v10 * tx
        vx1 = v01 * (1 - ty) + v11 * ty
        v = vx0 * (1 - ty) + vx1 * ty
        total += v * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / (norm + 1e-9)


def terrain_value(x, y):
    n1 = value_noise(x + 10000, y + 10000, scale=0.0018, octaves=4)
    n2 = value_noise(x - 25000, y - 25000, scale=0.0008, octaves=3)
    n = 0.6 * n1 + 0.4 * n2
    return (n - 0.5) * 1.4


def terrain_type(x, y):
    n = terrain_value(x, y)
    if n < -0.18: return "water"
    if n > 0.25: return "mountain"
    return "land"


def terrain_color(x, y):
    t = terrain_type(x, y)
    if t == "water":
        v = terrain_value(x, y)
        d = min(1, max(0, (-0.18 - v) * 3.5))
        b = (40, 95, 170)
        c = (18, 55, 120)
        return (int(b[0] * (1 - d) + c[0] * d),
                int(b[1] * (1 - d) + c[1] * d),
                int(b[2] * (1 - d) + c[2] * d))
    if t == "mountain":
        v = terrain_value(x, y)
        h = min(1, max(0, (v - 0.25) * 2))
        b = (120, 120, 120)
        c = (235, 235, 235)
        return (int(b[0] * (1 - h) + c[0] * h),
                int(b[1] * (1 - h) + c[1] * h),
                int(b[2] * (1 - h) + c[2] * h))
    v = terrain_value(x, y)
    f = min(1, max(0, (0.25 - abs(v)) * 2))
    d = (110, 150, 90)
    l = (75, 185, 100)
    return (int(d[0] * (1 - f) + l[0] * f),
            int(d[1] * (1 - f) + l[1] * f),
            int(d[2] * (1 - f) + l[2] * f))


def terrain_effects(x, y):
    t = terrain_type(x, y)
    if t == "water": return WATER_DRAG, WATER_ENERGY_MULT
    if t == "mountain": return MOUNTAIN_DRAG, MOUNTAIN_ENERGY_MULT
    return LAND_DRAG, 1.0


def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def vec_norm(x, y):
    d = math.hypot(x, y)
    if d < 1e-9: return 0, 0, 0
    return x / d, y / d, d


def cell_id(x, y, c):
    return (int(x) // c, int(y) // c)


def rebuild_agent_grid(agents):
    g = {}
    for i, a in enumerate(agents):
        c = cell_id(a.x, a.y, GRID_CELL)
        g.setdefault(c, []).append(i)
    return g


def rebuild_tree_grid(trees):
    g = {}
    for i, t in enumerate(trees):
        c = cell_id(t.x, t.y, TREE_GRID_CELL)
        g.setdefault(c, []).append(i)
    return g


def neighbor_cells(cx, cy):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (cx + dx, cy + dy)


class Brain:
    def __init__(self, i=NN_INPUTS, h=NN_HIDDEN, o=NN_OUTPUTS):
        self.i = i;
        self.h = h;
        self.o = o
        self.w1 = [[random.uniform(-1, 1) for _ in range(i)] for _ in range(h)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(h)] for _ in range(o)]

    def forward(self, inps):
        h = [0] * self.h
        for hh in range(self.h):
            s = 0;
            r = self.w1[hh]
            for ii in range(self.i): s += r[ii] * inps[ii]
            h[hh] = math.tanh(s)

        out = [0] * self.o
        for oo in range(self.o):
            s = 0;
            r = self.w2[oo]
            for hh in range(self.h): s += r[hh] * h[hh]
            out[oo] = math.tanh(s)
        return out

    def mutate(self, rate=0.08, mag=0.25):
        for hh in range(self.h):
            for ii in range(self.i):
                if random.random() < rate:
                    self.w1[hh][ii] += random.uniform(-mag, mag)
        for oo in range(self.o):
            for hh in range(self.h):
                if random.random() < rate:
                    self.w2[oo][hh] += random.uniform(-mag, mag)


class Tree:
    def __init__(self):
        self.x = random.uniform(0, WORLD_SIZE)
        self.y = random.uniform(0, WORLD_SIZE)
        for _ in range(10):
            if terrain_type(self.x, self.y) != "water":
                break
            self.x = random.uniform(0, WORLD_SIZE)
            self.y = random.uniform(0, WORLD_SIZE)
        self.food = random.uniform(TREE_MIN_FOOD, TREE_MAX_FOOD)

    def update(self):
        if 0 < self.food < TREE_MAX_FOOD:
            self.food = min(TREE_MAX_FOOD, self.food + TREE_REGEN_RATE)

    def respawn(self):
        self.x = random.uniform(0, WORLD_SIZE)
        self.y = random.uniform(0, WORLD_SIZE)
        for _ in range(10):
            if terrain_type(self.x, self.y) != "water":
                break
            self.x = random.uniform(0, WORLD_SIZE)
            self.y = random.uniform(0, WORLD_SIZE)
        self.food = random.uniform(TREE_MIN_FOOD, TREE_MAX_FOOD)


class Agent:
    def __init__(self, x, y, brain=None, genome=None):
        self.x = x;
        self.y = y
        self.vx = 0;
        self.vy = 0
        self.energy = 55 + random.random() * 25
        self.age = 0;
        self.cooldown = 0
        self.brain = brain if brain else Brain()
        if genome is None:
            self.genome = [random.random() for _ in range(GENOME_LEN)]
        else:
            self.genome = genome
        self.color = (
            int(self.genome[0] * 255),
            int(self.genome[1] * 255),
            int(self.genome[2] * 255)
        )

    def genome_dist(self, o):
        return math.dist(self.genome, o.genome)

    def is_enemy(self, o):
        return self.genome_dist(o) > ENEMY_THRESHOLD

    def sample_terrain_ahead(self):
        nx, ny, d = vec_norm(self.vx, self.vy)
        if d < 0.05:
            nx, ny = 0, -1
        lx = self.x + nx * 22
        ly = self.y + ny * 22
        t = terrain_type(lx, ly)
        return (
            1 if t == "water" else 0,
            1 if t == "land" else 0,
            1 if t == "mountain" else 0
        )

    def find_nearest_tree_chunked(self, trees, grid):
        cx, cy = cell_id(self.x, self.y, TREE_GRID_CELL)
        bi = None;
        bd = 1e18
        for nc in neighbor_cells(cx, cy):
            for ti in grid.get(nc, []):
                t = trees[ti]
                dx = t.x - self.x;
                dy = t.y - self.y
                d2 = dx * dx + dy * dy
                if d2 < bd:
                    bd = d2;
                    bi = ti
        if bi is None: return None, 1e9, 0, 0
        t = trees[bi]
        nx, ny, d = vec_norm(t.x - self.x, t.y - self.y)
        return t, d, nx, ny

    def find_nearest_agent_chunked(self, agents, grid):
        cx, cy = cell_id(self.x, self.y, GRID_CELL)
        bi = None;
        bd = 1e18
        for nc in neighbor_cells(cx, cy):
            for ai in grid.get(nc, []):
                o = agents[ai]
                if o is self: continue
                dx = o.x - self.x;
                dy = o.y - self.y
                d2 = dx * dx + dy * dy
                if d2 < bd:
                    bd = d2;
                    bi = ai
        if bi is None: return None, 1e9, 0, 0, 0
        o = agents[bi]
        nx, ny, d = vec_norm(o.x - self.x, o.y - self.y)
        return o, d, nx, ny, (1 if self.is_enemy(o) else 0)

    def update(self, agents, trees, agrid, tgrid):
        self.age += 1
        if self.cooldown > 0:
            self.cooldown -= 1

        drag, mt = terrain_effects(self.x, self.y)

        tr, td, fx, fy = self.find_nearest_tree_chunked(trees, tgrid)
        fd = clamp(td / 500, 0, 1)

        ot, od, ox, oy, en = self.find_nearest_agent_chunked(agents, agrid)
        odn = clamp(od / 260, 0, 1)

        w, l, m = self.sample_terrain_ahead()

        inp = [
            clamp(self.energy / 140, 0, 1),
            fd,
            fx, fy,
            odn,
            ox, oy,
            en,
            w, l, m,
            random.uniform(-1, 1)
        ]

        mx, my, eatp, repp, attp, sb = self.brain.forward(inp)

        acc = 0.55 + 0.35 * max(0, sb)
        self.vx += mx * acc
        self.vy += my * acc
        self.x += self.vx
        self.y += self.vy
        self.vx *= drag;
        self.vy *= drag

        self.x = clamp(self.x, 0, WORLD_SIZE)
        self.y = clamp(self.y, 0, WORLD_SIZE)

        sp = math.hypot(self.vx, self.vy)
        self.energy -= BASE_METABOLISM * mt + MOVE_COST * sp * 70

        if tr and eatp > 0.15 and tr.food > 0:
            if td < EAT_RADIUS:
                e = min(EAT_RATE, tr.food)
                tr.food -= e
                self.energy += e * 6
                if tr.food <= 0.01:
                    tr.food = 0
                    tr.respawn()

        if ot and attp > 0.35 and self.cooldown == 0 and self.energy > 5:
            if od < ATTACK_RADIUS:
                self.energy -= ATTACK_COST
                self.cooldown = ATTACK_COOLDOWN_TICKS
                dmg = ATTACK_DAMAGE * (1 if self.is_enemy(ot) else 0.4)
                ot.energy -= dmg
                self.energy += dmg * ATTACK_STEAL

                nx, ny, d = vec_norm(ot.x - self.x, ot.y - self.y)
                if d > 0:
                    ot.vx += nx * PUSH_FORCE
                    ot.vy += ny * PUSH_FORCE

                ### KILL PARTICLES + REWARD ADDED
                if ot.energy <= 0:
                    self.energy += KILL_REWARD
                    for _ in range(12):
                        ang = random.uniform(0, math.tau)
                        spd = random.uniform(1, 2)
                        kill_particles.append([
                            ot.x, ot.y,
                            math.cos(ang) * spd,
                            math.sin(ang) * spd,
                            random.randint(12, 25)
                        ])

        if repp > 0.55 and self.energy > REPRO_THRESHOLD and len(agents) < MAX_AGENTS:
            self.energy -= REPRO_COST
            br = pickle.loads(pickle.dumps(self.brain))
            br.mutate(0.08, 0.28)
            g = self.genome[:]
            for i in range(GENOME_LEN):
                if random.random() < 0.35:
                    g[i] = clamp(g[i] + random.uniform(-0.12, 0.12), 0, 1)
            agents.append(Agent(
                self.x + random.uniform(-6, 6),
                self.y + random.uniform(-6, 6),
                brain=br,
                genome=g
            ))

        return self.energy > 0


def save_world(agents, trees, camx, camy, zoom):
    d = {
        "agents": [(a.x, a.y, a.vx, a.vy, a.energy, a.age, a.cooldown, a.genome, a.brain) for a in agents],
        "trees": [(t.x, t.y, t.food) for t in trees],
        "camera": (camx, camy, zoom),
        "meta": {"seed": SEED, "time": time.time()}
    }
    with open("world.save", "wb") as f:
        pickle.dump(d, f)


def load_world():
    try:
        with open("world.save", "rb") as f:
            d = pickle.load(f)
        agents = []
        for x, y, vx, vy, en, ag, cd, ge, br in d["agents"]:
            a = Agent(x, y, brain=br, genome=ge)
            a.vx = vx;
            a.vy = vy;
            a.energy = en;
            a.age = ag;
            a.cooldown = cd
            agents.append(a)
        trees = []
        for x, y, f in d["trees"]:
            t = Tree();
            t.x = x;
            t.y = y;
            t.food = f;
            trees.append(t)
        camx, camy, zoom = d["camera"]
        return agents, trees, camx, camy, zoom
    except:
        return None


trees = [Tree() for _ in range(TREE_COUNT)]
agents = [Agent(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE)) for _ in range(START_AGENTS)]
cam_x = WORLD_SIZE / 2
cam_y = WORLD_SIZE / 2
zoom = 1.0

fast_modes = [1, 2, 4, 8, 16, 32]
fast_i = 0

running = True
while running:
    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            if e.key == pygame.K_f:
                save_world(agents, trees, cam_x, cam_y, zoom)
            if e.key == pygame.K_l:
                w = load_world()
                if w: agents, trees, cam_x, cam_y, zoom = w
            if e.key == pygame.K_x:
                fast_i = (fast_i + 1) % len(fast_modes)
        elif e.type == pygame.MOUSEWHEEL:
            if e.y > 0: zoom *= 1.2
            else: zoom *= 0.8
            zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM)

    keys = pygame.key.get_pressed()
    cs = CAMERA_SPEED_BASE / zoom
    if keys[pygame.K_w]: cam_y -= cs
    if keys[pygame.K_s]: cam_y += cs
    if keys[pygame.K_a]: cam_x -= cs
    if keys[pygame.K_d]: cam_x += cs
    cam_x = clamp(cam_x, 0, WORLD_SIZE)
    cam_y = clamp(cam_y, 0, WORLD_SIZE)

    steps = fast_modes[fast_i]
    for _ in range(steps):
        for t in trees: t.update()
        agrid = rebuild_agent_grid(agents)
        tgrid = rebuild_tree_grid(trees)
        alive = []
        for a in agents:
            if a.update(agents, trees, agrid, tgrid):
                alive.append(a)
        agents = alive
        if len(agents) < 40:
            for _ in range(25):
                agents.append(Agent(random.uniform(0, WORLD_SIZE),
                                    random.uniform(0, WORLD_SIZE)))

    screen.fill((10, 10, 16))
    step = CHUNK_STEP
    vw = WIDTH / (2 * zoom);
    vh = HEIGHT / (2 * zoom)
    sx = int((cam_x - vw) // step * step)
    ex = int((cam_x + vw) // step * step + step)
    sy = int((cam_y - vh) // step * step)
    ey = int((cam_y + vh) // step * step + step)
    for gx in range(sx, ex, step):
        for gy in range(sy, ey, step):
            if 0 <= gx <= WORLD_SIZE and 0 <= gy <= WORLD_SIZE:
                col = terrain_color(gx + step * 0.5, gy + step * 0.5)
                px = (gx - cam_x) * zoom + WIDTH / 2
                py = (gy - cam_y) * zoom + HEIGHT / 2
                pygame.draw.rect(screen, col, (px, py, step * zoom + 1, step * zoom + 1))

    for t in trees:
        px = (t.x - cam_x) * zoom + WIDTH / 2
        py = (t.y - cam_y) * zoom + HEIGHT / 2
        if -50 < px < WIDTH + 50 and -50 < py < HEIGHT + 50:
            r = max(2, int((3 + 6 * (t.food / TREE_MAX_FOOD)) * zoom))
            b = (35, 95, 45);
            c = (55, 165, 70)
            k = clamp(t.food / TREE_MAX_FOOD, 0, 1)
            col = (
                int(b[0] * (1 - k) + c[0] * k),
                int(b[1] * (1 - k) + c[1] * k),
                int(b[2] * (1 - k) + c[2] * k)
            )
            pygame.draw.circle(screen, col, (int(px), int(py)), r)

    for a in agents:
        px = (a.x - cam_x) * zoom + WIDTH / 2
        py = (a.y - cam_y) * zoom + HEIGHT / 2
        if -30 < px < WIDTH + 30 and -30 < py < HEIGHT + 30:
            r = max(2, int(3.2 * zoom))
            pygame.draw.circle(screen, (10, 10, 10), (int(px), int(py)), r + 1)
            pygame.draw.circle(screen, a.color, (int(px), int(py)), r)

    ### SAFE PARTICLE UPDATE LOOP ADDED
    for p in kill_particles[:]:
        x, y, vx, vy, life = p
        x += vx
        y += vy
        life -= 1

        px = (x - cam_x) * zoom + WIDTH / 2
        py = (y - cam_y) * zoom + HEIGHT / 2
        if 0 < life:
            pygame.draw.circle(screen, (255, 80, 60), (int(px), int(py)), max(1, int(2 * zoom)))
            p[0] = x
            p[1] = y
            p[4] = life
        else:
            kill_particles.remove(p)

    hud = f"Agents:{len(agents)} Trees:{len(trees)} Zoom:{zoom:.2f} Speed:{fast_modes[fast_i]}x"
    screen.blit(font.render(hud, True, (235, 235, 235)), (10, 8))
    screen.blit(font.render('X = cycle speed | WASD move | wheel zoom | F save | L load | ESC quit',
                            True, (210, 210, 210)), (10, 28))

    pygame.display.flip()

pygame.quit()
