
import pygame
import random
import math
import pickle
import time

# ------------------------ Config ------------------------
WIDTH, HEIGHT = 1100, 750
WORLD_SIZE = 1000

START_AGENTS = 100
TREE_COUNT = 30
MAX_AGENTS = 2000

SEED = 12378
random.seed(SEED)

FPS = 120

# Energy & behavior
BASE_METABOLISM = 0.05
MOVE_COST = 0.0009

REPRO_THRESHOLD = 90
REPRO_COST = 45

EAT_RADIUS = 12
EAT_RATE = 0.8
TREE_MAX_FOOD = 30
TREE_MIN_FOOD = 10
TREE_REGEN_RATE = 0.02  # only regens if not eaten to 0 (trees respawn when 0)

# Combat
ATTACK_RADIUS = 14
ATTACK_COOLDOWN_TICKS = 16
ATTACK_DAMAGE = 9.0
ATTACK_COST = 1.2
ATTACK_STEAL = 0.35  # fraction of damage converted to attacker energy
PUSH_FORCE = 0.9

# Terrain physics
WATER_DRAG = 0.55
MOUNTAIN_DRAG = 0.80
LAND_DRAG = 0.88

WATER_ENERGY_MULT = 1.30
MOUNTAIN_ENERGY_MULT = 1.45

# Camera
START_ZOOM = 1.0
MIN_ZOOM, MAX_ZOOM = 0.25, 3.5
CAMERA_SPEED_BASE = 12.0

# Rendering / perf
CHUNK_STEP = 60            # terrain draw grid size
GRID_CELL = 70             # spatial hash cell size (bigger => fewer buckets)
TREE_GRID_CELL = 120       # separate grid for trees

# Genetics
GENOME_LEN = 3             # used for color + "species"
ENEMY_THRESHOLD = 0.35     # genetic distance above which considered enemy

# Brain sizes (bigger mind)
NN_INPUTS = 12
NN_HIDDEN = 24
NN_OUTPUTS = 6  # move_x, move_y, eat, reproduce, attack, speed_bias


# ------------------------ Init ------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolving Creatures - Combat + Terrain + Chunking")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)


# ------------------------ Utility: deterministic noise ------------------------
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
        vx1 = v01 * (1 - tx) + v11 * tx
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
    n = (n - 0.5) * 1.4
    return n

def terrain_type(x, y):
    n = terrain_value(x, y)
    if n < -0.18:
        return "water"
    elif n > 0.25:
        return "mountain"
    else:
        return "land"

def terrain_color(x, y):
    t = terrain_type(x, y)
    if t == "water":
        v = terrain_value(x, y)
        depth = min(1.0, max(0.0, (-0.18 - v) * 3.5))
        base = (40, 95, 170)
        deep = (18, 55, 120)
        return (
            int(base[0] * (1 - depth) + deep[0] * depth),
            int(base[1] * (1 - depth) + deep[1] * depth),
            int(base[2] * (1 - depth) + deep[2] * depth),
        )
    elif t == "mountain":
        v = terrain_value(x, y)
        height = min(1.0, max(0.0, (v - 0.25) * 2.0))
        low = (120, 120, 120)
        high = (235, 235, 235)
        return (
            int(low[0] * (1 - height) + high[0] * height),
            int(low[1] * (1 - height) + high[1] * height),
            int(low[2] * (1 - height) + high[2] * height),
        )
    else:
        v = terrain_value(x, y)
        fert = min(1.0, max(0.0, (0.25 - abs(v)) * 2.0))
        dry = (110, 150, 90)
        lush = (75, 185, 100)
        return (
            int(dry[0] * (1 - fert) + lush[0] * fert),
            int(dry[1] * (1 - fert) + lush[1] * fert),
            int(dry[2] * (1 - fert) + lush[2] * fert),
        )

def terrain_effects(x, y):
    t = terrain_type(x, y)
    if t == "water":
        return WATER_DRAG, WATER_ENERGY_MULT
    if t == "mountain":
        return MOUNTAIN_DRAG, MOUNTAIN_ENERGY_MULT
    return LAND_DRAG, 1.0

def clamp(v, a, b):
    return a if v < a else (b if v > b else v)

def vec_norm(x, y):
    d = math.hypot(x, y)
    if d < 1e-9:
        return 0.0, 0.0, 0.0
    return x / d, y / d, d


# ------------------------ Chunking (spatial hash) ------------------------
def cell_id(x, y, cell):
    return (int(x) // cell, int(y) // cell)

def rebuild_agent_grid(agents):
    grid = {}
    for idx, a in enumerate(agents):
        c = cell_id(a.x, a.y, GRID_CELL)
        grid.setdefault(c, []).append(idx)
    return grid

def rebuild_tree_grid(trees):
    grid = {}
    for idx, t in enumerate(trees):
        c = cell_id(t.x, t.y, TREE_GRID_CELL)
        grid.setdefault(c, []).append(idx)
    return grid

def neighbor_cells(cx, cy):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (cx + dx, cy + dy)


# ------------------------ Neural Network ------------------------
class Brain:
    def __init__(self, i=NN_INPUTS, h=NN_HIDDEN, o=NN_OUTPUTS):
        self.i = i
        self.h = h
        self.o = o
        self.w1 = [[random.uniform(-1, 1) for _ in range(i)] for _ in range(h)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(h)] for _ in range(o)]

    def forward(self, inputs):
        hidden = [0.0] * self.h
        for hh in range(self.h):
            s = 0.0
            row = self.w1[hh]
            for ii in range(self.i):
                s += row[ii] * inputs[ii]
            hidden[hh] = math.tanh(s)

        outputs = [0.0] * self.o
        for oo in range(self.o):
            s = 0.0
            row = self.w2[oo]
            for hh in range(self.h):
                s += row[hh] * hidden[hh]
            outputs[oo] = math.tanh(s)
        return outputs

    def mutate(self, rate=0.08, mag=0.25):
        for hh in range(self.h):
            for ii in range(self.i):
                if random.random() < rate:
                    self.w1[hh][ii] += random.uniform(-mag, mag)
        for oo in range(self.o):
            for hh in range(self.h):
                if random.random() < rate:
                    self.w2[oo][hh] += random.uniform(-mag, mag)


# ------------------------ Entities ------------------------
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
        if self.food > 0 and self.food < TREE_MAX_FOOD:
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
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.energy = 55.0 + random.random() * 25.0
        self.age = 0
        self.cooldown = 0

        self.brain = brain if brain else Brain()

        # genome in [0,1]
        if genome is None:
            self.genome = [random.random() for _ in range(GENOME_LEN)]
        else:
            self.genome = genome

        # color derived directly from genome => bigger difference = bigger color difference
        self.color = (
            int(clamp(self.genome[0], 0, 1) * 255),
            int(clamp(self.genome[1], 0, 1) * 255),
            int(clamp(self.genome[2], 0, 1) * 255),
        )

    def genome_dist(self, other):
        d0 = self.genome[0] - other.genome[0]
        d1 = self.genome[1] - other.genome[1]
        d2 = self.genome[2] - other.genome[2]
        return math.sqrt(d0*d0 + d1*d1 + d2*d2)

    def is_enemy(self, other):
        return self.genome_dist(other) > ENEMY_THRESHOLD

    def sample_terrain_ahead(self):
        # look in direction of current velocity; if standing still, sample "up"
        dx, dy = self.vx, self.vy
        nx, ny, d = vec_norm(dx, dy)
        if d < 0.05:
            nx, ny = 0.0, -1.0
        look = 22.0
        tx = self.x + nx * look
        ty = self.y + ny * look
        t = terrain_type(tx, ty)
        # one-hot encoding
        is_water = 1.0 if t == "water" else 0.0
        is_land = 1.0 if t == "land" else 0.0
        is_mtn = 1.0 if t == "mountain" else 0.0
        return is_water, is_land, is_mtn

    def find_nearest_tree_chunked(self, trees, tree_grid):
        cx, cy = cell_id(self.x, self.y, TREE_GRID_CELL)
        best_i = None
        best_d2 = float("inf")

        for nc in neighbor_cells(cx, cy):
            for ti in tree_grid.get(nc, []):
                t = trees[ti]
                dx = t.x - self.x
                dy = t.y - self.y
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = ti

        if best_i is None:
            return None, float("inf"), 0.0, 0.0

        t = trees[best_i]
        nx, ny, d = vec_norm(t.x - self.x, t.y - self.y)
        return t, d, nx, ny

    def find_nearest_agent_chunked(self, agents, agent_grid):
        cx, cy = cell_id(self.x, self.y, GRID_CELL)
        best_i = None
        best_d2 = float("inf")

        for nc in neighbor_cells(cx, cy):
            for ai in agent_grid.get(nc, []):
                other = agents[ai]
                if other is self:
                    continue
                dx = other.x - self.x
                dy = other.y - self.y
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = ai

        if best_i is None:
            return None, float("inf"), 0.0, 0.0, 0.0

        other = agents[best_i]
        nx, ny, d = vec_norm(other.x - self.x, other.y - self.y)
        enemy = 1.0 if self.is_enemy(other) else 0.0
        return other, d, nx, ny, enemy

    def update(self, agents, trees, agent_grid, tree_grid):
        self.age += 1
        if self.cooldown > 0:
            self.cooldown -= 1

        # Terrain at current position
        drag, metab_mult = terrain_effects(self.x, self.y)

        # Nearest tree (chunked)
        tree, tree_dist, food_dx, food_dy = self.find_nearest_tree_chunked(trees, tree_grid)
        food_dist_norm = clamp(tree_dist / 500.0, 0.0, 1.0)

        # Nearest agent (chunked) as "vision" for enemy/friend
        other, other_dist, oth_dx, oth_dy, is_enemy = self.find_nearest_agent_chunked(agents, agent_grid)
        other_dist_norm = clamp(other_dist / 260.0, 0.0, 1.0)

        # Terrain ahead sensing
        ahead_water, ahead_land, ahead_mtn = self.sample_terrain_ahead()

        # Inputs (12)
        inputs = [
            clamp(self.energy / 140.0, 0.0, 1.0),     # 0 energy
            food_dist_norm,                            # 1
            food_dx,                                   # 2
            food_dy,                                   # 3
            other_dist_norm,                           # 4
            oth_dx,                                    # 5
            oth_dy,                                    # 6
            is_enemy,                                  # 7
            ahead_water,                               # 8
            ahead_land,                                # 9
            ahead_mtn,                                 # 10
            random.uniform(-1, 1),                     # 11 noise
        ]

        move_x, move_y, eat_sig, repro_sig, attack_sig, speed_bias = self.brain.forward(inputs)

        # Movement
        accel = 0.55 + 0.35 * max(0.0, speed_bias)  # 0.55..0.90
        self.vx += move_x * accel
        self.vy += move_y * accel

        self.x += self.vx
        self.y += self.vy

        self.vx *= drag
        self.vy *= drag

        self.x = clamp(self.x, 0.0, WORLD_SIZE)
        self.y = clamp(self.y, 0.0, WORLD_SIZE)

        speed = math.hypot(self.vx, self.vy)
        self.energy -= (BASE_METABOLISM * metab_mult) + (MOVE_COST * speed * 70.0)

        # Eat food
        if tree and eat_sig > 0.15 and tree.food > 0:
            if tree_dist < EAT_RADIUS:
                eat = min(EAT_RATE, tree.food)
                tree.food -= eat
                self.energy += eat * 6.0
                if tree.food <= 0.01:
                    # food is gone: respawn tree elsewhere
                    tree.food = 0.0
                    tree.respawn()

        # Attack / push
        if other and attack_sig > 0.35 and self.cooldown == 0 and self.energy > 5.0:
            if other_dist < ATTACK_RADIUS:
                # attacking costs energy regardless
                self.energy -= ATTACK_COST
                self.cooldown = ATTACK_COOLDOWN_TICKS

                # damage scales with "enemy-ness" so evolution can learn discrimination
                enemy_mult = 1.0 if self.is_enemy(other) else 0.4
                dmg = ATTACK_DAMAGE * enemy_mult

                other.energy -= dmg
                self.energy += dmg * ATTACK_STEAL

                # push / knockback
                nx, ny, d = vec_norm(other.x - self.x, other.y - self.y)
                if d > 0:
                    other.vx += nx * PUSH_FORCE
                    other.vy += ny * PUSH_FORCE

        # Reproduce
        if repro_sig > 0.55 and self.energy > REPRO_THRESHOLD and len(agents) < MAX_AGENTS:
            self.energy -= REPRO_COST

            child_brain = pickle.loads(pickle.dumps(self.brain))
            child_brain.mutate(rate=0.08, mag=0.28)

            # mutate genome => color changes strongly with genome distance
            child_genome = self.genome[:]
            for i in range(GENOME_LEN):
                if random.random() < 0.35:
                    child_genome[i] = clamp(child_genome[i] + random.uniform(-0.12, 0.12), 0.0, 1.0)

            agents.append(Agent(
                self.x + random.uniform(-6, 6),
                self.y + random.uniform(-6, 6),
                brain=child_brain,
                genome=child_genome
            ))

        # Death
        return self.energy > 0.0


# ------------------------ Save / Load ------------------------
def save_world(agents, trees, cam_x, cam_y, zoom):
    data = {
        "agents": [(a.x, a.y, a.vx, a.vy, a.energy, a.age, a.cooldown, a.genome, a.brain) for a in agents],
        "trees": [(t.x, t.y, t.food) for t in trees],
        "camera": (cam_x, cam_y, zoom),
        "meta": {"seed": SEED, "time": time.time()}
    }
    with open("world.save", "wb") as f:
        pickle.dump(data, f)
    print("Saved:", len(agents), "agents")

def load_world():
    try:
        with open("world.save", "rb") as f:
            data = pickle.load(f)

        agents = []
        for x, y, vx, vy, energy, age, cooldown, genome, brain in data.get("agents", []):
            a = Agent(x, y, brain=brain, genome=genome)
            a.vx, a.vy = vx, vy
            a.energy = energy
            a.age = age
            a.cooldown = cooldown
            agents.append(a)

        trees = []
        for x, y, food in data.get("trees", []):
            t = Tree()
            t.x, t.y, t.food = x, y, food
            trees.append(t)

        cam_x, cam_y, zoom = data.get("camera", (WORLD_SIZE/2, WORLD_SIZE/2, START_ZOOM))
        print("Loaded:", len(agents), "agents")
        return agents, trees, cam_x, cam_y, zoom
    except Exception as e:
        print("Load failed:", e)
        return None


# ------------------------ Setup ------------------------
trees = [Tree() for _ in range(TREE_COUNT)]
agents = [Agent(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE)) for _ in range(START_AGENTS)]

cam_x = WORLD_SIZE / 2
cam_y = WORLD_SIZE / 2
zoom = START_ZOOM


# ------------------------ Main Loop ------------------------
running = True
while running:
    clock.tick(FPS)

    # input
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_f:
                save_world(agents, trees, cam_x, cam_y, zoom)
            elif e.key == pygame.K_l:
                loaded = load_world()
                if loaded:
                    agents, trees, cam_x, cam_y, zoom = loaded
        elif e.type == pygame.MOUSEWHEEL:
            if e.y > 0:
                zoom *= 1.2
            else:
                zoom *= 0.8
            zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM)

    # camera
    keys = pygame.key.get_pressed()
    cam_speed = CAMERA_SPEED_BASE / zoom
    if keys[pygame.K_w]:
        cam_y -= cam_speed
    if keys[pygame.K_s]:
        cam_y += cam_speed
    if keys[pygame.K_a]:
        cam_x -= cam_speed
    if keys[pygame.K_d]:
        cam_x += cam_speed
    cam_x = clamp(cam_x, 0.0, WORLD_SIZE)
    cam_y = clamp(cam_y, 0.0, WORLD_SIZE)

    # update trees
    for t in trees:
        t.update()

    # rebuild chunk grids (fast hash)
    agent_grid = rebuild_agent_grid(agents)
    tree_grid = rebuild_tree_grid(trees)

    # update agents
    alive = []
    for a in agents:
        if a.update(agents, trees, agent_grid, tree_grid):
            alive.append(a)
    agents = alive

    # soft repopulation
    if len(agents) < 40:
        for _ in range(25):
            agents.append(Agent(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE)))

    # ------------------------ Render ------------------------
    screen.fill((10, 10, 16))

    # terrain draw only visible region
    step = CHUNK_STEP
    view_w = WIDTH / (2 * zoom)
    view_h = HEIGHT / (2 * zoom)

    start_x = int((cam_x - view_w) // step * step)
    end_x   = int((cam_x + view_w) // step * step + step)
    start_y = int((cam_y - view_h) // step * step)
    end_y   = int((cam_y + view_h) // step * step + step)

    for gx in range(start_x, end_x, step):
        for gy in range(start_y, end_y, step):
            if gx < 0 or gy < 0 or gx > WORLD_SIZE or gy > WORLD_SIZE:
                continue
            col = terrain_color(gx + step * 0.5, gy + step * 0.5)
            sx = (gx - cam_x) * zoom + WIDTH / 2
            sy = (gy - cam_y) * zoom + HEIGHT / 2
            pygame.draw.rect(screen, col, (sx, sy, step * zoom + 1, step * zoom + 1))

    # trees
    for t in trees:
        sx = (t.x - cam_x) * zoom + WIDTH / 2
        sy = (t.y - cam_y) * zoom + HEIGHT / 2
        if sx < -50 or sx > WIDTH + 50 or sy < -50 or sy > HEIGHT + 50:
            continue
        r = max(2, int((3 + 6 * (t.food / TREE_MAX_FOOD)) * zoom))
        base_col = (35, 95, 45)
        rich_col = (55, 165, 70)
        k = clamp(t.food / TREE_MAX_FOOD, 0.0, 1.0)
        col = (int(base_col[0]*(1-k) + rich_col[0]*k),
               int(base_col[1]*(1-k) + rich_col[1]*k),
               int(base_col[2]*(1-k) + rich_col[2]*k))
        pygame.draw.circle(screen, col, (int(sx), int(sy)), r)

    # agents
    for a in agents:
        sx = (a.x - cam_x) * zoom + WIDTH / 2
        sy = (a.y - cam_y) * zoom + HEIGHT / 2
        if sx < -30 or sx > WIDTH + 30 or sy < -30 or sy > HEIGHT + 30:
            continue
        r = max(2, int(3.2 * zoom))
        pygame.draw.circle(screen, (10, 10, 10), (int(sx), int(sy)), r + 1)
        pygame.draw.circle(screen, a.color, (int(sx), int(sy)), r)

    hud = f"Agents:{len(agents)} Trees:{len(trees)} Zoom:{zoom:.2f} Cam:({int(cam_x)},{int(cam_y)})"
    screen.blit(font.render(hud, True, (235, 235, 235)), (10, 8))
    screen.blit(font.render("WASD move | wheel zoom | F save | L load | ESC quit", True, (210, 210, 210)), (10, 28))
    screen.blit(font.render("Combat: agents can attack/push when NN output 'attack' is high", True, (180, 180, 180)), (10, 48))

    pygame.display.flip()

pygame.quit()