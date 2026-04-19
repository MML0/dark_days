import pygame, random, math, json, os
from collections import defaultdict

CONFIG_FILE="config.json"

WIDTH,HEIGHT=1500,900
NUM_BOIDS= 1000
CELL_SIZE=35

NEIGHBOR_RADIUS=28
SEPARATION_RADIUS=10

pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
clock=pygame.time.Clock()
pygame.font.init()
font=pygame.font.Font(None,20)

# ---------------------------
# UI
# ---------------------------

class Slider:
    def __init__(self,x,y,w,minv,maxv,val,label):
        self.rect=pygame.Rect(x,y,w,6)
        self.min=minv
        self.max=maxv
        self.val=val
        self.label=label
        self.drag=False

    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            if self.knob().collidepoint(e.pos):
                self.drag=True

        if e.type==pygame.MOUSEBUTTONUP:
            self.drag=False

        if e.type==pygame.MOUSEMOTION and self.drag:
            px=max(self.rect.left,min(e.pos[0],self.rect.right))
            t=(px-self.rect.left)/self.rect.w
            self.val=self.min+t*(self.max-self.min)

    def knob(self):
        t=(self.val-self.min)/(self.max-self.min)
        x=self.rect.left+t*self.rect.w
        return pygame.Rect(x-8,self.rect.centery-8,16,16)

    def draw(self):
        pygame.draw.rect(screen,(100,100,100),self.rect)
        pygame.draw.circle(screen,(255,255,255),self.knob().center,8)

        txt=font.render(
            f"{self.label}: {self.val:.2f}",
            True,
            (255,255,255)
        )
        screen.blit(txt,(self.rect.x,self.rect.y-22))

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        cfg=json.load(f)
else:
    cfg={}

sep_slider=Slider(20,30,220,0,3,cfg.get("sep",0.64),"Separation")
ali_slider=Slider(20,80,220,0,2.5,cfg.get("align",1.7),"Alignment")
coh_slider=Slider(20,130,220,0,2.5,cfg.get("coh",0.7),"Cohesion")
spd_slider=Slider(20,180,220,0.2,8,cfg.get("speed",1.2),"Speed")
noise_slider=Slider(20,230,220,0,0.5,cfg.get("noise",0.18),"Noise")

sliders=[
sep_slider,
ali_slider,
coh_slider,
spd_slider,
noise_slider
]

save_btn=pygame.Rect(20,270,80,30)

# ---------------------------
# Boids
# ---------------------------

class Boid:
    __slots__=("x","y","vx","vy")
    def __init__(self):
        self.x=random.random()*WIDTH
        self.y=random.random()*HEIGHT
        a=random.random()*math.tau
        self.vx=math.cos(a)*2
        self.vy=math.sin(a)*2

boids=[Boid() for _ in range(NUM_BOIDS)]

particles=[]

def limit(vx,vy,maxs,mins,density=0):

    boost=min(density*0.03,1.5)

    target_min=mins+boost

    s=(vx*vx+vy*vy)**0.5

    if s<target_min and s>0.0001:
        return vx/s*target_min,vy/s*target_min

    if s<=0.0001:
        a=random.random()*math.tau
        return math.cos(a)*target_min,math.sin(a)*target_min

    if s>maxs:
        return vx/s*maxs,vy/s*maxs

    return vx,vy


def fish(b):

    a=math.atan2(b.vy,b.vx)

    hx=b.x+math.cos(a)*8
    hy=b.y+math.sin(a)*8

    t1x=b.x+math.cos(a+2.5)*4
    t1y=b.y+math.sin(a+2.5)*4

    t2x=b.x+math.cos(a-2.5)*4
    t2y=b.y+math.sin(a-2.5)*4

    pygame.draw.polygon(
        screen,
        (255,255,255),
        [(hx,hy),(t1x,t1y),(t2x,t2y)]
    )

# cheap persistent trail
fade=pygame.Surface((WIDTH,HEIGHT))
fade.fill((0,0,0))
fade.set_alpha(48)

# ---------------------------
# Main
# ---------------------------

running=True

while running:

    for e in pygame.event.get():

        if e.type==pygame.QUIT:
            running=False

        if e.type==pygame.MOUSEBUTTONDOWN:
            if save_btn.collidepoint(e.pos):

                with open(CONFIG_FILE,"w") as f:
                    json.dump({
                        "sep":sep_slider.val,
                        "align":ali_slider.val,
                        "coh":coh_slider.val,
                        "speed":spd_slider.val,
                        "noise":noise_slider.val
                    },f)

        for s in sliders:
            s.handle(e)

    SEP=sep_slider.val
    ALIGN=ali_slider.val
    COH=coh_slider.val
    MAX_SPEED=spd_slider.val
    NOISE=noise_slider.val

    # spatial chunks
    grid=defaultdict(list)

    for i,b in enumerate(boids):
        grid[
            (int(b.x//CELL_SIZE),
             int(b.y//CELL_SIZE))
        ].append(i)

    # update
    for i,b in enumerate(boids):

        cx=int(b.x//CELL_SIZE)
        cy=int(b.y//CELL_SIZE)

        sx=sy=ax=ay=cxv=cyv=0
        count=0

        for gx in (-1,0,1):
            for gy in (-1,0,1):

                for j in grid.get((cx+gx,cy+gy),[]):

                    if i==j:
                        continue

                    o=boids[j]

                    dx=o.x-b.x
                    dy=o.y-b.y

                    if dx>WIDTH/2: dx-=WIDTH
                    if dx<-WIDTH/2: dx+=WIDTH
                    if dy>HEIGHT/2: dy-=HEIGHT
                    if dy<-HEIGHT/2: dy+=HEIGHT

                    d2=dx*dx+dy*dy

                    if d2<NEIGHBOR_RADIUS**2:

                        count+=1

                        ax+=o.vx
                        ay+=o.vy

                        cxv+=dx
                        cyv+=dy

                        if d2<SEPARATION_RADIUS**2 and d2>0.01:

                            force=1/(d2+0.1)

                            sx-=dx*force*8
                            sy-=dy*force*8

                            # collision sparks
                            if d2<6:
                                if len(particles)<300:
                                    particles.append([
                                        b.x,b.y,
                                        random.uniform(-1,1),
                                        random.uniform(-1,1),
                                        14
                                    ])

        if count:

            ax=ax/count-b.vx
            ay=ay/count-b.vy

            cxv/=count
            cyv/=count

            b.vx+=(
                sx*SEP*0.02+
                ax*ALIGN*0.04+
                cxv*COH*0.002
            )

            b.vy+=(
                sy*SEP*0.02+
                ay*ALIGN*0.04+
                cyv*COH*0.002
            )

            # noise
            ang=math.atan2(b.vy,b.vx)

            local_noise=NOISE*(1+count*0.02)

            ang+=random.uniform(
                -local_noise,
                local_noise
            )

            speed=(b.vx*b.vx+b.vy*b.vy)**0.5

            b.vx=math.cos(ang)*speed
            b.vy=math.sin(ang)*speed

        MIN_SPEED=MAX_SPEED*0.55

        b.vx,b.vy=limit(
            b.vx,
            b.vy,
            MAX_SPEED,
            MIN_SPEED,
            count
        )

    # move
    for b in boids:
        b.x=(b.x+b.vx)%WIDTH
        b.y=(b.y+b.vy)%HEIGHT

    # draw with trails
    screen.blit(fade,(0,0))

    for b in boids:
        fish(b)

    # particles
    for p in particles[:]:

        p[0]+=p[2]
        p[1]+=p[3]

        p[4]-=1

        if p[4]<=0:
            particles.remove(p)
            continue

        pygame.draw.circle(
            screen,
            (125,125,255),
            (int(p[0]),int(p[1])),
            1
        )

    for s in sliders:
        s.draw()

    pygame.draw.rect(
        screen,
        (80,80,110),
        save_btn
    )

    screen.blit(
        font.render(
            "SAVE",
            True,
            (255,255,255)
        ),
        (35,278)
    )

    pygame.display.flip()
    clock.tick(120)

pygame.quit()