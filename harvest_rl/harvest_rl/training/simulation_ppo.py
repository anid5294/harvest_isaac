"""
2D drone simulation (physics + pygame). No RL code.

Controls:
  UP / DOWN     more / less total thrust
  LEFT / RIGHT  tilt left / right (thrust difference)
  G             toggle gusts
  R             reset (new start and goal)
  ESC           quit
"""

import math
import torch
import pygame
import random
import numpy as np
from torch import nn
from policy import PolicyNetwork, CriticNetwork
# ---------------------------------------------------------------- settings
WORLD_W, WORLD_H = 10.0, 6.0      # world size in metres
SCALE = 100                       # pixels per metre
HUD_H = 40                        # height of the info bar in pixels
FPS = 50                          # screen updates per second

PHYS_DT = 0.005                   # seconds per physics update
SUBSTEPS = 4                      # physics updates per frame (4 x 0.005 = 0.02 s)

GRAVITY = 9.81


# ---------------------------------------------------------------- drone physics
class Drone:
    MASS = 0.5            # kg
    ARM = 0.15            # m, centre to rotor
    INERTIA = 0.005       # kg*m^2
    DRAG = 0.1            # N per m/s
    MAX_TILT = math.radians(80)

    def __init__(self, x, z):
        self.x, self.z = x, z          # position (z = height, up is positive)
        self.vx, self.vz = 0.0, 0.0    # velocity
        self.theta = 0.0               # tilt, positive = tilted right
        self.omega = 0.0               # angular velocity
        self.hover = self.MASS * GRAVITY / 2      # thrust per rotor to hover
        self.max_thrust = 2 * self.hover
        self.t_left = self.t_right = self.hover   # current rotor thrusts

    def set_thrusts(self, left, right):
        self.t_left = min(max(left, 0.0), self.max_thrust)
        self.t_right = min(max(right, 0.0), self.max_thrust)

    def update(self, dt, ext_fx=0.0, ext_fz=0.0):
        total = self.t_left + self.t_right

        # Forces: thrust along body "up", gravity, drag, external (gusts)
        fx = total * math.sin(self.theta) - self.DRAG * self.vx + ext_fx
        fz = total * math.cos(self.theta) - self.MASS * GRAVITY - self.DRAG * self.vz + ext_fz

        # Torque: stronger left rotor tilts the drone right
        torque = (self.t_left - self.t_right) * self.ARM

        ax, az = fx / self.MASS, fz / self.MASS
        alpha = torque / self.INERTIA

        # Semi-implicit Euler: velocities first, then positions
        self.vx += ax * dt
        self.vz += az * dt
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.omega += alpha * dt
        self.theta += self.omega * dt

    def speed(self):
        return math.hypot(self.vx, self.vz)

    def crashed(self):
        return (self.x < 0 or self.x > WORLD_W or
                self.z < 0 or self.z > WORLD_H or
                abs(self.theta) > self.MAX_TILT)


# ---------------------------------------------------------------- disturbances
class GustGenerator:
    RATE = 0.3                # average gusts per second
    MIN_F, MAX_F = 0.5, 2.0   # N (drone weighs about 4.9 N)
    MIN_T, MAX_T = 0.3, 1.0   # seconds

    def __init__(self):
        self.enabled = False
        self.fx = self.fz = 0.0
        self.time_left = 0.0

    def update(self, dt):
        if not self.enabled:
            self.fx = self.fz = 0.0
            return
        if self.time_left > 0:
            self.time_left -= dt
            if self.time_left <= 0:
                self.fx = self.fz = 0.0
        elif random.random() < self.RATE * dt:
            angle = random.uniform(0, 2 * math.pi)
            strength = random.uniform(self.MIN_F, self.MAX_F)
            self.fx = strength * math.cos(angle)
            self.fz = strength * math.sin(angle)
            self.time_left = random.uniform(self.MIN_T, self.MAX_T)


# ---------------------------------------------------------------- world
class World:
    GOAL_RADIUS = 0.2
    GOAL_SPEED = 0.3

    def __init__(self):
        self.gusts = GustGenerator()
        self.reset()

    def reset(self):
        m = 1.0
        sx, sz = random.uniform(m, WORLD_W - m), random.uniform(m, WORLD_H - m)
        while True:
            gx, gz = random.uniform(m, WORLD_W - m), random.uniform(m, WORLD_H - m)
            # gx, gz = random.uniform(m, WORLD_W - m), (random.uniform(WORLD_H / 2, WORLD_H - m) if random.random() < 0.8 else random.uniform(m, WORLD_H / 2))
            if math.hypot(gx - sx, gz - sz) > 2.0:
                break
        self.drone = Drone(sx, sz)
        self.goal = (gx, gz)
        self.time = 0.0

    def step(self, left, right):
        """Advance one frame: set thrusts, run several physics substeps."""
        self.drone.set_thrusts(left, right)
        for _ in range(SUBSTEPS):
            self.gusts.update(PHYS_DT)
            self.drone.update(PHYS_DT, self.gusts.fx, self.gusts.fz)
        self.time += PHYS_DT * SUBSTEPS

    def dist_to_goal(self):
        return math.hypot(self.goal[0] - self.drone.x, self.goal[1] - self.drone.z)

    def at_goal(self):
        # return self.dist_to_goal() < self.GOAL_RADIUS and self.drone.speed() < self.GOAL_SPEED
        return self.dist_to_goal() < self.GOAL_RADIUS


# ---------------------------------------------------------------- drawing
class Renderer:
    DRAW_ARM = 0.35       # visual arm length (bigger than real, for visibility)

    def __init__(self):
        self.w, self.h = int(WORLD_W * SCALE), int(WORLD_H * SCALE)
        self.screen = pygame.display.set_mode((self.w, self.h + HUD_H))
        pygame.display.set_caption("2D drone sim")
        self.font = pygame.font.SysFont(None, 22)

    def to_screen(self, x, z):
        # world metres (z up) -> screen pixels (y down)
        return int(x * SCALE), int(self.h - z * SCALE)

    def arrow(self, color, x0, z0, x1, z1, width=3):
        s, e = self.to_screen(x0, z0), self.to_screen(x1, z1)
        pygame.draw.line(self.screen, color, s, e, width)
        dx, dy = e[0] - s[0], e[1] - s[1]
        length = math.hypot(dx, dy)
        if length < 4:
            return
        ux, uy = dx / length, dy / length
        head = [e,
                (e[0] - 10 * ux + 5 * uy, e[1] - 10 * uy - 5 * ux),
                (e[0] - 10 * ux - 5 * uy, e[1] - 10 * uy + 5 * ux)]
        pygame.draw.polygon(self.screen, color, head)

    def draw(self, world, message=""):
        scr, d = self.screen, world.drone
        scr.fill((245, 244, 240))

        # Goal: ring normally, filled when the drone is holding inside it
        g = self.to_screen(*world.goal)
        r = int(world.GOAL_RADIUS * SCALE)
        pygame.draw.circle(scr, (29, 158, 117), g, r, 0 if world.at_goal() else 2)

        # Body axes in world frame
        c, s = math.cos(d.theta), math.sin(d.theta)
        right_x, right_z = c, -s          # along the arm, toward right rotor
        up_x, up_z = s, c                 # thrust direction

        lx, lz = d.x - self.DRAW_ARM * right_x, d.z - self.DRAW_ARM * right_z
        rx, rz = d.x + self.DRAW_ARM * right_x, d.z + self.DRAW_ARM * right_z

        # Arm, body, rotors
        pygame.draw.line(scr, (60, 60, 60), self.to_screen(lx, lz), self.to_screen(rx, rz), 6)
        pygame.draw.circle(scr, (60, 60, 60), self.to_screen(d.x, d.z), 8)
        pygame.draw.circle(scr, (100, 100, 100), self.to_screen(lx, lz), 6)
        pygame.draw.circle(scr, (100, 100, 100), self.to_screen(rx, rz), 6)

        # Thrust arrows, length proportional to thrust
        for (px, pz), t in (((lx, lz), d.t_left), ((rx, rz), d.t_right)):
            k = 0.6 * t / d.max_thrust
            self.arrow((29, 158, 117), px, pz, px + k * up_x, pz + k * up_z)

        # Gust arrow
        gu = world.gusts
        if gu.fx or gu.fz:
            self.arrow((216, 90, 48), d.x, d.z, d.x + 0.4 * gu.fx, d.z + 0.4 * gu.fz, 4)

        # Info bar
        pygame.draw.rect(scr, (230, 228, 222), (0, self.h, self.w, HUD_H))
        info = (f"t {world.time:5.1f}s   dist {world.dist_to_goal():.2f} m   "
                f"speed {d.speed():.2f} m/s   tilt {math.degrees(d.theta):4.0f} deg   "
                f"gusts {'ON' if gu.enabled else 'off'} {message}")
        scr.blit(self.font.render(info, True, (40, 40, 40)), (10, self.h + 12))

        pygame.display.flip()


def get_obs(world):
    d = world.drone
    gx, gz = world.goal
    return np.array([
        (gx - d.x) / WORLD_W,     # goal direction, horizontal
        (gz - d.z) / WORLD_H,     # goal direction, vertical
        d.vx / 5.0,               # horizontal velocity
        d.vz / 5.0,               # vertical velocity
        math.sin(d.theta),        # tilt
        math.cos(d.theta),        # tilt
        d.omega / 10.0,           # angular velocity
        d.x / WORLD_W,            # position between left (0) and right (1) wall
        d.z / WORLD_H,            # position between floor (0) and ceiling (1)
    ], dtype=np.float32)


def get_reward(world):
    return torch.linalg.norm(torch.tensor(world.goal) - torch.tensor([world.drone.x, world.drone.z]))

def build_returns(episode_rewards, discount_factor=.99):

    G_t = []
    for i in range(len(episode_rewards)):
        j = i
        curr_return = 0
        for _ in range(j, len(episode_rewards)):
            curr_return += discount_factor**(_ - j) * episode_rewards[_]
        G_t.append(curr_return)

    G_t = torch.tensor(G_t, dtype=torch.float32)
    return G_t




# ---------------------------------------------------------------- main loop
def main():
    pygame.init()
    clock = pygame.time.Clock()
    world = World()
    view = Renderer()
    message = ""
    policy_network = PolicyNetwork(world)
    # policy_network.load_state_dict(torch.load('ppo_policy_network_1790978193.46174.pt', weights_only=True))
    
    critic_network = CriticNetwork(world)
    critic_loss_fn = nn.MSELoss()

    steps = 0
    # resetting for new episoide
    episode_rewards = []
    episode_policies = []
    episode_critic_value_predictions = []
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=3e-5, weight_decay=1e-7)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=3e-5, weight_decay=1e-7)
    running = True
    prev_reward = None
    episode_closest_dist_achieved = []
    closest = torch.inf
    while running:
        
        steps += 1
        if steps == 2000:
            
            # backpropinf whatveer it has learnt
            episode_rewards = torch.from_numpy(np.array(episode_rewards))
            episode_policies = torch.stack(episode_policies)
            G_t = build_returns(episode_rewards)
            
            # calculating the advantage
            G_t = torch.from_numpy(np.array(G_t))
            episode_critic_value_predictions_tensor = torch.stack(episode_critic_value_predictions).squeeze(1)
            episode_advantage = G_t - episode_critic_value_predictions_tensor
            
            # critic backprop
            critic_loss = critic_loss_fn(episode_critic_value_predictions_tensor, G_t)
            critic_loss.backward()
            critic_optimizer.step()
            
            # evaluate the objective function
            x = episode_advantage.detach() @ episode_policies
            objective = -1 * 1/ (len(episode_policies)) * x
            objective.backward()
            optimizer.step()
            message = f"Timeout! - reset Loss: {objective.item()} Episode Length: {len(episode_rewards)}  Critic Loss: {critic_loss.item()}"
            
            # resetting for new episoide
            episode_rewards = []
            episode_policies = []
            episode_critic_value_predictions = []
            prev_reward = None
            steps = 0
            episode_closest_dist_achieved.append(closest)
            closest = torch.inf
            world.reset()
            continue
            
        optimizer.zero_grad()
        critic_optimizer.zero_grad()
        # 1. Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    world.gusts.enabled = not world.gusts.enabled

        # 2. Keyboard -> rotor thrusts (0 input = hover thrust)
        hover = world.drone.hover
        gaussian_params = policy_network(get_obs(world))
        mus, sigmas = gaussian_params[:2], gaussian_params[2:]
    
        # sample an action from this gaussian distribution
        current_policy_distribution = torch.distributions.Normal(loc=mus, scale=sigmas)
        values = current_policy_distribution.sample()
        
        # critic stuff
        critic_state_value_pred = critic_network(get_obs(world))
        episode_critic_value_predictions.append(critic_state_value_pred)
        
        world.step(hover *(1 + float(values[0].numpy())), hover * (1 +float(values[1].numpy())))
        
        # just a var to track what was the closest state achieved duing that particular episode
        closest = world.dist_to_goal() if world.dist_to_goal() < closest else closest
        
        
        
        if world.at_goal():
            # perform that action and then calculate the reward
            reward = torch.tensor(0.0, dtype=torch.float32)
            
            # append current reward after at_goal state
            episode_rewards.append(torch.tensor(20, dtype=torch.float32) * (prev_reward - reward) - .005)
            episode_policies.append(current_policy_distribution.log_prob(values).sum())
            episode_rewards = torch.from_numpy(np.array(episode_rewards))
            
            # last reward gets +20 since it reached the goal
            episode_rewards[-1] += 20.0
            episode_policies = torch.stack(episode_policies)
            G_t = build_returns(episode_rewards)
            
            # calculating the advantage
            G_t = torch.from_numpy(np.array(G_t))
            episode_critic_value_predictions_tensor = torch.stack(episode_critic_value_predictions).squeeze(1)
            episode_advantage = G_t - episode_critic_value_predictions_tensor

            # critic backprop
            critic_loss = critic_loss_fn(episode_critic_value_predictions_tensor, G_t)
            critic_loss.backward()
            critic_optimizer.step()
            
            # evaluate the objective function
            x = episode_advantage.detach() @ episode_policies
            objective = -1 * 1/ (len(episode_policies)) * x
            objective.backward()
            # use the gradient of that action probability (weighted by the reward) to tune policy netowrk
            
            optimizer.step()
            message = f"Success! - reset Loss: {objective.item()} Episode Length: {len(episode_rewards)}  Critic Loss: {critic_loss.item()}"
            # # resetting for new episoide
            # episode_rewards = []
            # episode_policies = []
            # episode_critic_value_predictions = []
            # prev_reward = None
            # steps = 0
            # episode_closest_dist_achieved.append(closest)
            # closest = torch.inf
            # world.reset()
            # continue
            
        else:
            reward = get_reward(world)
        
        if prev_reward == None:
            episode_rewards.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            episode_rewards.append(torch.tensor(20, dtype=torch.float32) * (prev_reward - reward) - .005)
        episode_policies.append(current_policy_distribution.log_prob(values).sum())
        
        
        prev_reward = reward
        
        # 3. Crash check
        if world.drone.crashed():
            
            episode_rewards = torch.from_numpy(np.array(episode_rewards))
            episode_rewards[-1] -= 20.0
            episode_policies = torch.stack(episode_policies)
            G_t = build_returns(episode_rewards)
            
            # calculating the advantage
            G_t = torch.from_numpy(np.array(G_t))
            episode_critic_value_predictions_tensor = torch.stack(episode_critic_value_predictions).squeeze(1)
            episode_advantage = G_t - episode_critic_value_predictions_tensor
            
            # critic backprop
            critic_loss = critic_loss_fn(episode_critic_value_predictions_tensor, G_t)
            critic_loss.backward()
            critic_optimizer.step()
            
            # evaluate the objective function
            x = episode_advantage.detach() @ episode_policies
            objective = -1 * 1/ (len(episode_policies)) * x
            objective.backward()
            # use the gradient of that action probability (weighted by the reward) to tune policy netowrk
            
            optimizer.step()
            
            message = f"Crashed - reset Loss: {objective.item()} Episode Length: {len(episode_rewards)}  Critic Loss: {critic_loss.item()}"
            # resetting for new episoide
            episode_rewards = []
            episode_policies = []
            episode_critic_value_predictions = []
            prev_reward = None
            steps = 0
            episode_closest_dist_achieved.append(closest)
            closest = torch.inf
            world.reset()

        # 4. Draw
        view.draw(world, message)
        clock.tick(FPS)
        
        
    import time
    torch.save(policy_network.state_dict(), f"ppo_policy_network_{time.time()}.pt")
    pygame.quit()


if __name__ == "__main__":
    main()