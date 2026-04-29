import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
from constants import FREIGHT, ACTIONS

class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)
        self.agent = None
        self.score = 0
        self.prev_score = 0
        self.prev_pellet_dist = 0
        self.prev_scared_count = 0
        self.prev_direction = LEFT
        # The last direction chosen by the agent, cached between node arrivals.
        self.pending_direction = LEFT
        # Counts frames since the agent was last queried for a new action.
        self.frames_since_action = 0

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

        state = self.getState()

        # Only ask the agent for a new direction at node boundaries OR every 15 frames.
        # Previously get_action was called every single frame, so with random exploration
        # Pac-Man would reverse direction up to 60 times per second. Now reversals can
        # still happen (mid-tile fleeing is valid), just much less frantically.
        self.frames_since_action += 1
        if self.overshotTarget() or self.frames_since_action >= 15:
            self.pending_direction = self.agent.get_action(state)
            self.frames_since_action = 0
        direction = self.pending_direction

        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

        # store transition and train
        next_state = self.getState()
        score_delta = self.score - self.prev_score
        reward = score_delta
        if done := not self.alive:
            reward -= 100
        else:
            reward += 0.1  # survival bonus
        nearest = min(self.pellets.pelletList, key=lambda p: abs(self.position.x - p.position.x) + abs(self.position.y - p.position.y)) if self.pellets.pelletList else None
        if nearest:
            curr_pellet_dist = abs(self.position.x - nearest.position.x) + abs(self.position.y - nearest.position.y)
            reward += (self.prev_pellet_dist - curr_pellet_dist) * 0.1
            self.prev_pellet_dist = curr_pellet_dist
        scared_count = sum(1 for g in self.ghosts if g.mode.current == FREIGHT)
        if scared_count > self.prev_scared_count:
            reward += 30  # power pellet bonus
        self.prev_scared_count = scared_count
        if score_delta >= 200:
            reward += 50  # ghost eating bonus
        if direction != self.prev_direction:
            reward -= 0.5  # direction change penalty
        self.prev_direction = direction
        action_idx = ACTIONS.index(direction)
        self.agent.buffer.push(state, action_idx, reward, next_state, done)
        self.agent.train()
        self.prev_score = self.score

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP  

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
    
    def setGameObjects(self, pellets, ghosts):
        self.pellets = pellets
        self.ghosts = ghosts

    def getState(self):
        # Builds the 30-number snapshot that the neural network reads every decision.
        # All values are normalised to roughly the 0–1 range so the network treats
        # every feature equally — without this, raw pixel coordinates (0–448) would
        # drown out the 0/1 boolean flags during training.
        state = []

        # Pac-Man's absolute position on the map, normalised by screen size.
        state.append(self.position.x / SCREENWIDTH)
        state.append(self.position.y / SCREENHEIGHT)

        # For each of the 4 ghosts: relative offset from Pac-Man, scared flag,
        # and ghost's current direction.
        # Relative (dx/dy) rather than absolute is intentional — what matters is
        # "ghost is 3 tiles to my left", not "ghost is at pixel 200".
        # Ghost direction is normalised by 4.0 (max index in ACTIONS list).
        for ghost in self.ghosts:
            dx = ghost.position.x - self.position.x
            dy = ghost.position.y - self.position.y
            scared = 1 if ghost.mode.current == FREIGHT else 0
            state.append(dx / SCREENWIDTH)
            state.append(dy / SCREENHEIGHT)
            state.append(scared)
            state.append((ACTIONS.index(ghost.direction) if ghost.direction in ACTIONS else 0) / 4.0)

        # Nearest pellet position — kept absolute (not relative) to avoid a full
        # retrain. A minor inconsistency with ghost coords but not worth the cost.
        nearest = min(self.pellets.pelletList, key=lambda p: abs(self.position.x - p.position.x) + abs(self.position.y - p.position.y))
        state.append(nearest.position.x / SCREENWIDTH)
        state.append(nearest.position.y / SCREENHEIGHT)

        # How many pellets are left, normalised by 300 (a safe upper bound).
        state.append(len(self.pellets.pelletList) / 300.0)

        # One-hot encoding of Pac-Man's current direction (5 values, one per action).
        for d in ACTIONS:
            state.append(1 if self.direction == d else 0)

        # Which directions Pac-Man can physically move from the current node.
        # Helps the network avoid trying to walk into walls.
        for d in [UP, DOWN, LEFT, RIGHT]:
            state.append(1 if self.node.neighbors[d] is not None else 0)

        return state
