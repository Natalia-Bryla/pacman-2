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
        direction = self.agent.get_action(state)

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
        state = []
        state.append(self.position.x)
        state.append(self.position.y)
        for ghost in self.ghosts:
            dx = ghost.position.x - self.position.x
            dy = ghost.position.y - self.position.y
            scared = 1 if ghost.mode.current == FREIGHT else 0
            state.append(dx)
            state.append(dy)
            state.append(scared)
            state.append(ACTIONS.index(ghost.direction) if ghost.direction in ACTIONS else 0)
        nearest = min(self.pellets.pelletList, key=lambda p: abs(self.position.x - p.position.x) + abs(self.position.y - p.position.y))
        state.append(nearest.position.x)
        state.append(nearest.position.y)
        state.append(len(self.pellets.pelletList))
        for d in ACTIONS:
            state.append(1 if self.direction == d else 0)
        for d in [UP, DOWN, LEFT, RIGHT]:
            state.append(1 if self.node.neighbors[d] is not None else 0)
        return state
