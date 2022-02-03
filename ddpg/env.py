from gym import Env
from gym.spaces import Box
from gym.utils import seeding
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import math
import pygame

class UnicycleEnv(Env):
    def __init__(self):
        self.env_size = 5
        self.padding = 30
        self.win_size = 100*self.env_size + 2*self.padding
        self.first_call = True
        self.img_count = 0
        self.L = 0.2
        self.episode_length = 1000
        self.point_goal = np.array([self.env_size,self.env_size,math.pi/4])
        obstacles = Polygon([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]])
        obstacles = obstacles.union(
                             Polygon([[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]]))
        obstacles = obstacles.union(
                             Polygon([[3, 1], [4, 1], [4, 2], [3, 2], [3, 1]]))
        obstacles = obstacles.union(
                             Polygon([[1, 3], [2, 3], [2, 4], [1, 4], [1, 3]]))
        self.boundary = Polygon([[0, 0], [self.env_size, 0], 
                   [self.env_size, self.env_size], [0, self.env_size], [0, 0]])
        self.boundary = self.boundary.difference(obstacles)
        self.goal = Polygon([[self.env_size, self.env_size-1], 
                             [self.env_size, self.env_size], 
                             [self.env_size-1, self.env_size], 
                             [self.env_size, self.env_size-1]])
        self.corners = np.array([[1,2],[2,1],[1,4],[2,3],[3,2],[4,1],[3,4],
                                 [4,3]])
        #self.corners_dist = np.array([5.06449510224598, 4.123105625617661,
        #                              3.6502815398728847,2.23606797749979])
        self.corners_dist = np.array([4.243, 3.162,
                                      2.828,  1.414])
        self.action_space = Box(low=np.array([-1,-1 ]), high=np.array([1, 1]))
        self.observation_space = Box(low=np.array([0,0,0]), 
                                     high=np.array([5,5,2*np.pi]))
        self.seed()
        self.reset()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def input_transform(self, action):   
        # unused
        theta = self.state[2]
        in0 = math.cos(theta)*action[0] + math.sin(theta)*action[1]
        in1 = - 1/self.L*math.sin(theta)*action[0] \
              + 1/self.L*math.cos(theta)*action[1]
        input = np.array([in0, in1])
        return input

    def shortest_path_length(self, x):
        # compute the shortest path connecting x to the goal through
        # corners of the obstacles
        point=x[:2]
        if (x>=4).any():
            dist1 = ((point[0] - 4.5)**2 + (point[1] - 5)**2)**0.5
            dist2 = ((point[0] - 5)**2 + (point[1] - 4.5)**2)**0.5
            return min(dist1,dist2)
        else:
            best_distance=10
            for i in range(len(self.corners)):
                dist = ((point[0] - self.corners[i,0])**2 
                        + (point[1] - self.corners[i,1])**2)**0.5
                if dist < best_distance and (point < self.corners[i,:]).all():
                    closest = i
                    best_distance = dist
            best_distance += self.corners_dist[math.floor(closest/2)]
            return best_distance

    def step(self, action):

        def unicycle_derivative(t, state):
            # defined here so it can access action without receiving 
            # it as input (solve_ivp only wants a function depending
            # on time and initial state)
            der1 = action[0]*math.cos(state[2])
            der2 = action[0]*math.sin(state[2])
            out = np.array([der1,der2,action[1]])
            return out

        action = action.numpy().flatten()  
        action[0] = (action[0] + 1)/2
        done = False
        x0 = Point(self.state)
        sol = solve_ivp(unicycle_derivative, [0,0.1], x0)
        self.state = sol.y[:,-1]
        self.state[2] = self.state[2] % 2*np.pi
        x=Point(self.state)
        dist = x.distance(Point(x0)) 

        R=0
        self.episode_length -= 1
        if self.episode_length <= 0:
            #R = 1000
            done = True
        if not self.boundary.covers(x):
            R = 1000
            done = True
        elif self.goal.covers(x):
            R = -10000
            done = True
        dist_cost = 2*np.linalg.norm(self.point_goal[:2] - self.state[:2])**2
        #dist_cost = 2*np.linalg.norm(self.point_goal - self.state)**2
        #dist_cost = 10*np.linalg.norm(self.point_goal - self.state[:2])
        input_cost = 1*action[0] + 10*abs(action[1])
        #theta_cost = 10*(abs(-(self.state[2] - math.pi/4)^2+math.pi^2/16))**0.5
        cost = dist_cost + input_cost + R 

        info={}

        return self.state, cost, done, dist, info

    def render(self, states=None):
        #pygame.time.delay(100)
        WIN = pygame.display.set_mode((self.win_size,self.win_size))
        pygame.display.set_caption('Unicycle')
        WIN.fill((255,255,255))
        black=(0,0,0)
        grey = (50,50,50)
        blue = (10,10,250)
        red = (250,10,10)
        pygame.draw.line(WIN, grey, (self.padding,self.padding),
                        (self.win_size-self.padding,self.padding), 1)
        pygame.draw.line(WIN, grey, (self.padding,self.padding),
                        (self.padding,self.win_size-self.padding), 1)
        pygame.draw.line(WIN, grey, (self.win_size-self.padding,self.padding),
            (self.win_size-self.padding,self.win_size-self.padding), 1)
        pygame.draw.line(WIN, grey, (self.padding,self.win_size-self.padding),
            (self.win_size-self.padding,self.win_size-self.padding), 1)
        pygame.draw.rect(WIN, blue, (self.padding+20*self.env_size,
                                     self.padding+20*self.env_size,
                                     20*self.env_size,20*self.env_size))
        pygame.draw.rect(WIN, blue, (self.padding+60*self.env_size,
                                     self.padding+20*self.env_size,
                                     20*self.env_size,20*self.env_size))
        pygame.draw.rect(WIN, blue, (self.padding+20*self.env_size,
                                     self.padding+60*self.env_size,
                                     20*self.env_size,20*self.env_size))
        pygame.draw.rect(WIN, blue, (self.padding+60*self.env_size,
                                     self.padding+60*self.env_size,
                                     20*self.env_size,20*self.env_size))
        pygame.draw.polygon(WIN, red, ((self.padding+80*self.env_size,
                self.padding),(self.win_size-self.padding,self.padding),
                (self.win_size-self.padding,self.padding+20*self.env_size)))
        pygame.draw.circle(WIN, black, (self.padding+round(100*self.state[0]),
                    self.padding+100*self.env_size-round(100*self.state[1])),5)
        if states is not None and len(states) >= 2:
            for i in range(1, len(states)):
                pygame.draw.line(WIN, black,
                    (self.padding+round(100*states[i-1,0]),
                    self.padding+100*self.env_size-round(100*states[i-1,1])),
                    (self.padding+round(100*states[i,0]),
                    self.padding+100*self.env_size-round(100*states[i,1])),2)
            pygame.image.save(WIN, 'plots/trajectories/traj' 
                            + str(self.img_count)+ '_' + str(i) +'.png')
            self.img_count +=1

        pygame.display.update()


    def reset(self):
        # sample random states until one is in the boundary
        notOK = True
        while notOK:
            x = np.array([self.np_random.uniform(low=np.array([0,0,0]), 
                                                high=np.array([5,5,2*np.pi]))])
            xp = Point(x.flatten())
            if not self.boundary.covers(xp) or self.goal.covers(xp):
                notOK = True
            else:
                notOK = False
        self.state = x.flatten()
        self.episode_length = 1000
        return self.state