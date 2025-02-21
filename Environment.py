from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class Mygame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space=Box(low=0,high=255, shape=(1,83,100),dtype=np.uint8)
        self.action_space=Discrete(3)
        self.cap=mss()

    def step(self,action):
        ##my key map 0:space 1:Down 2:no action
        pass
    def reset(self):
        pass
    def render(self):
        pass
    def get_observation(self):
        pass
    def get_done(self):
        pass
    def close(self):
        pass

GameEnv=Mygame()
print(GameEnv.action_space.sample())


    