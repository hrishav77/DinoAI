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
        self.observation_space=Box(low=0,high=255, shape=(1,85,120),dtype=np.uint8)
        self.action_space=Discrete(3)
        self.cap=mss()
        self.game_location={'top':150,'left':90,'width':350,'height':250}
        self.game_over_loc={'top':230,'left':330,'width':300,'height':40}


    def step(self,action):
        ##my key map 0:space 1:Down 2:no action
        action_map={
            0:'space',
            1:'down',
            2:'none'
        }
        if action!=2:
            pydirectinput.press(action_map[action])
        isover,game_over_ss=self.get_gameover()
        new_obs=self.get_observation()
        reward=1
        data={}
        truncated = False
        return new_obs,reward,isover,truncated,data
         
    def reset(self, seed=None, options=None):
        """ Reset environment with Gymnasium's expected signature. """
        super().reset(seed=seed)  # Call parent reset to handle seeding
        time.sleep(1)
        pydirectinput.click(x=200, y=200)
        pydirectinput.press('space')
        return self.get_observation()
    def render(self):
        cv2.imshow('Game',np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF==ord('q'):
            self.close
    def get_observation(self):
        ss_pic=self.cap.grab(self.game_location)
        img_array=np.array(ss_pic)[:,:,:3]
        gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
        target_width = 120
        aspect_ratio = gray.shape[1] / gray.shape[0]
        target_height = int(target_width / aspect_ratio)
        resized = cv2.resize(gray, (target_width, target_height))
        channel=np.reshape(resized,(1,target_height,target_width))
        return channel
    
    def get_gameover(self):
        game_over_ss=np.array(self.cap.grab(self.game_over_loc))[:,:,:3]
        game_over_str=["GAME","GAHE"]
        over=False
        resultString=pytesseract.image_to_string(game_over_ss)[:4]
        if resultString in game_over_str:
            over=True
        # plt.ion() 
        # plt.imshow(game_over_ss)
        # plt.show()
        return over,game_over_ss

    def close(self):
        cv2.destroyAllWindows()

GameEnv=Mygame()

# GameEnv.reset()
# plt.ion()
# plt.imshow(cv2.cvtColor(GameEnv.get_observation()[0],cv2.COLOR_BGR2RGB))
# plt.show()
# input("Press Enter to exit...") 

    