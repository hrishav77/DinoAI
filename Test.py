from Environment import Mygame
import cv2
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import time
GameEnv=Mygame()

# obs=GameEnv.get_observation()
# isover=GameEnv.get_gameover()
mymodel=DQN.load("./train/best_model_70000.zip")
for episode in range(5): 
    obs = GameEnv.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = mymodel.predict(obs)
        obs, reward, done,truncate, info = GameEnv.step(int(action))
        time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)