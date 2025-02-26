import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from Environment import Mygame
from DQNTrain import TrainAndLoggingCallback

GameEnv=Mygame()
LOG_DIR='./logs'

mymodel=DQN('CnnPolicy',GameEnv,tensorboard_log=LOG_DIR,verbose=1,buffer_size=20000,learning_starts=1000)
callback = TrainAndLoggingCallback(check_freq=5000, save_path="./train2")
mymodel.learn(total_timesteps=70000, callback=callback)
mymodel.save("./train/dqn_finetuned")
 