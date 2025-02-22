import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from Environment import Mygame

GameEnv=Mygame()
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
CHECKPOINT_DIR='./train'
LOG_DIR='./logs'

callback=TrainAndLoggingCallback(check_freq=5,save_path=CHECKPOINT_DIR)

mymodel=DQN('CnnPolicy',GameEnv,tensorboard_log=LOG_DIR,verbose=1,buffer_size=40000,learning_starts=700)
mymodel.learn(total_timesteps=10,callback=callback)
