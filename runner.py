from src.ga import GA
from sessions.session import Session, MultiSession

# Example of MultiSession
config_one = 'config_files/config_one'
config_fast = 'config_files/config_faster'
ss = Session(GA(env_key='MiniGrid-Empty-Noise-8x8-v0', population=100,
           max_generations=20,
           max_episode_eval=100,
           sigma=0.005,
           truncation=7,
           elite_trials=5,
           n_elites=1), "SingleSession")
ss.start()
#ms = MultiSession([GA(env_key='MiniGrid-Empty-Noise-8x8-v0', population=100,
#           max_generations=20,
#           max_episode_eval=100,
#           sigma=0.005,
#           truncation=7,
#           elite_trials=5,
#           n_elites=1)], "TestSessions")
#ms.start()