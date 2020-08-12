import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
#------------------------------------------------------------------------------------------

env = gym.make('LunarLander-v2')
# vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=0)

#--------------------------------------------------------------------------------------
#-----------------------------------Training and Saving the Agent-----------------------
# Train the agent, define timesteps
model.learn(total_timesteps=10000)
# Save the agent
model.save("a2c_lunar")
del model  # delete trained model to demonstrate loading
#---------------------------------------------------------------------------------------

#------------------------------------------TESTING/EVALUATION-------------------------
#----------------------------------Load the Model and evaluate and run-----------------
model = A2C.load("a2c_lunar")
#----------------------------------
epi_history =[]
epi_col_ar = np.zeros((5, 1000,2))# max 1000 steps for 2 columns, state_x,state_y
# Start Episode Loop
for y in range(5):
  print('Current Episode Number:')
  print(y)
  obs = env.reset()
  state = obs # for state data storage
  state_bag = [state]
  sav= np.zeros([1000,1])
  #------------------Enjoy trained agent
  for i in range(1000):
      env.render()
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      state_bag.append(obs) #save observations in state bag 
      sav[i,:] = rewards    #save rewards in sav array
      if dones[0]:
        sam = np.sum(sav)
        epi_history.append(sam)
        stat = np.asarray(state_bag)#convert to array type
        state =  np.delete(stat,stat.shape[0]-1,0)# Remove terminal state
        #--Store the episode wise data--but in 2D slices, each slice representing each episode--------------
        xo = epi_col_ar[0:state.shape[0],:,:]
        with open('state_all_nonga','w') as outfile: #-----**--Name of the file in which all episodes data is save--****
          for slice_2d in xo:
            np.savetxt(outfile,slice_2d)
        np.savetxt('mypol_epi_hist_80',epi_history)
        print("Episode finished after {} timesteps".format(i+1))
        print("Accumulated reward {}".format(sum(sav)))
        break
  env.close()
