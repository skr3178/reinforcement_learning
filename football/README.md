The current folder of Gfootball research only compiles with python3.10 or else the the docker container (More on that later).
Compiling on the python 3.10 allows one to start the ganme, however, it does not possible to run it with checkpoints. Attempts to compile with py 3.6 failed due to some missing libraries and possible C++ based dependencies. The trained model originally uses baselines which is pretty outdated and relies on tensorflow 1.15. This is unsupported in python 3.10 (dependency coming from python version).
The original game has an engine designed specifically for it. 

PPO or other models were attempted to be updated with stable_baselines3 which is more recent and uses pytorch. 

It uses the default AI based model as adversary for playing (C++ scripted).

Training on PPO based stable_baselines3. The function for training is based on a special case of CNN called impala network. This was included in the baselines repo, however was ommited in the more recent  stable_baslines3 folder. 

Below is the script for testing the human/player only mode and for training on stable_baselines3 module. 
There is however, some challenges in running the ppo model for training and further usage as adversary or further training. 

To install create python env 3.10 on conda running

``conda create -y -n gfootball310 python=3.10
conda create -y -n gfootball310 python=3.10``

Then run the following commands:

``sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip``

downgrade setuptools to install gym

``pip3 install setuptools==65.5.0``
# pip install -r requirements.txt 

Then upgrade again to install gfootball
``python3 -m pip install --upgrade pip setuptools psutil wheel
git clone https://github.com/google-research/football.git
cd football
python3 -m pip install .``

Update the GCC 
``conda install -c conda-forge libstdcxx-ng``
``python3 -m gfootball.play_game --action_set=full``

``sudo apt-get install -y gcc-10 g++-10
sudo apt-get install -y libstdc++6
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX``

To run it against an AI adversary
```python3 -m gfootball.play_game --action_set=full```

To install stable_baselines3 and attempt to run against it

``pip3 install setuptools==65.5.0
pip install stable-baselines3==1.8.0
pip install wheel==0.38.4
pip install gym==0.21.0
python3 -m gfootball.examples.run_sb3_ppo --level=academy_empty_goal_close ``

To run test on different envs, use the following commands: 
``python3 test_env.py``

To save some images while training use:
``python3 -m gfootball.examples.run_sb3_ppo --dump_full_episodes=True --render=True``


1. Test Model Performance (No rendering, fast)
``python3 play_vs_sb3_model.py ./ppo_sb3_gfootball.zip test``


2. Watch AI vs AI (Both teams controlled by your model)
``python3 play_vs_sb3_model.py ./ppo_sb3_gfootball.zip ai-vs-ai``


3. Play Against the AI (You vs AI - default mode)
``python3 play_vs_sb3_model.py ./ppo_sb3_gfootball.zip human-vs-ai``


4. Play on Different Level:
``python3 play_vs_sb3_model.py ./ppo_sb3_gfootball.zip ai-vs-ai academy_empty_goal_close``


5. To simply test the model with rendering
``python3 load_sb3_model.py``


Some explanation about the extracted vs extracted stacked frameworks: 

## "extracted" (without stacking)
Representation: Uses the "extracted" representation (also called "super minimap" or SMM)
Observation format: 4-channel visual representation with dimensions (72, 96, 4)
Channels:
Channel 1: Left team player positions
Channel 2: Right team player positions
Channel 3: Ball position
Channel 4: Active player position
Temporal information: Only current frame (no history)

## "extracted_stacked" (with stacking)

Representation: Same "extracted" representation as above
Observation format: 16-channel visual representation with dimensions (72, 96, 16)
Channels: Same 4 channels as above, but stacked over 4 consecutive frames
Temporal information: Contains information from the last 4 frames concatenated along the channel dimension

## Temporal context

Why use stacking?
Frame stacking is crucial for reinforcement learning in dynamic environments because:
Motion detection: The agent can see how players and the ball are moving
Velocity information: Direction and speed of movement
Temporal patterns: Sequences of actions and their consequences
Better decision making: More context for making informed decisions


To train and AI model:

```bash ./run_sb3_ppo_script.sh```

To run or play against and AI model run the commands

```python3 -m gfootball.play_game --players "keyboard:left_players=1;ppo_sb3:right_players=1,checkpoint=/home/skr3178/Downloads/football/gfootball/examples/ppo_sb3_gfootball_impala_params.zip" --level=11_vs_11_easy_stochastic```

