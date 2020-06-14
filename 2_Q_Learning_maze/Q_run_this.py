"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from Q_maze_env import Maze    #环境
from Q_RL_brain import QLearningTable   #大脑


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()   #初始化环境   observation是list格式，即为机器人的坐标
        print(observation)
        print(type(observation))
        while True:
            # fresh env
            env.render()       #渲染环境
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions))) #list(range(env.n_actions)) 将动作转换成相应的数目，且放进list里面。
    env.after(100, update)
    env.mainloop()