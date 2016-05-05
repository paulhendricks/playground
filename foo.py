"""TO BE EDITED

Not Complete!
"""
import gym


def main():
    env = gym.make('CartPole-v0')
    env.reset()
    for i_episode in xrange(20):
        observation = env.reset()
        for t in xrange(100):
            env.render()
            print observation
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print "Episode finished after {} timesteps".format(t + 1)
                break


if __name__ == '__main__':
    main()
