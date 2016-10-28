"""Run the agent in the environment

Complete!
"""

__all__ = ['run']


def run(env, agent):
    outdir = '/tmp/' + agent.name + '-results'
    env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)

    episode_count = 200
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    # Dump result info to disk
    env.monitor.close()
