"""Utility functions for running an agent in an environment.
"""

__all__ = ['run_experiment']


def run_experiment(env, agent, episode_count=200, max_steps=200, watch=True):
    if watch:
        outdir = '/tmp/' + agent.name + '-results'
        env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)
    reward = 0
    done = False
    for _ in range(episode_count):
        ob = env.reset()
        for _ in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    if watch:
        # Dump result info to disk
        env.monitor.close()
    return agent
