import argparse
import os
from stable_baselines3 import PPO
from jansen_world import JansenEnv

def find_best_eval(model_path="jansen_rl_active_50k", episodes=50, xml_path="jansen_assembly_red_articulated.xml"):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    env = JansenEnv(render_mode=None, xml_path=xml_path)
    model = PPO.load(model_path)
    
    rewards = []
    print(f"Evaluating {episodes} episodes...")
    for i in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        if (i+1) % 10 == 0:
            print(f"  Finished {i+1} episodes...")

    best_reward = max(rewards)
    avg_reward = sum(rewards) / len(rewards)
    
    print("\n" + "="*40)
    print(f"BEST EPISODE REWARD: {best_reward:.2f}")
    print(f"AVERAGE REWARD:      {avg_reward:.2f}")
    print("="*40)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO model over many episodes and report best reward.")
    parser.add_argument("--model", type=str, default="jansen_rl_active_50k")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--xml", type=str, default="jansen_assembly_red_articulated.xml")
    args = parser.parse_args()
    find_best_eval(model_path=args.model, episodes=args.episodes, xml_path=args.xml)
