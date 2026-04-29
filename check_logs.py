import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_best_reward(log_dir):
    # Find the tfevents file
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                event_file = os.path.join(root, file)
                print(f"Reading {event_file}...")
                
                # Load the events
                ea = EventAccumulator(event_file)
                ea.Reload()
                
                # Check available tags
                tags = ea.Tags()['scalars']
                # print(f"Available tags: {tags}")
                
                # Look for reward tags
                target_tags = ['rollout/ep_rew_mean', 'train/value_loss', 'rollout/ep_len_mean']
                
                found_rewards = []
                for tag in tags:
                    if 'rew' in tag or 'reward' in tag:
                        found_rewards.append(tag)
                
                if not found_rewards:
                    # If ep_rew_mean is missing, we might have to estimate from value_loss or something
                    # but usually it's there if a Monitor was used.
                    # Since I didn't use Monitor, it might be missing.
                    print("No reward tags found in Tensorboard log.")
                    return None
                
                for tag in found_rewards:
                    events = ea.Scalars(tag)
                    best_event = max(events, key=lambda x: x.value)
                    print(f"Tag: {tag}")
                    print(f"  Best Value: {best_event.value:.2f} at Step: {best_event.step}")
                
                return True
    return False

if __name__ == "__main__":
    log_path = "./tensorboard_logs/PPO_5"
    if not get_best_reward(log_path):
        print("Could not find any logs.")
