# Import the W&B Python Library and log into W&B
import wandb
import random

wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.x ** 3 + config.y + random.random() * 2.0
    return score

def main():
    wandb.init()
    wandb.define_metric('score', summary='min,max,mean,last')
    for i in range(10):
      score = objective(wandb.config)
      wandb.log({'score': score})
    wandb.finish()

# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'x': {'max': 0.1, 'min': 0.01},
        'y': {'values': [1, 3, 7]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-min-max-sweep'
    )

wandb.agent(sweep_id, function=main, count=10)