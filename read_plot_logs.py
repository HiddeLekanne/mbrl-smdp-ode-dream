# import os
import argparse
import re
import matplotlib.pyplot as plt


def read_file(filename):
    with open("logs/" + filename, "r") as file:
        contents = file.read()
        return contents


# Extract the experiment details
def extract_experiment(data, dictionary):
    experiment = re.search(r'Experiment: (\d+), Model: (.+), Environment: (\w+), Seed: (\d+)', data)
    # Experiment: 25502, Model: latent - ode, Environment: HalfCheetah_Simulator, Seed: 1
    if experiment:
        dictionary["experiment_id"] = experiment.group(1)
        dictionary["model"] = experiment.group(2)
        dictionary["environment"] = experiment.group(3)
        dictionary["seed"] = experiment.group(4)


# Extract the hyperparameters
def extract_hyperparameters(data, dictionary):
    hyperparams = re.search(
        r'gamma: ([\d.]+), latent_dim: (\d+), lr: ([\d.]+), batch_size: (\d+), eps_decay: ([\d.]+), max steps: (\d+), latent_policy: (\w+), obs_normal: (\w+)',
        data)
    if hyperparams:
        dictionary["gamma"] = hyperparams.group(1)
        dictionary["latent_dim"] = hyperparams.group(2)
        dictionary["lr"] = hyperparams.group(3)
        dictionary["batch_size"] = hyperparams.group(4)
        dictionary["eps_decay"] = hyperparams.group(5)
        dictionary["max_steps"] = hyperparams.group(6)
        dictionary["latent_policy"] = hyperparams.group(7)
        dictionary["obs_normal"] = hyperparams.group(8)


# Extract the rewards for each episode
def extract_rewards(data, dictionary):
    rewards = re.findall(r'Episode (\d+) \| total env steps = (\d+) \| env steps = (\d+) \| reward = ([-\d.]+)', data)
    if rewards:
        dictionary["episodes"] = []
        for result in rewards:
            # print(result)
            dictionary["episodes"] = {"episode": result[0],
                                      "total steps": result[1],
                                      "episode steps": result[2],
                                      "reward": result[3]}
        # total_reward = sum(float(r) for r in rewards)
        # print(f"Total reward for {len(rewards)} episodes: {total_reward}")
    print(dictionary["episodes"])


def extract_model_results(data, dictionary):
    train_epochs = re.findall(
        r'Epoch \d+ \| training MSE = ([\d.]+) \| test MSE = ([\d.]+) \| training dt loss = ([\d.]+) \| test dt loss = ([\d.]+) \| time = ([\d.]+.\d+) s',
        data)
    if train_epochs:
        num_epochs = len(train_epochs)
        dictionary["epochs"] = []
        for epoch in train_epochs:
            dictionary["epochs"].append({"train_mse": epoch[0],
                                         "test_mse": epoch[1],
                                         "train_dt_loss": epoch[2],
                                         "test_dt_loss": epoch[3],
                                         "train_time": epoch[4]})


def extract_check_points(data, dictionary):
    check_points = re.findall(
        r'MBMF Epoch (\d+) \| total env steps = (\d+) \| avg reward over last epoch = ([-\d.]+) \| eval reward = ([-\d.]+) \| time = ([\d.]+.\d+) s',
        data)
    if check_points:
        dictionary["checkpoints"] = {}
        for result in check_points:
            print(result)
            dictionary["checkpoints"][result[1]] = {
                "average reward": result[2],
                "eval reward": result[3],
                "time" : result[4]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plotting results')
    parser.add_argument('--log_name', type=str, default='log_latent-ode_hopper_29306.log',
                        help='The path to the log to be plotted')
    parser.add_argument('--all_logs', type=bool, default=False,
                        help="Whether just combine and plot all logs for each environment.")
    # parser.add_argument('--trained_model_path', type=str, default='', help='the pre-trained environment model path')
    args = parser.parse_args()

    log = read_file(args.log_name)
    # print(log)
    data_dict = {}
    extract_experiment(log, data_dict)
    extract_hyperparameters(log, data_dict)
    extract_rewards(log, data_dict)
    extract_model_results(log, data_dict)
    extract_check_points(log, data_dict)

    print(data_dict)

    x = []
    y = []
    for key, checkpoint in data_dict["checkpoints"].items():
        x.append(int(key))
        y.append(float(checkpoint["average reward"]))

    plt.plot(x, y)
    plt.show()
