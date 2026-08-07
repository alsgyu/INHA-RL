import isaacgym

from utils.fast_sac_runner import FastSACK1Runner


if __name__ == "__main__":
    runner = FastSACK1Runner(test=False)
    runner.train()
