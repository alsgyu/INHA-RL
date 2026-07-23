import isaacgym

from utils.sirl_worldmodel_runner import SIRLWorldModelRunner


if __name__ == "__main__":
    runner = SIRLWorldModelRunner(test=False)
    runner.train()
