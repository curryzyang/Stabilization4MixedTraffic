"""Run the noise experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic.experiment_runners import main

if __name__ == "__main__":
    main("noise_experiment")
