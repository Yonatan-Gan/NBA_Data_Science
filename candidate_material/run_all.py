"""Run both alternative analyses without touching the original project files."""

from q1_analysis import main as run_q1
from q2_analysis import main as run_q2


if __name__ == "__main__":
    print("\n=== Q1 candidate analysis ===")
    run_q1()
    print("\n=== Q2 candidate analysis ===")
    run_q2()

