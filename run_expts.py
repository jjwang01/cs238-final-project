#!/usr/bin/evn python
import subprocess
import sys


def main():
    # combinations = [("qlearn", "qlearn"), ("qlearn", "qlearn_conv"), ("qlearn", "positional"), ("qlearn", "mobility"), ("qlearn", "random")]
    if len(sys.argv) != 2:
        print("Usage should be run_expts.py <num_runs>")
        sys.exit()

    try:
        num_runs = int(sys.argv[1])
    except ValueError:
        print("Please input an integer.")
        sys.exit()

    combinations = [("qlearn", "qlearn_conv"), ("qlearn", "positional"), ("qlearn", "mobility"), ("qlearn", "random")]
    for combo in combinations:
        print(f"____NEW COMBO:{combo}____")
        for _ in range(num_runs):
            a = subprocess.run(["python", "main.py", combo[0], combo[1], "10"])
            print(a)

if __name__ == '__main__':
    main()