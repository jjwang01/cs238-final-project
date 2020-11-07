#!/usr/bin/evn python
import subprocess
import sys


def main():
    # combinations = [("qlearn", "qlearn"), ("qlearn", "qlearn_conv"), ("qlearn", "positional"), ("qlearn", "mobility"), ("qlearn", "random")]
    combinations = [("qlearn", "qlearn_conv"), ("qlearn", "positional"), ("qlearn", "mobility"), ("qlearn", "random")]
    for combo in combinations:
        print(f"____NEW COMBO:{combo}____")
        for _ in range(10):
            a = subprocess.run(["python", "main.py", combo[0], combo[1], "10"])
            print(a)

if __name__ == '__main__':
    main()