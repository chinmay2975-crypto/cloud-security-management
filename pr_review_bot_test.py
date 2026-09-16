import os
import subprocess


def run_command(cmd):
    subprocess.run(cmd, shell=True)
