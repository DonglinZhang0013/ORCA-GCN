import subprocess

commands = [
    ["python", "main.py", "--score_func", "conve", "--opn", "ccorr", "--gpu", "0", "--data", "FB15k-237"],
]

for command in commands:
    print(f"Start: {' '.join(command)}")
    subprocess.run(command)
    print(f"Done: {' '.join(command)}\n")