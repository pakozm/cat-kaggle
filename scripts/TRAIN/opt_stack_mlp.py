from hyperopt import fmin, tpe, hp, STATUS_OK
import subprocess

def wrapper(lst):
    args = ['april-ann', 'scripts/TRAIN/train_stack_mlp_gbm.lua']
    args.extend(map(str,lst))
    print(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    proc_out, proc_err = proc.communicate()
    # <you might have to do some more elaborate parsing of foo's output here>
    line = proc_out.splitlines()[-1]
    score = float(line.split()[3])
    print(line)
    return score

space = [
    2**hp.randint('h1', 10),
    2**hp.randint('h2', 10),
    2**hp.randint('h3', 10),
    0.0,
    0.0,
    0.0,
    0.1,
    0.2,
    0.01,
    0.1,
    0.02,
]

best_n = fmin(wrapper, space, algo=tpe.suggest, max_evals=100)

print best_n
