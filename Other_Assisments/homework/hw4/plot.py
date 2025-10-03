import os
import argparse

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

f, ax = plt.subplots(1, 1)

valid_exps = []
data_frames = []
for exp in (args.exps or []):
    log_fname = os.path.join('data', exp, 'log.csv')
    if not os.path.exists(log_fname):
        print(f"⚠️  Missing log file: {log_fname}. Skipping.")
        continue
    try:
        csv = pandas.read_csv(log_fname)
        data_frames.append(csv)
        valid_exps.append(exp)
    except Exception as e:
        print(f"⚠️  Could not read {log_fname}: {e}. Skipping.")

if not data_frames:
    print("⚠️  No valid log files found; nothing to plot.")
    # Save an empty placeholder plot if save path is provided
    if args.save:
        os.makedirs('plots', exist_ok=True)
        f.savefig(os.path.join('plots', args.save + '.jpg'))
    else:
        plt.close(f)
    raise SystemExit(0)

for i, (exp, csv) in enumerate(zip(valid_exps, data_frames)):
    color = cm.viridis(i / float(max(1, len(valid_exps))))
    ax.plot(csv['Itr'], csv['ReturnAvg'], color=color, label=exp)
    ax.fill_between(csv['Itr'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
                    color=color, alpha=0.2)

ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Return')

if args.save:
    os.makedirs('plots', exist_ok=True)
    f.savefig(os.path.join('plots', args.save + '.jpg'))
else:
    plt.show()