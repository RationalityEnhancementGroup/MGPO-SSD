#!/usr/bin/env python3
import pandas as pd
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class Labeler(object):
    """Assigns unique integer labels."""
    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label


names = ['participants', 'mouselab-mdp', 'survey-text']
pid_label = Labeler()

def main(v1, v2):
    path = 'data/combined/' + '_'.join([v1, v2])
    os.makedirs(path, exist_ok=True)

    data = []
    for v in [v1, v2]:
        ids = pd.read_csv(f'data/human_raw/{v}/identifiers.csv').set_index('pid')
        d = {}
        data.append(d)
        for name in names:
            df = pd.read_csv(f'data/human/{v}/{name}.csv').set_index('pid')
            df['worker_id'] = ids.worker_id
            df.reset_index(inplace=True, drop=True)
            df['pid'] = df.worker_id.apply(pid_label)
            df.set_index('pid', inplace=True)
            # df.drop('worker_id', inplace=True, axis=1)
            d[name] = df

    pdf = data[0]['participants']
    pdf.bonus = (data[1]['participants'].bonus + 0.55).round(2)
    pdf.bonus.fillna(0, inplace=True)

    def get_time(stage):
        ms = data[stage-1]['survey-text'].groupby('pid').time_elapsed.max()
        return (ms / 60000).fillna(0).round(2)
    pdf['stage1_time'] = get_time(1)
    pdf['stage2_time'] = get_time(2)
    pdf['total_time'] = (pdf.stage1_time + pdf.stage2_time).round(2)

    frames = {
        'participants': pdf,
        'stage1': data[0]['mouselab-mdp'],
        'stage2': data[1]['mouselab-mdp'],
    }

    for name, df in frames.items():
        fp = f'{path}/{name}.csv'
        df.to_csv(fp)
        print('Wrote', fp)

if __name__ == '__main__':
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'stage1',
        help="Experiment version for stage 1.")
    parser.add_argument(
        'stage2',
        help="Experiment version for stage 1.")
    
    args = parser.parse_args()
    main(args.stage1, args.stage2)