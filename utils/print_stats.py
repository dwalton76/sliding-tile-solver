#!/usr/bin/env python3

import sys

stats = {}
total_entries = 0

with open(sys.argv[1], 'r') as fh:

    for line in fh:
        (state, steps) = line.strip().split(':')

        steps_len = len(list(steps))

        if steps_len not in stats:
            stats[steps_len] = 0
        stats[steps_len] += 1
        total_entries += 1

for steps_len in sorted(stats.keys()):
    print("%2d : %d entries (%d%%)" % (steps_len, stats[steps_len], int((stats[steps_len] * 100)/ total_entries) ))

print("\n%d total entries" % total_entries)
