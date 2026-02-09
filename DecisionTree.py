import csv, math
from collections import Counter

# entropy
def entropy(data):
    total = len(data)
    cnt = Counter(r[-1] for r in data)
    return -sum((c/total)*math.log2(c/total) for c in cnt.values())

# best attribute
def best_attr(data, cols):
    base = entropy(data)
    gains = []
    for c in cols:
        splits = {}
        for r in data:
            splits.setdefault(r[c], []).append(r)
        gain = base - sum((len(s)/len(data))*entropy(s) for s in splits.values())
        gains.append((gain, c))
    return max(gains)[1]

# build tree
def build(data, cols, head):
    labels = [r[-1] for r in data]

    if len(set(labels)) == 1:
        return labels[0]
    if not cols:
        return Counter(labels).most_common(1)[0][0]

    b = best_attr(data, cols)
    tree = {head[b]: {}}

    for v in set(r[b] for r in data):
        sub = [r for r in data if r[b] == v]
        tree[head[b]][v] = build(sub, [c for c in cols if c!=b], head)

    return tree

# main
with open("dataa.csv") as f:
    rows = list(csv.reader(f))
    head, data = rows[0], rows[1:]

tree = build(data, list(range(len(head)-1)), head)
print(tree)
