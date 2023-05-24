import tqdm

lines = []

cats = {}

with open('yeast.data') as f:
    for ln in tqdm.tqdm(f.readlines()):
        ln = ln.split('  ')
        ln = ln[1:]
        if ln[-1] not in cats.keys():
            cats[ln[-1]] = len(cats)
        ln[-1] = str(cats[ln[-1]])
        lines.append(' '.join(ln))

print(cats)

with open('data.txt', 'w+') as f:
    for ln in lines:
        f.write(ln.strip() + '\n')
