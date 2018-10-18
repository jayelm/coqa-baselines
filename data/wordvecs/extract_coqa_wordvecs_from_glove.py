with open('coqa.crawl-300d-2M.vec', 'r') as fin:
    _ = next(fin)  # First line
    coqa_tokens = set()
    for line in fin:
        token, *vals = line.split(' ')
        coqa_tokens.add(token)

linestowrite = []

with open('glove.6B.50d.txt', 'r') as glovin:
        n, dim = next(glovin).split(' ')
        for line in glovin:
            token, *vals = line.split(' ')
            if token in coqa_tokens:
                linestowrite.append(line)

with open('coqa.glove.6B.50d.txt', 'w') as fout:
    new_n = len(linestowrite)
    fout.write('{} {}'.format(new_n, dim))
    for line in linestowrite:
        fout.write(line)
