#!/usr/bin/env python3

import os, collections, re

DIR = "../sample"
filenames = sorted(os.listdir(DIR))
del filenames[250:]
titles = []

index_by_term = collections.defaultdict(set)

for i, filename in enumerate(filenames):
    print(filename)
    with open(os.path.join(DIR, filename)) as f:
        title = f.readline().strip()
        titles.append(title)

        text = f.read().lower()

        match_criteria = re.compile(r'[^a-zA-Z0-9\/]|_')
        clear_text = re.sub(match_criteria, ' ', text)

        tokens = clear_text.split()

    for term in tokens:
        index_by_term[term].add(i)

print("Loaded {} documents.".format(len(filenames)))
print("There are {} distinct terms in the corpus.".format(len(index_by_term)))

def handle_query(query):
    results = index_by_term[query]
    if results:
        for doc_index in list(results)[:10]:
            print("{:20s} {}".format(titles[doc_index], filenames[doc_index]))
    else:
        print("no hits")

while True:
    try:
        query = input("> ").strip()
    except EOFError as exc:
        break
    if query != "":
        handle_query(query)
