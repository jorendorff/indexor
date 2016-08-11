#!/usr/bin/env python3

from collections import defaultdict, Counter
import os, re, struct

DIR = "../sample"

DOCUMENTS_FILE = "../documents.txt"
TERMS_FILE = "../terms.txt"
INDEX_DATA_FILE = "../index.dat"

BYTES_PER_WORD = 4

class Term:
    def __init__(self):
        self.df = 0
        self.start = 0
        self.nbytes = 0
        self.nbytes_written = 0

    def add(self, n):
        self.df += 1
        self.nbytes += (2 + n) * BYTES_PER_WORD

    def write_to(self, f, doc_id, offsets):
        f.seek(self.start + self.nbytes_written)
        binary_data = (struct.pack("<II", doc_id, len(offsets)) +
                       b''.join(struct.pack("<I", off) for off in offsets))
        write_size = len(binary_data)
        assert write_size == (2 + len(offsets)) * BYTES_PER_WORD  # match formula in add() above
        if self.nbytes_written + write_size > self.nbytes:
            print("*** Error: writing too much data for a term; index would be corrupted")
            return
        f.write(binary_data)
        self.nbytes_written += write_size


documents = []
terms = defaultdict(Term)

def tokenize_file(filename):
    with open(os.path.join(DIR, filename)) as f:
        text = f.read().lower()

    garbage_re = re.compile(r'[^a-zA-Z0-9\/]')
    clear_text = re.sub(garbage_re, ' ', text)
    tokens = clear_text.split()
    return tokens

with open(DOCUMENTS_FILE, "w") as documents_file:
    for filename in sorted(os.listdir(DIR)):
        if filename.split() != filename:
            print("*** skipping file {:r} (space in filename)".format(filename))
            continue

        print(filename)
        documents.append(filename)
        documents_file.write(filename + "\n")

        tokens = tokenize_file(filename)
        for term, freq in Counter(tokens).items():
            terms[term].add(freq)

with open(TERMS_FILE, "w") as terms_file:
    point = 0
    for term, md in sorted(terms.items(), key=lambda pair: (-pair[1].nbytes, pair[0])):
        md.start = point
        point += md.nbytes
        terms_file.write("{} {} {:x}..{:x}\n".format(term, md.df, md.start, point))

print("writing index data...")
with open(INDEX_DATA_FILE, "wb") as index_data_file:
    for document_id, filename in enumerate(documents):
        print(filename)

        tokens = tokenize_file(filename)
        offsets_by_term = defaultdict(list)
        for i, term in enumerate(tokens):
            offsets_by_term[term].append(i)

        for term, offsets in offsets_by_term.items():
            if term not in terms:
                print("*** term {:r} not found (file changed on disk during index building?)".format(term))
                continue
            terms[term].write_to(index_data_file, document_id, offsets)

for term, md in terms.items():
    if md.nbytes_written != md.nbytes:
        print("*** term {:r}: expected {} bytes, wrote {} bytes".format(term, md.nbytes, md.nbytes_written))

