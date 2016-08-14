#!/usr/bin/env python3
# build_index.py - Build a search index for a collection of text files.

from collections import defaultdict, Counter
import os, re, struct

DOCUMENTS_FILE = "documents.txt"
TERMS_FILE = "terms.txt"
INDEX_DATA_FILE = "index.dat"

BYTES_PER_WORD = 4

class Term:
    """ Bookkeeping for a term. One of these is created for each distinct term
    (searchable word) that occurs in the search corpus.
    """

    def __init__(self):
        self.df = 0      # document frequence (number of docs that contain this term)
        self.nbytes = 0  # size of search hit data for this term, in bytes (computed in step 1)
        self.start = 0   # byte offset within the index data file where hit data is stored (step 2)
        self.nbytes_written = 0  # amount of hit data already written, in bytes (used during step 3)

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


def tokenize_file(filename):
    """ Load a text file. Return a list of all its words, in order of appearance. """
    with open(filename) as f:
        text = f.read().lower()
    return re.findall(r'[a-z0-9/]+', text)


def build_index(source_dir, index_dir):
    documents = []
    terms = defaultdict(Term)

    # Step 1: Read all source files. Find all words that occur in those files and
    # how much storage it will take to save all the search hits for each word.
    with open(os.path.join(index_dir, DOCUMENTS_FILE), "w") as documents_file:
        for filename in sorted(os.listdir(source_dir)):
            if len(filename.split()) != 1:
                print("*** skipping file {!r} (space in filename)".format(filename))
                continue

            print(filename)
            documents.append(filename)
            documents_file.write(filename + "\n")

            tokens = tokenize_file(os.path.join(source_dir, filename))
            for term, freq in Counter(tokens).items():
                terms[term].add(freq)

    # Step 2: Write the "terms file", a text file with one line per term.  This
    # involves deciding, for every word, exactly where (at what offset) within
    # the "index data file" we are going to store all the search hits.
    with open(os.path.join(index_dir, TERMS_FILE), "w") as terms_file:
        point = 0
        for term, md in sorted(terms.items(), key=lambda pair: (-pair[1].nbytes, pair[0])):
            md.start = point
            point += md.nbytes
            terms_file.write("{} {} {:x}..{:x}\n".format(term, md.df, md.start, point))

    # Step 3: Read all the files a second time to build the binary index data file.
    print("writing index data...")
    with open(os.path.join(index_dir, INDEX_DATA_FILE), "wb") as index_data_file:
        for document_id, filename in enumerate(documents):
            print(filename)

            tokens = tokenize_file(os.path.join(source_dir, filename))
            offsets_by_term = defaultdict(list)
            for i, term in enumerate(tokens):
                offsets_by_term[term].append(i)

            for term, offsets in offsets_by_term.items():
                if term not in terms:
                    print("*** term {!r} not found (file changed on disk during index building?)".format(term))
                    continue
                terms[term].write_to(index_data_file, document_id, offsets)

    # Step 4: Check that the second time we read all those files, we got the
    # same number of search hits for each word that we did the first time.  Any
    # output here indicates a bug, or else a source file changed on disk
    # between step 1 and step 3.
    for term, md in terms.items():
        if md.nbytes_written != md.nbytes:
            print("*** term {!r}: expected {} bytes, wrote {} bytes".format(term, md.nbytes, md.nbytes_written))

build_index("../sample", "..")
