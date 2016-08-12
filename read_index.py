#!/usr/bin/env python3

import os, struct


# We expect to find three files in an index directory:
DOCUMENTS_FILE = "documents.txt"  # list of source document filenames
TERMS_FILE = "terms.txt"          # in-memory summary data
INDEX_DATA_FILE = "index.dat"     # all hit data, grouped by term


class Index:
    """ Object for answering queries about a document set.

    To use `Index` you have to have an index directory created by `build_index.py`.
    Pass the name of that directory to `Index()`; it will read some data from disk
    and stand ready to answer queries (see `read_term`).
    """

    def __init__(self, index_dir):
        # Load the list of source documents.
        with open(os.path.join(index_dir, DOCUMENTS_FILE)) as df:
            self.documents = [line[:-1] for line in df]

        # Open the file that contains all hit data.
        self.data_file = open(os.path.join(index_dir, INDEX_DATA_FILE), "rb")

        # Load the entire summary. This can be pretty big, about half the size
        # of all the documents!
        self.terms = {}
        with open(os.path.join(index_dir, TERMS_FILE)) as f:
            for line in f:
                term, df, offset_range = line.split()
                if term in self.terms:
                    raise ValueError("term listed multiple times: {!r}".format(term))
                start_str, stop_str = offset_range.split("..")
                self.terms[term] = (int(df), int(start_str, 16), int(stop_str, 16))

    def read_term(self, term):
        """ Find out where a term appears.

        This returns a pair `(ndocs, docs)`, where `docs` is an iterator
        and `ndocs` is the number of values that `docs` will produce.

        Each value yielded by `docs` is a pair `(filename, hits)` where
        `filename` is a string (the filename of a source document) and
        `hits` is a list of ints telling where `term` appears in that document.
        The numbers count words, not characters, so if `idx.read_term("pennant")`
        returns an iterator that yields

            ('yankees.txt', [147, 2062])

        then words #147 and #2062 of `yankees.txt` are both "pennant".
        """

        # If we've never heard of this term, it doesn't appear in the corpus.
        if term not in self.terms:
            return 0, []

        # Read all hit data for this term from the data file. The summary tells
        # us where it's stored; we can read it all in one gulp. It may be a lot
        # or a little, it just depends on the term.
        df, start, stop = self.terms[term]
        self.data_file.seek(start)
        bytes = self.data_file.read(stop - start)

        # index_entries() interprets the raw binary data we just read.
        return df, self.index_entries(bytes)

    def index_entries(self, bytes):
        # Start at the beginning of `bytes` and scan through to the end.
        offset = 0         # current read position within `bytes`
        stop = len(bytes)  # stopping point
        while offset < stop:
            # Read and yield the hits in one document.

            # The data for one document starts with the document id (4 bytes)
            # and the number of hits in that document (4 bytes).
            if stop - offset < 8:
                raise ValueError("entry glitch")
            doc_id, nhits = struct.unpack("<II", bytes[offset:offset+8])
            offset += 8

            # Then there's an integer (4 bytes) for each hit within that
            # document. (If one of these integers is 147, it means the 147th
            # word in this particular document is a search hit.)
            if stop - offset < nhits * 4:
                raise ValueError("entry length glitch")
            hits = [struct.unpack("<I", bytes[i:i+4])[0]
                    for i in range(offset, offset + nhits * 4, 4)]
            offset += nhits * 4

            yield (self.documents[doc_id], hits)


def handle_query(index, query):
    df, entries = index.read_term(query)
    i = 0
    for filename, hits in entries:
        i += 1
        if len(hits) == 1:
            print(filename)
        else:
            print("{} ({} hits)".format(filename, len(hits)))
    print("{} entries".format(i))
    if i != df:
        print("*** expected {} entries".format(df))
        print((i, df))


index = Index("..")
while True:
    try:
        query = input("> ").strip()
    except EOFError as exc:
        break
    if query != "":
        handle_query(index, query)
