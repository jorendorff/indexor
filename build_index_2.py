#!/usr/bin/env python3
# build_index_2.py - Build a search index for a collection of text files.
#
# This builds an index that's compatible with the one built by build_index.py,
# using a different algorithm to produce (roughly) the same result.
#
# build_index.py does two passes over the input data; the second pass uses a
# lot of seeks. Seeks are slow. In this implementation, the first pass over the
# input creates bite-sized index files (perhaps 6.5MB each). Once it has
# created all these files, it merges them. It's basically transposing a large
# matrix using a huge merge sort.  This involves O(log N) passes over the data;
# but there are no seeks. It's faster, for the small (500MB) sample I'm using
# to benchmark this.

from collections import defaultdict, deque
import os, re, struct, tempfile

DOCUMENTS_FILE = "documents.txt"
TERMS_FILE = "terms.txt"
INDEX_DATA_FILE = "index.dat"
TMP_DIR = "tmp"

BYTES_PER_WORD = 4

HEADER_BYTES = 3 * BYTES_PER_WORD

class Stream:
    def __init__(self, f):
        self.f = f
        self.reload()

    def has_next(self):
        return self.next_term_id is not None

    def reload(self):
        entry_header = self.f.read(HEADER_BYTES)
        if len(entry_header) == 0:
            self.next_term_id = self.next_nbytes = self.next_df = None
        else:
            self.next_term_id, self.next_nbytes, self.next_df = struct.unpack("<III", entry_header)

    def copy_payload_to(self, out):
        out.write(self.f.read(self.next_nbytes))
        self.reload()


def merge_streams(streams, out):
    print("Merging {} streams...".format(len(streams)))

    summary = {}
    point = 0
    count = sum(s.has_next() for s in streams)
    while count:
        term_id = None
        df = None
        nbytes = None
        for s in streams:
            s_term = s.next_term_id
            if s_term is not None:
                if term_id is None or s_term < term_id:
                    term_id = s_term
                    nbytes = s.next_nbytes
                    df = s.next_df
                elif s_term == term_id:
                    nbytes += s.next_nbytes
                    df += s.next_df

        entry_header = struct.pack("<III", term_id, nbytes, df)
        out.write(entry_header)
        for s in streams:
            if s.next_term_id == term_id:
                s.copy_payload_to(out)
                if not s.has_next():
                    count -= 1
        summary[term_id] = (df, point, HEADER_BYTES + nbytes)
        point += HEADER_BYTES + nbytes

    assert all(not s.has_next() for s in streams)
    out.flush()

    print("...done, {} bytes written".format(point))

    return summary

def merge_many_files(filenames, out_filename):
    """ NOTE: destroys files as they are read """

    # How many streams to merge at a time
    NSTREAMS = 8

    q = deque(filenames)

    next_list = []
    def merge_next():
        final_merge = len(q) == 0
        if final_merge:
            out = open(out_filename, 'wb')
        else:
            out = tempfile.TemporaryFile()
        term_data = merge_streams(next_list, out)
        del next_list[:]
        if not final_merge:
            out.seek(0)
            q.append(out)
        return term_data

    while q:
        f = q.popleft()
        if isinstance(f, str):
            # A filename.
            next_list.append(Stream(open(f, 'rb')))
            os.unlink(f)
        else:
            # A temporary file.
            next_list.append(Stream(f))
        if len(next_list) == NSTREAMS or len(q) == 0:
            term_data = merge_next()

    return term_data


def tokenize_file(filename):
    """ Load a text file. Return a list of all its words, in order of appearance. """
    with open(filename) as f:
        text = f.read().lower()
    return re.findall(r'[a-z0-9/]+', text)


def save_tmp_file(tmp_buffer, filename):
    print("dumping {} ...".format(filename))
    with open(filename, 'wb') as f:
        for term_id, (df, term_data) in sorted(tmp_buffer.items()):
            # entry header and document header togther:
            f.write(struct.pack("<III", term_id, len(term_data), df))
            f.write(term_data)
        print("...wrote {} bytes".format(f.tell()))

def build_index(source_dir, index_dir):
    documents = []

    def generate_term_id():
        n = generate_term_id.next
        generate_term_id.next += 1
        return n
    generate_term_id.next = 0

    terms = defaultdict(generate_term_id)

    tmp_dir = os.path.join(index_dir, TMP_DIR)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    tmp_filenames = []

    # Step 1: Read all source files. Translate each one into a temporary data file.
    with open(os.path.join(index_dir, DOCUMENTS_FILE), "w") as documents_file:
        NICE_SIZE = 1000000 # one million words
        tmp_buffer = defaultdict(lambda: (0, bytearray()))
        tmp_word_count = 0
        file_list = sorted(os.listdir(source_dir))
        for file_index, filename in enumerate(file_list):
            if len(filename.split()) != 1:
                print("*** skipping file {!r} (space in filename)".format(filename))
                continue

            document_id = len(documents)
            documents.append(filename)
            documents_file.write(filename + "\n")

            indexed = defaultdict(bytearray)
            tokens = tokenize_file(os.path.join(source_dir, filename))
            for i, t in enumerate(tokens):
                term_id = terms[t]
                indexed[term_id] += struct.pack("<I", i)

            # Copy the little index into the middle-sized index.
            for term_id, offsets in indexed.items():
                df, term_bytes = tmp_buffer[term_id]
                df += 1
                term_bytes += struct.pack("<II", document_id, len(offsets) // BYTES_PER_WORD)
                term_bytes += offsets
                tmp_buffer[term_id] = df, term_bytes
            tmp_word_count += len(tokens)
            indexed = None

            # If the middle-sized index is big enough (or we're on the last file) flush to disk.
            if tmp_word_count >= NICE_SIZE or file_index == len(file_list) - 1:
                tmp_filename = os.path.join(tmp_dir, filename + ".dat")
                save_tmp_file(tmp_buffer, tmp_filename)
                tmp_filenames.append(tmp_filename)
                tmp_buffer = defaultdict(lambda: (0, bytearray()))
                tmp_word_count = 0

    # Step 2: Merge all temp files.
    term_data = merge_many_files(tmp_filenames, os.path.join(index_dir, INDEX_DATA_FILE))

    # Step 3: Write the terms file.
    with open(os.path.join(index_dir, TERMS_FILE), 'w') as f:
        for term, term_id in sorted(terms.items(), key=lambda pair: pair[1]):
            df, start, nbytes = term_data[term_id]
            f.write("{} {} {:x}..{:x}\n".format(term, df, start + HEADER_BYTES, start + nbytes))


build_index("../sample", "..")
