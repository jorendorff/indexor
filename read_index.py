import struct

DOCUMENTS_FILE = "../documents.txt"
TERMS_FILE = "../terms.txt"
INDEX_DATA_FILE = "../index.dat"

class Index:
    def __init__(self, documents_file, terms_file, index_data_filename):
        with open(documents_file) as df:
            self.documents = [line[:-1] for line in df]

        terms = {}
        self.data_file = open(index_data_filename, "rb")

        with open(terms_file) as f:
            for line in f:
                term, df, offset_range = line.split()
                if term in terms:
                    raise ValueError("term listed multiple times: {:r}".format(term))
                start_str, stop_str = offset_range.split("..")
                terms[term] = (int(df), int(start_str, 16), int(stop_str, 16))

        self.terms = terms

    def read_term(self, term):
        if term not in self.terms:
            return 0, []

        df, start, stop = self.terms[term]
        self.data_file.seek(start)
        bytes = self.data_file.read(stop - start)

        return df, self.index_entries(bytes)

    def index_entries(self, bytes):
        offset = 0
        stop = len(bytes)
        while offset < stop:
            if stop - offset < 8:
                raise ValueError("entry glitch")
            doc_id, nhits = struct.unpack("<II", bytes[offset:offset+8])
            offset += 8

            if stop - offset < nhits * 4:
                raise ValueError("entry length glitch")
            hits = [struct.unpack("<I", bytes[i:i+4])[0]
                    for i in range(offset, offset + nhits * 4, 4)]
            offset += nhits * 4

            yield (self.documents[doc_id], hits)

index = Index(DOCUMENTS_FILE, TERMS_FILE, INDEX_DATA_FILE)

def handle_query(query):
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

while True:
    try:
        query = input("> ").strip()
    except EOFError as exc:
        break
    if query != "":
        handle_query(query)
