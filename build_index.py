#!/usr/bin/env python3

import os
import collections
import shelve

DIR = "../sample"

with shelve.open("search-index") as index:
    for filename in sorted(os.listdir(DIR)):
        print(filename)
        with open(os.path.join(DIR, filename)) as f:
            tokens = f.read().lower().split()

            # Now store every token
            for token in tokens:
                if token in index:
                    temp = index[token]  # big O(n) read
                else:
                    temp = []
                temp.append(filename)
                index[token] = temp  # big O(n) write

 
