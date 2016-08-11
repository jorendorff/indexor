# indexor - A toy search engine!

It's not done yet! But it can do some things:

1.  Download https://www.dropbox.com/s/lv44vyl8ia46llx/sample.tar.bz2?dl=0
    and unzip it *next to* this directory.
    It creates a directory named `sample` (that should be a sibling directory of this repo)
    with about 8,476 files in it.

2.  Run `./build_index.py` to try to build the index.

    This creates a few big files in the parent directory.
    With our sample from step 1, they take up half a GB of disk space.
    (It would be pretty easy to cut that in half; haven't bothered yet.)

3.  Run `./read_index.py` to read raw entries out of the index.

    This is something less than a real search engine.
    You can only query one term at a time, and the output is bare-bones.
    But it works!
    With our sample, it takes a few seconds to start up;
    after that, queries are fast.
