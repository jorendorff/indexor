//! Build a search index for a collection of text files.
//!
//! This builds an index that's compatible with the one built by build_index.py,
//! using a different algorithm to produce (roughly) the same result.
//!
//! The original does two passes over the input data; the second pass uses a
//! lot of seeks. Seeks are slow. In this implementation, the first pass over the
//! input creates bite-sized index files (perhaps 6.5MB each). Once it has
//! created all these files, it merges them. It's basically transposing a large
//! matrix using a huge merge sort.  This involves O(log N) passes over the data;
//! but there are no seeks. It's faster, for the small (500MB) sample I'm using
//! to benchmark this.

extern crate byteorder;

use std::fs::File;
use std::io::{self, BufReader};
use byteorder::{LittleEndian, WriteBytesExt};


const DIR: &'static str = "./playpen/sample";
const DOCUMENTS_FILE: &'static str = "./playpen/documents.txt";
const TERMS_FILE: &'static str = "./playpen/terms.txt";
const INDEX_DATA_FILE: &'static str = "./playpen/index.dat";
const TMP_DIR: &'static str = "./playpen/tmp";

struct TermHeader {
    term_id: u32,
    nbytes: u32,
    df: u32
}

const TERM_HEADER_NBYTES: usize = 12;

impl TermHeader {
    fn read(f: &mut BufReader<File>) -> io::Result<Option<TermHeader>> {
        let mut buf = [0; 12];
        match f.read_exact(&mut buf) {
            Err(err) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    return Ok(None);
                } else {
                    return Err(err);
                }
            }
            Ok(()) => {}
        }
        Ok(Some(TermHeader {
            term_id: LittleEndian::read_u32(&buf[0..4]),
            nbytes: LittleEndian::read_u32(&buf[4..8]),
            df: LittleEndian::read_u32(&buf[8..12])
        }))
    }

    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        try!(w.write_u32::<LittleEndian>(self.term_id));
        try!(w.write_u32::<LittleEndian>(self.nbytes));
        try!(w.write_u32::<LittleEndian>(self.df));
        Ok(())
    }
}

struct Stream {
    f: BufReader<File>,
    next_header: Option<TermHeader>
}

impl Stream {
    fn new(f: BufReader<File>) -> io::Result<Stream> {
        let next_header = try!(TermHeader::read(&mut f));
        Ok(Stream { f, next_header })
    }

    fn has_next(&self) -> bool { self.next_header.is_some() }

    /// Copy the payload of the current term record to the specified output stream,
    /// then read the header for the next term record.
    fn copy_payload_to<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        let nbytes = self.next_header.expect("do not try to copy_payload_to if no header").nbytes;
        let mut buf = Vec::with_capacity(nbytes);
        buf.resize(nbytes, 0);
        try!(f.read_exact(&mut buf));
        try!(out.write_all(&buf));
        let header = try!(TermHeader::read(&mut self.f));
        self.next_header = header;
        Ok(())
    }
}

trait MergeLog {
    fn log_merged_entry(&mut self, term_id: u32, df: u32, start: u64, nbytes: u64);
}

impl MergeLog for () {
    fn log_merged_entry(&mut self, _term_id: u32, _df: u32, _start: u64, _nbytes: u64) {}
}    

impl MergeLog for Vec<(u32, u32, u64, u64)> {
    fn log_merged_entry(&mut self, term_id: u32, df: u32, start: u64, nbytes: u64) {
        self.push((term_id, df, start, nbytes));
    }
}

fn merge_streams<W: Write, L: MergeLog>(streams: &[Stream], out: &mut W, log: &mut L)
    -> io::Result<()>
{
    println!("Merging {} streams...", streams.len());

    let mut point: u64 = 0;
    let mut count = streams.iter().filter(|s| s.has_next()).count();
    while count > 0 {
        let mut term_id = None;
        let mut nbytes = 0;
        let mut df = 0;
        for s in streams {
            match s.next_header {
                None => {}
                Some(ref hdr) => {
                    let s_term = hdr.term_id;
                    if term_id.is_none() || s_term < term_id.unwrap() {
                        term_id = Some(s_term);
                        nbytes = hdr.nbytes;
                        df = hdr.df;
                    } else if s_term == term_id.unwrap() {
                        nbytes += hdr.nbytes;
                        df += hdr.df;
                    }
                }
            }
        }

        let entry_header = TermHeader {
            term_id: term_id.expect("bug in algorithm!"),
            nbytes: nbytes,
            df: df
        };
        try!(entry_header.write_to(out));
        point += TERM_HEADER_NBYTES as u64;

        for s in streams {
            match s.next_header {
                Some(TermHeader { term_id: s_term_id }) | s_term_id == term_id => {
                    try!(s.copy_payload_to(out));
                    if !s.has_next() {
                        count -= 1;
                    }
                }
                _ => {}
            }
        }
        log.log_merged_entry(term_id, df, point, nbytes as u64);
        point += nbytes as u64;
    }

    assert!(streams.all(|s| !s.has_next()));
    try!(out.flush());

    println!("...done, {} bytes written", point);
}



/// NOTE: this destroys files as they are read!
fn merge_many_files(filenames: &[&Path], out_filename: &Path) -> io::Result<()> {
    // How many streams to merge at a time
    const NSTREAMS: usize = 8;

    let mut stacks = vec![vec![]];

    for filename in filenames {
        push(&mut stacks, filename);
    }
    

    
    
}

// Suppose you have 64 files. Then the only thing that makes sense is to go
// 8x8, then 1x8.  Clearly that is better than going three rounds by doing
// 16x4, then 2x8, then 1x2. But if you have a choice, should you prefer an
// early round that produces as few files as possible, or as many as possible?
// I think as few as possible, inasmuch as in theory merge has an O(log
// NSTREAMS) constant factor -- and in practice there's an O(NSTREAMS) constant
// factor since we don't actually build a heap.

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



extern crate rayon;

/// Using a specially stupid hash function is good for a 3% overall speed boost.
mod fasthash {
    use std;
    use std::num::Wrapping;
    use std::hash::{Hash, Hasher, BuildHasher};

    pub struct FastHasher(Wrapping<u64>);

    impl Hasher for FastHasher {
        fn write(&mut self, bytes: &[u8]) {
            // Bob Jenkins' one-at-a-time hash, thoughtlessly extended to 64 bits
            let mut h = self.0;
            for &b in bytes {
                h += Wrapping(b as u64);
                h += h << 10;
                h ^= h >> 6;
            }
            self.0 = h;
        }
        fn finish(&self) -> u64 {
            let mut h = self.0;
            h += h << 3;
            h ^= h >> 11;
            h += h << 15;
            h.0
        }
    }

    pub struct UseFastHasher;

    impl BuildHasher for UseFastHasher {
        type Hasher = FastHasher;
        fn build_hasher(&self) -> FastHasher { FastHasher(Wrapping(0)) }
    }

    pub type HashMap<K, V> = std::collections::HashMap<K, V, UseFastHasher>;

    pub fn new_hash_map<K: Eq + Hash, V>() -> HashMap<K, V> {
        HashMap::with_hasher(UseFastHasher)
    }
}

mod stopwatch {
    use std::time::{Instant, Duration};

    pub struct Stopwatch {
        start: Instant,
        last: Instant
    }

    fn to_seconds(d: Duration) -> f64 {
        (d.as_secs() as f64) + 1e-9 * (d.subsec_nanos() as f64)
    }

    impl Stopwatch {
        pub fn new() -> Stopwatch {
            let t = Instant::now();
            Stopwatch { start: t, last: t }
        }

        pub fn log(&mut self, msg: &str) {
            let t = Instant::now();
            let d1 = t - self.start;
            let d2 = t - self.last;
            println!("{:7.3}s {:7.3}s {}", to_seconds(d1), to_seconds(d2), msg);
            self.last = t;
        }
    }
}

use std::io;
use std::io::{BufWriter, SeekFrom};
use std::io::prelude::*;
use std::fs;
use std::fs::{File, DirEntry};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use rayon::prelude::*;
use fasthash::{HashMap, new_hash_map};
use stopwatch::Stopwatch;

const BYTES_PER_WORD: usize = 4;

fn read_file_lowercase(filename: &Path) -> io::Result<String> {
    let mut bytes = vec![];
    {
        let mut f = try!(File::open(filename));
        try!(f.read_to_end(&mut bytes));
    }

    // We used to use f.read_to_string() and then str::to_lowercase() here. But
    // str::to_lowercase() is slow: it decodes the entire string as UTF-8 and
    // copies it to a new buffer, re-encoding it back into UTF-8, one character
    // at a time. This was soaking up 64% of the CPU time spent by the whole
    // program. We don't need it: we're going to ignore all non-ASCII text
    // anyway. So here we walk the buffer, clobbering non-ASCII bytes and
    // downcasing ASCII letters.
    for b in &mut *bytes {
        if *b >= b'A' && *b <= b'Z' {
            *b += b'a' - b'A';
        } else if *b > b'\x7f' {
            *b = b'?';
        }
    }

    // This unwrap() can't fail because we eliminated all non-ASCII bytes above.
    Ok(String::from_utf8(bytes).unwrap())
}

fn tokenize(text: &str) -> Vec<&str> {
    static WORD_CHARS: [bool; 128] = [
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, true,
        true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  false, false, false, false, false, false,

        false, true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  false, false, false, false, false,
        false, true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  false, false, false, false, false,
        ];
    let is_word_byte = |b| b < 0x80 && WORD_CHARS[b as usize];

    let mut words = vec![];

    let bytes = text.as_bytes();
    let stop = bytes.len();
    let mut i = 0;
    'pass: while i < stop {
        while !is_word_byte(bytes[i]) {
            i += 1;
            if i == stop { break 'pass; }
        }
        let mut j = i + 1;
        while j < stop && is_word_byte(bytes[j]) {
            j += 1;
        }
        words.push(&text[i..j]);
        i = j + 1;
    }

    words
}

fn unpoison<T, U>(lock_result: Result<T, std::sync::PoisonError<U>>) -> io::Result<T> {
    match lock_result {
        Ok(v) => Ok(v),
        Err(_) => Err(io::Error::new(std::io::ErrorKind::Other,
                                     "contagion (some other thread panicked in a mutex)"))
    }
}

struct TakeFirstError;

impl<E> rayon::par_iter::reduce::ReduceOp<Result<(), E>> for TakeFirstError {
    fn start_value(&self) -> Result<(), E> {
        Ok(())
    }
    fn reduce(&self, value1: Result<(), E>, value2: Result<(), E>) -> Result<(), E> {
        value1.and(value2)
    }
}


fn load_files(entries: Vec<io::Result<DirEntry>>,
              documents_mutex: &Mutex<Vec<String>>,
              terms_mutex: &Mutex<HashMap<String, Term>>)
    -> io::Result<()>
{
    let result = entries
        .par_iter()
        .weight_max()
        .map(|entry| -> io::Result<()> {
            let entry = match entry {
                &Ok(ref e) => e,
                &Err(ref err) => {
                    // Can't clone err:
                    // return Err((*err).clone())  // no method named `clone` found
                    // so fake it
                    return Err(io::Error::new(std::io::ErrorKind::Other,
                                              format!("{}", *err)));
                }
            };
            let filename = match entry.file_name().into_string() {
                Ok(s) => s,
                Err(_) => {
                    println!("*** skipping file {:?} (non-unicode bytes in filename)", entry.path());
                    return Ok(());
                }
            };
            if filename.split_whitespace().count() != 1 {
                println!("*** skipping file {:?} (space in filename)", filename);
                return Ok(());
            }

            {
                let mut documents = try!(unpoison(documents_mutex.lock()));
                documents.push(filename);
            }

            let text = try!(read_file_lowercase(&entry.path()));
            let tokens = tokenize(&text);
            let mut counter: HashMap<&str, usize> = new_hash_map();
            for term in tokens {
                let n = counter.entry(term).or_insert(0);
                *n += 1;
            }
            let mut terms = try!(unpoison(terms_mutex.lock()));
            for (term_str, freq) in counter {
                if let Some(term_md) = terms.get_mut(term_str) {
                    term_md.add(freq);
                    continue;
                }
                let mut term_md = Term::new();
                term_md.add(freq);
                terms.insert(term_str.to_string(), term_md);
            }

            Ok(())
        })
        .reduce(&TakeFirstError);
    try!(result);
    Ok(())
}

fn write_documents_file(documents: &Vec<String>) -> io::Result<()> {
    let mut documents_file = BufWriter::new(try!(File::create(DOCUMENTS_FILE)));
    for filename in documents {
        try!(writeln!(documents_file, "{}", filename));
    }
    Ok(())
}

fn spawn_terms_file_writer(terms_output_records: Vec<(String, usize, u64, u64)>)
    -> std::thread::JoinHandle<io::Result<()>>
{
    // ...Then we send that data to a separate thread to be written to disk.
    std::thread::spawn(move || {
        let mut terms_file = BufWriter::new(try!(File::create(TERMS_FILE)));
        for (term_str, df, start, stop) in terms_output_records {
            try!(writeln!(terms_file, "{} {} {:x}..{:x}", term_str, df, start, stop));
        }
        Ok(())
    })
}

fn make_index() -> io::Result<()> {
    ???
    Ok(())
}

fn main() {
    match make_index() {
        Err(err) => println!("build_index: {}", err),
        Ok(()) => {}
    }
}
