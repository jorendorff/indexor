//! Build a search index for a collection of text files.
//!
//! This builds an index that's compatible with the one built by build_index.py,
//! using a different algorithm to produce (roughly) the same result.
//!
//! The original did two passes over the input data; the second pass used a lot
//! of seeks. Seeks are slow. In this implementation, the first pass over the
//! input creates bite-sized index files (perhaps 6.5MB each). Once it has
//! created all these files, it merges them. It's basically transposing a large
//! matrix using a huge merge sort.  This involves O(log N) passes over the
//! data; but there are no seeks. It's faster, for the small (500MB) sample I'm
//! using to benchmark this.

#![deny(unused_must_use)]

extern crate byteorder;

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter};
use std::io::prelude::*;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use std::thread::{self, JoinHandle};
use std::sync::mpsc::{self, Receiver};

use fasthash::{HashMap, new_hash_map};
use stopwatch::Stopwatch;


const CORPUS_DIR: &'static str = "./playpen/sample";
const INDEX_DIR: &'static str = "./playpen";

const DOCUMENTS_FILE: &'static str = "documents.txt";
const TERMS_FILE: &'static str = "terms.txt";
const INDEX_DATA_FILE: &'static str = "index.dat";
const TMP_DIR: &'static str = "tmp";


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


// --- Fun stopwatch for crude performance measurement --------------------------------------------

mod stopwatch {
    use std::time::{Instant, Duration};

    pub struct Stopwatch {
        sigil: char,
        start: Instant,
        last: Instant
    }

    fn to_seconds(d: Duration) -> f64 {
        (d.as_secs() as f64) + 1e-9 * (d.subsec_nanos() as f64)
    }

    impl Stopwatch {
        pub fn new(sigil: char) -> Stopwatch {
            let t = Instant::now();
            Stopwatch { sigil: sigil, start: t, last: t }
        }

        pub fn log<S: AsRef<str>>(&mut self, msg: S) {
            let t = Instant::now();
            let d1 = t - self.start;
            let d2 = t - self.last;
            println!("{} {:7.3}s {:7.3}s {}", self.sigil, to_seconds(d1), to_seconds(d2), msg.as_ref());
            self.last = t;
        }
    }
}


// --- Stage 0: Figuring out which files to index -------------------------------------------------

fn list_dir<D: AsRef<Path>>(dir: D) -> io::Result<Vec<OsString>> {
    let mut v: Vec<OsString> =
        try!(
            try!(fs::read_dir(dir))
                .map(|r| r.map(|entry| entry.file_name().to_owned()))
                .collect());
    v.sort();
    Ok(v)
}

fn write_documents_file(documents: &Vec<String>, documents_filename: &Path) -> io::Result<()> {
    let mut documents_file = BufWriter::new(try!(File::create(documents_filename)));
    for filename in documents {
        try!(writeln!(documents_file, "{}", filename));
    }
    Ok(())
}

/// Make a list of documents to index. Before returning, this saves the list to DOCUMENTS_FILE.
fn list_documents(source_dir: &Path, index_dir: &Path) -> io::Result<Vec<String>> {
    let file_list = try!(list_dir(source_dir));
    let documents: Vec<String> =
        file_list.iter()
        .filter_map(|os_filename| {
            match os_filename.to_str() {
                None => {
                    println!("*** skipping file {:?} (filename is not valid unicode)", os_filename);
                    None
                }
                Some(s) => {
                    if s.find(char::is_whitespace).is_some() {
                        println!("*** skipping file {:?} (space in filename)", s);
                        None
                    } else {
                        Some(s.to_string())
                    }
                }
            }
        })
        .collect();

    try!(write_documents_file(&documents, &index_dir.join(DOCUMENTS_FILE)));

    Ok(documents)
}


// --- Stage 1: File input ------------------------------------------------------------------------

fn read_file_lowercase<P: AsRef<Path>>(filename: P) -> io::Result<String> {
    let filename = filename.as_ref();

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

fn read_source_files(source_dir: &Path, documents: Vec<String>)
    -> (Receiver<String>, JoinHandle<io::Result<()>>)
{
    let (sender, receiver) = mpsc::sync_channel(32);

    let source_dir = source_dir.to_owned();
    let handle = thread::spawn(move || {
        //let mut s = Stopwatch::new('R');
        for filename in documents {
            let path = source_dir.join(filename);
            let res = try!(read_file_lowercase(path));
            //s.log("read file");
            sender.send(res).unwrap();
            //s.log("sent");
        }
        //s.log("done");
        Ok(())
    });

    (receiver, handle)
}



// --- Stage 2: Tokenization ----------------------------------------------------------------------

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

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct TermId(u32);

// A TermMap assigns a unique id to each distinct term (word) that we encounter.
struct TermMap {
    term_strings: Vec<String>,
    terms: HashMap<String, TermId>,
}

impl TermMap {
    fn new() -> TermMap {
        TermMap {
            term_strings: vec![],
            terms: new_hash_map()
        }
    }

    fn get(&mut self, term: &str) -> TermId {
        // terms.entry().or_insert_with() doesn't work here
        match self.terms.get(term) {
            Some(rt) => return *rt,
            None => {}
        }

        let id = self.term_strings.len();
        self.term_strings.push(term.to_string());
        assert!(id <= u32::max_value() as usize);
        let t = TermId(id as u32);
        self.terms.insert(term.to_string(), t);
        t
    }

    fn id_to_str(&self, id: TermId) -> &str {
        &self.term_strings[id.0 as usize]
    }
}


// --- Stage 3: Build in-memory index -------------------------------------------------------------

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct DocumentId(u32);

struct TermInfo {
    data: Vec<u8>,
    df: u32
}

struct InMemoryIndex {
    word_count: usize,
    map: HashMap<TermId, TermInfo>
}

impl InMemoryIndex {
    fn new() -> InMemoryIndex {
        InMemoryIndex { word_count: 0, map: new_hash_map() }
    }

    fn write(&mut self, term_id: TermId, offset: u32) {
        self.word_count += 1;
        self.map
            .entry(term_id)
            .or_insert_with(|| {
                TermInfo { data: Vec::with_capacity(4), df: 1 }
            })
            .data
            .write_u32::<LittleEndian>(offset)
            .unwrap();
    }

    fn add_document_index(&mut self, document_id: DocumentId, other: InMemoryIndex) {
        for (term_id, mut info) in other.map {
            let target =
                self.map
                .entry(term_id)
                .or_insert_with(|| {
                    TermInfo { data: Vec::with_capacity(4 + info.data.len()), df: 0 }
                });
            // Write hit header: (document id, number of hits in document)
            target.data.write_u32::<LittleEndian>(document_id.0).unwrap();
            target.data.write_u32::<LittleEndian>(info.data.len() as u32 / 4).unwrap();
            target.data.append(&mut info.data);
            target.df += info.df;
        }
        self.word_count += other.word_count;
    }
}

/// Tokenize a text and turn it into an in-memory index.
fn index_text(terms: &mut TermMap, text: String) -> InMemoryIndex {
    let tokens = tokenize(&text);
    let mut little_index = InMemoryIndex::new();
    assert!(tokens.len() <= u32::max_value() as usize);
    for (i, t) in tokens.iter().enumerate() {
        let term_id = terms.get(*t);
        little_index.write(term_id, i as u32);
    }
    little_index
}

fn index_texts(texts: Receiver<String>)
    -> (Receiver<InMemoryIndex>, JoinHandle<TermMap>)
{
    let (sender, receiver) = mpsc::channel();

    let handle = thread::spawn(move || {
        //let mut s = Stopwatch::new('I');
        let mut terms = TermMap::new();
        for text in texts {
            //s.log("received text");
            let index = index_text(&mut terms, text);
            //s.log(format!("indexed {} words", index.word_count));
            sender.send(index).unwrap();
            //s.log("sent");
        }

        //s.log("done");
        terms
    });

    (receiver, handle)
}

fn accumulate_indexes(little_indexes: Receiver<InMemoryIndex>)
    -> (Receiver<InMemoryIndex>, JoinHandle<()>)
{
    let (sender, receiver) = mpsc::channel();

    let handle = thread::spawn(move || {
        const NICE_SIZE: usize = 100_000_000;  // a hundred million words is a nice size

        //let mut s = Stopwatch::new('A');
        let mut big_index = InMemoryIndex::new();
        for (file_number, little_index) in little_indexes.iter().enumerate() {
            //s.log("received little index");
            assert!(file_number <= u32::max_value() as usize);
            let document_id = DocumentId(file_number as u32);

            // Copy the little index into the middle-sized in-memory index.
            big_index.add_document_index(document_id, little_index);
            //s.log("merged into in-memory index");

            // If the middle-sized index is big enough (or we're on the last file) flush to disk.
            if big_index.word_count >= NICE_SIZE {
                sender.send(big_index).unwrap();
                //s.log("sent");
                big_index = InMemoryIndex::new();
            }
        }
        //s.log("no more little indexes");

        if big_index.word_count > 0 {
            sender.send(big_index).unwrap();
        }
        //s.log("done");
    });

    (receiver, handle)
}


// --- Stage 4: Write index files to disk ---------------------------------------------------------

struct TempDir {
    dir: PathBuf,
    n: usize
}

impl TempDir {
    fn new(dir: PathBuf) -> TempDir {
        TempDir {
            dir: dir,
            n: 1
        }
    }

    fn create(&mut self) -> io::Result<(PathBuf, BufWriter<File>)> {
        let mut try = 1;
        loop {
            let filename = self.dir.join(PathBuf::from(format!("tmp{:08x}.dat", self.n)));
            self.n += 1;
            match fs::OpenOptions::new()
                  .write(true)
                  .create_new(true)
                  .open(&filename)
            {
                Ok(f) =>
                    return Ok((filename, BufWriter::new(f))),
                Err(exc) =>
                    if try < 999 && exc.kind() == io::ErrorKind::AlreadyExists {
                        // keep going
                    } else {
                        return Err(exc);
                    }
            }
            try += 1;
        }
        
    }
}

struct TermHeader {
    term_id: TermId,
    nbytes: u32,
    df: u32
}

const TERM_HEADER_NBYTES: usize = 12;

impl TermHeader {
    /// Write a term header to the given file.
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        try!(w.write_u32::<LittleEndian>(self.term_id.0));
        try!(w.write_u32::<LittleEndian>(self.nbytes));
        try!(w.write_u32::<LittleEndian>(self.df));
        Ok(())
    }

    /// Read a term header from the given file. 
    ///
    /// Returns `Ok(None) if `f` is at end-of-file.
    ///
    fn read<R: Read>(f: &mut R) -> io::Result<Option<TermHeader>> {
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
            term_id: TermId(LittleEndian::read_u32(&buf[0..4])),
            nbytes: LittleEndian::read_u32(&buf[4..8]),
            df: LittleEndian::read_u32(&buf[8..12])
        }))
    }
}

fn save_temp_index_file(temp_dir: &mut TempDir, index: InMemoryIndex) -> io::Result<PathBuf> {
    let (filename, mut out) = try!(temp_dir.create());
    let mut v: Vec<(TermId, TermInfo)> = index.map.into_iter().collect();
    v.sort_by_key(|&(id, _)| id);
    for (term_id, term_info) in v {
        let hdr = TermHeader {
            term_id: term_id,
            nbytes: term_info.data.len() as u32,
            df: term_info.df
        };
        try!(hdr.write_to(&mut out));
        try!(out.write_all(&term_info.data));
    }
    println!("wrote file {}", filename.display());
    Ok(filename)
}


// --- Stage 5: Merge index files -----------------------------------------------------------------

/// A `IndexFileReader` does a single linear pass over an index file from
/// beginning to end. Needless to say, this is not how an index is normally
/// used! This is only used when merging multiple index files.
struct IndexFileReader {
    f: BufReader<File>,
    next_header: Option<TermHeader>
}

impl IndexFileReader {
    fn open(filename: &Path) -> io::Result<IndexFileReader> {
        let f = try!(File::open(filename));
        IndexFileReader::new(BufReader::new(f))
    }
    
    fn new(mut f: BufReader<File>) -> io::Result<IndexFileReader> {
        let header = try!(TermHeader::read(&mut f));
        Ok(IndexFileReader { f: f, next_header: header })
    }

    fn has_next(&self) -> bool { self.next_header.is_some() }

    /// Copy the payload of the current term record to the specified output stream,
    /// then read the header for the next term record.
    fn copy_payload_to<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        let nbytes =
            match self.next_header {
                None => panic!("do not try to copy_payload_to if no header"),
                Some(ref hdr) => hdr.nbytes as usize
            };
        let mut buf = Vec::with_capacity(nbytes);
        buf.resize(nbytes, 0);
        try!(self.f.read_exact(&mut buf));
        try!(out.write_all(&buf));
        let header = try!(TermHeader::read(&mut self.f));
        self.next_header = header;
        Ok(())
    }
}

// This kind of index is not the same thing as an "in-memory index" or an
// "index file".  It's an index (small data structure that helps you find what
// you need in a larger data structure) of the index (mapping of terms to
// documents and offsets).
type IndexEntry = (TermId, u32, u64, u64);
struct Index(Vec<IndexEntry>);

impl Index {
    fn new() -> Index { Index(vec![]) }

    fn write(&mut self, term_id: TermId, df: u32, start: u64, nbytes: u64) {
        self.0.push((term_id, df, start, nbytes));
    }
}

fn merge_streams<W: Write>(filenames: &[PathBuf], out: &mut W)
    -> io::Result<Index>
{
    println!("Merging {} streams...", filenames.len());

    let mut streams: Vec<IndexFileReader> =
        try!(filenames.iter().map(|p| IndexFileReader::open(p)).collect());

    let mut index = Index::new();

    let mut point: u64 = 0;
    let mut count = streams.iter().filter(|s| s.has_next()).count();
    while count > 0 {
        let mut term_id = None;
        let mut nbytes = 0;
        let mut df = 0;
        for s in &streams {
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
        let term_id = term_id.expect("bug in algorithm!");

        let entry_header = TermHeader {
            term_id: term_id,
            nbytes: nbytes,
            df: df
        };
        try!(entry_header.write_to(out));
        point += TERM_HEADER_NBYTES as u64;

        for s in &mut streams {
            match s.next_header {
                Some(TermHeader { term_id: s_term_id, .. }) if s_term_id == term_id => {
                    try!(s.copy_payload_to(out));
                    if !s.has_next() {
                        count -= 1;
                    }
                }
                _ => {}
            }
        }
        index.write(term_id, df, point, nbytes as u64);
        point += nbytes as u64;
    }

    assert!(streams.iter().all(|s| !s.has_next()));
    try!(out.flush());

    println!("...done, {} bytes written", point);
    Ok(index)
}


// How many files to merge at a time
const NSTREAMS: usize = 8;

fn push(stacks: &mut Vec<Vec<PathBuf>>,
        mut level: usize,
        mut f: PathBuf,
        temp_dir: &mut TempDir,
        mut index: Index)
    -> io::Result<Index>
{
    loop {
        if level == stacks.len() {
            stacks.push(vec![]);
        }
        stacks[level].push(f);
        if stacks[level].len() < NSTREAMS {
            break;
        }
        let (filename, mut out) = try!(temp_dir.create());
        index = try!(merge_streams(&stacks[level], &mut out));
        stacks[level].clear();
        f = filename;
        level += 1;
    }
    Ok(index)
}

fn merge_reversed(filenames: &mut Vec<PathBuf>, temp_dir: &mut TempDir) -> io::Result<Index> {
    filenames.reverse();
    let (merged_filename, mut out) = try!(temp_dir.create());
    let index = try!(merge_streams(&filenames, &mut out));
    filenames.clear();
    filenames.push(merged_filename);
    Ok(index)
}

fn cleanup(stacks: Vec<Vec<PathBuf>>,
           temp_dir: &mut TempDir,
           last_index: Index)
           -> io::Result<(PathBuf, Index)>
{
    let mut tmp = Vec::with_capacity(NSTREAMS);
    let mut index = last_index;
    for stack in stacks {
        for filename in stack.into_iter().rev() {
            tmp.push(filename);
            if tmp.len() == NSTREAMS {
                index = try!(merge_reversed(&mut tmp, temp_dir));
            }
        }
    }

    if tmp.len() > 1 {
        index = try!(merge_reversed(&mut tmp, temp_dir));
    }
    assert_eq!(tmp.len(), 1);
    Ok((tmp.pop().expect("just asserted"), index))
}

/// NOTE: this should, but does not, destroy files as they are read!
fn merge_many_files(stopwatch: &mut Stopwatch,
                    temp_dir: &mut TempDir,
                    filenames: Vec<PathBuf>,
                    out_filename: &Path)
    -> io::Result<Index>
{
    let mut stacks = vec![vec![]];

    let mut index = Index::new();
    for filename in filenames {
        index = try!(push(&mut stacks, 0, filename, temp_dir, Index::new() /* BUG */));
    }
    stopwatch.log("pushed all files to stacks");
    let (last_file, index) = try!(cleanup(stacks, temp_dir, index));
    stopwatch.log("merged all files into one");

    // On my dev machine, this `fs::rename()` call blocks for about a second.
    // I confirmed that it really is just doing a `rename()` system call.
    // My guess is that the filesystem blocks rename operations until pending
    // writes are flushed to disk; here we're renaming a huge file that we just
    // wrote, so a lot of data is probably buffered in the kernel, waiting for
    // the hardware to work.
    try!(fs::rename(&last_file, out_filename));
    stopwatch.log(format!("renamed {} to {}", last_file.display(), out_filename.display()));
    Ok(index)
}


// --- build_index --------------------------------------------------------------------------------

fn build_index<SD: AsRef<Path>, ID: AsRef<Path>>(source_dir: SD, index_dir: ID)
    -> io::Result<()>
{
    let mut stopwatch = Stopwatch::new(' ');

    let source_dir = source_dir.as_ref();
    let index_dir = index_dir.as_ref();

    let temp_dir = index_dir.join(TMP_DIR);
    try!(fs::create_dir_all(&temp_dir));
    let mut temp_dir = TempDir::new(temp_dir);
    stopwatch.log("startup");

    // Stage 0: Figure out which files we are going to process.
    let documents = try!(list_documents(source_dir, index_dir));
    stopwatch.log("wrote documents file");

    // Stage 1: Read all source files.
    let (files, reader_thread_handle) = read_source_files(source_dir, documents);

    // Stage 2-3a: Tokenize each source file and turn it into a small in-memory index.
    let (small_indexes, terms_thread_handle) = index_texts(files);

    // Stage 3b: Merge the small indexes into bigger indexes that just fit into memory.
    let (big_indexes, acc_thread_handle) = accumulate_indexes(small_indexes);

    // Stage 4: Save the bigger indexes to temporary index files.
    let mut temp_filenames = vec![];
    for big_index in big_indexes {
        stopwatch.log(format!("loaded {} words", big_index.word_count));
        let temp_filename = try!(save_temp_index_file(&mut temp_dir, big_index));
        stopwatch.log(format!("saved to index file {}", temp_filename.display()));
        temp_filenames.push(temp_filename);
    }
    stopwatch.log("done writing index files");

    // Stage 5: Merge all temp files.
    let index = try!(merge_many_files(&mut stopwatch,
                                      &mut temp_dir,
                                      temp_filenames,
                                      &index_dir.join(INDEX_DATA_FILE)));

    // Stage 6: Write the terms file.
    let terms = terms_thread_handle.join().unwrap();
    let terms_filename = index_dir.join(TERMS_FILE);
    let mut terms_file = BufWriter::new(try!(File::create(&terms_filename)));
    for (term_id, df, start, nbytes) in index.0 {
        let term_str = terms.id_to_str(term_id);
        try!(writeln!(terms_file, "{} {} {:x}..{:x}", term_str, df, start, start + nbytes));
    }
    stopwatch.log("wrote terms file");

    try!(reader_thread_handle.join().unwrap());
    acc_thread_handle.join().unwrap();
    stopwatch.log("joined all other threads");

    Ok(())
}

fn main() {
    match build_index(CORPUS_DIR, INDEX_DIR) {
        Err(err) => println!("build_index: {}", err),
        Ok(()) => {}
    }
}
