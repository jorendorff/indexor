extern crate byteorder;
extern crate rayon;

use std::io;
use std::io::SeekFrom;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::sync::Mutex;
use rayon::prelude::*;

const DIR: &'static str = "./playpen/sample";
const DOCUMENTS_FILE: &'static str = "./playpen/documents.txt";
const TERMS_FILE: &'static str = "./playpen/terms.txt";
const INDEX_DATA_FILE: &'static str = "./playpen/index.dat";

const BYTES_PER_WORD: usize = 4;

struct Term {
    df: usize,
    start: u64,
    nbytes: u64,
    nbytes_written: u64
}

impl Term {
    fn new() -> Term {
        Term {
            df: 0,
            start: 0,
            nbytes: 0,
            nbytes_written: 0
        }
    }

    fn add(&mut self, nhits: usize) {
        self.df += 1;
        self.nbytes += Self::entry_size(nhits) as u64;
    }

    fn entry_size(nhits: usize) -> usize {
        (2 + nhits) * BYTES_PER_WORD
    }

    fn prepare_write(&mut self, doc_id: usize, offsets: &[usize]) -> io::Result<(u64, Vec<u8>)> {
        let write_offset = self.start + self.nbytes_written;
        let nhits = offsets.len();
        let mut tmp: Vec<u8> = Vec::with_capacity(Self::entry_size(nhits));

        {
            let mut write_u32 = |size| {
                assert!(size <= std::u32::MAX as usize);
                tmp.write_u32::<LittleEndian>(size as u32)
            };

            try!(write_u32(doc_id));
            try!(write_u32(nhits));
            for i in offsets {
                try!(write_u32(*i));
            }
        }
        assert_eq!(tmp.len(), Self::entry_size(nhits));

        if self.nbytes_written + tmp.len() as u64 > self.nbytes {
            return Err(io::Error::new(std::io::ErrorKind::InvalidData,
                                      "index record full (probably a source file \
                                       changed on disk during indexing)"));
        }
        self.nbytes_written += tmp.len() as u64;
        Ok((write_offset, tmp))
    }
}

fn read_file_lowercase(filename: &Path) -> io::Result<String> {
    let mut f = try!(File::open(filename));
    let mut text = String::new();
    try!(f.read_to_string(&mut text));
    Ok(text.to_lowercase())
}

fn tokenize(text: &str) -> Vec<&str> {
    text
        .split(|c: char| c > '\x7f' || !(c.is_alphanumeric() || c == '/' || c == '\\'))
        .filter(|s: &&str| !s.is_empty())
        .collect()
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

fn make_index() -> io::Result<()> {
    use std::time::Instant;
    let t00 = Instant::now();

    let dir_path = PathBuf::from(DIR);
    let documents_mutex = Mutex::new(vec![]);
    let terms_mutex: Mutex<HashMap<String, Term>> = Mutex::new(HashMap::new());

    let t01;
    {
        let entries = try!(fs::read_dir(&dir_path)).collect::<Vec<_>>();

        t01 = Instant::now();
        println!("Scanned directory in {:?}", t01 - t00);

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

                //println!("{}", filename);
                {
                    let mut documents = try!(unpoison(documents_mutex.lock()));
                    documents.push(filename);
                }

                let text = try!(read_file_lowercase(&entry.path()));
                let tokens = tokenize(&text);
                let mut counter: HashMap<&str, usize> = HashMap::new();
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
    }

    let t02 = Instant::now();
    println!("loaded in-memory index in {:?}", t02 - t01);

    let documents = try!(unpoison(documents_mutex.into_inner()));
    {
        let mut documents_file = try!(File::create(DOCUMENTS_FILE));
        for filename in &documents {
            try!(writeln!(documents_file, "{}", filename));
        }
    }

    let t1 = Instant::now();
    println!("wrote documents file in {:?}", t1 - t02);

    // The terms file is a sort of index into the index. First we record in
    // memory all the data we want to save in the terms file...
    let mut terms_output_records = vec![];
    {
        let mut point = 0;
        let mut terms = try!(unpoison(terms_mutex.lock()));
        for (term_str, mut term_md) in &mut *terms {
            term_md.start = point;
            point += term_md.nbytes;
            terms_output_records.push((term_str.to_string(), term_md.df, term_md.start, point));
            //try!(writeln!(terms_file, "{} {} {:x}..{:x}", term_str, term_md.df, term_md.start, point));
        }
    }

    let t1_1 = Instant::now();
    println!("computed terms file in {:?}", t1_1 - t1);

    // ...Then we send that data to a separate thread to be written to disk.
    let terms_file_writer_thread: std::thread::JoinHandle<io::Result<()>> = std::thread::spawn(move || {
        let mut terms_file = try!(File::create(TERMS_FILE));
        for (term_str, df, start, stop) in terms_output_records {
            try!(writeln!(terms_file, "{} {} {:x}..{:x}", term_str, df, start, stop));
        }
        Ok(())
    });

    let t2 = Instant::now();
    println!("launched terms thread in {:?}", t2 - t1_1);

    let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<(u64, Vec<u8>)>>(10);
    let index_data_writer_thread = std::thread::spawn(move || -> io::Result<()> {
        let mut index_data_file = try!(File::create(INDEX_DATA_FILE));
        for writes in receiver.into_iter() {
            for (write_offset, data) in writes {
                try!(index_data_file.seek(SeekFrom::Start(write_offset)));
                try!(index_data_file.write_all(&data));
            }
        }
        Ok(())
    });

    let sender_mutex = Mutex::new(sender);

    let t2_1 = Instant::now();
    println!("launched index data file writer thread in {:?}", t2_1 - t2);

    println!("writing index data...");
    {
        let result = documents
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .weight_max()
            .map(|&(document_id, filename)| -> io::Result<()> {
                //println!("{}", filename);
                let text = try!(read_file_lowercase(&dir_path.join(filename)));
                let tokens = tokenize(&text);
                let mut offsets_by_term: HashMap<&str, Vec<usize>> = HashMap::new();
                for (i, term_str) in tokens.into_iter().enumerate() {
                    offsets_by_term.entry(term_str).or_insert_with(Vec::new).push(i);
                }

                let mut writes = vec![];
                {
                    let mut terms = try!(unpoison(terms_mutex.lock()));
                    for (&term_str, offsets) in &offsets_by_term {
                        match terms.get_mut(term_str) {
                            Some(term_md) => {
                                writes.push(try!(term_md.prepare_write(document_id, offsets)));
                            }
                            None => {
                                println!("*** term {:?} not found (file changed on disk \
                                          during index building, most likely)", term_str);
                            }
                        }
                    }
                }

                // Ignore an error sending here: it means there was a problem
                // on the receiving end, and that will be reported separately.
                let _ = try!(unpoison(sender_mutex.lock())).send(writes);

                // let mut index_data_file = try!(unpoison(index_data_file_mutex.lock()));
                // for (write_offset, data) in writes {
                //     try!(index_data_file.seek(SeekFrom::Start(write_offset)));
                //     try!(index_data_file.write_all(&data));
                // }
                Ok(())
            })
            .reduce(&TakeFirstError);
        try!(result);
    }

    let t2_2 = Instant::now();
    println!("generated all bytes for data file in {:?}", t2_2 - t2_1);

    drop(sender_mutex);  // hang up, so the writer thread knows it's done
    try!(index_data_writer_thread.join().unwrap());
    let t3 = Instant::now();
    println!("finished writing data file in {:?}", t3 - t2_2);

    {
        let terms = try!(unpoison(terms_mutex.into_inner()));
        for (term_str, term_md) in terms {
            if term_md.nbytes_written != term_md.nbytes {
                println!("*** term {:?}: expected {} bytes, wrote {} bytes",
                         term_str, term_md.nbytes, term_md.nbytes_written);
            }
        }
    }

    let t4 = Instant::now();
    println!("finished assertions and dropping the in-memory index in {:?}", t4 - t3);

    try!(terms_file_writer_thread.join().unwrap());
    let t5 = Instant::now();
    println!("joined terms thread in {:?}", t5 - t4);

    Ok(())
}

fn main() {
    match make_index() {
        Err(err) => println!("build_index: {}", err),
        Ok(()) => {}
    }
}
