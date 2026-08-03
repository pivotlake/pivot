//! [`FileBytes`]: bytes on their way out of the engine, held on the ring.
//!
//! A file the write path produces is megabytes of encoded pages, and the ring
//! hands out memory in [`BUFFER_SIZE`](super::BUFFER_SIZE) slabs, so those bytes
//! cannot be one allocation. They do not need to be: nothing reads a file back,
//! it is only written out in order. So a file is kept as the runs it was
//! produced in, each one a slab, and the writer walks them.
//!
//! That is what lets the assembler stop copying. A page is written into a slab
//! once by the encoder, and every stage after that moves the slab rather than
//! its bytes.
//!
//! The two questions the I/O layer asks are the reason this is a type rather
//! than a bare `Vec<Slab>`: how many bytes there are in total, and which bytes
//! come next after `n` of them have been written. The second one is subtle,
//! because a write can complete short in the middle of a run, so resuming means
//! finding the run holding that byte and continuing from partway into it.

use crate::memory::slab::Slab;

/// The bytes of one output file, as the runs they were produced in.
///
/// Every run is ring memory, so the last owner has to drop on a dispatch
/// worker: releasing a slab reaches the worker's memory context.
#[derive(Default)]
pub struct FileBytes {
    runs: Vec<Slab>,
    len: usize,
}

impl FileBytes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `run` to the file. Its whole region counts, so a caller that
    /// allocated a slab larger than it filled must not pass it here.
    pub fn push(&mut self, run: Slab) {
        self.len += run.size();
        self.runs.push(run);
    }

    /// Append every run of `other`, in order.
    pub fn append(&mut self, other: FileBytes) {
        self.len += other.len;
        self.runs.extend(other.runs);
    }

    /// Total bytes across every run, which is the size of the file.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The runs in order. Concatenated, they are the file.
    pub fn runs(&self) -> impl Iterator<Item = &[u8]> {
        self.runs.iter().map(Slab::as_slice)
    }

    /// The bytes to write next once `offset` of them have been written: the
    /// rest of the run `offset` falls in, or an empty slice once the whole file
    /// has been written.
    ///
    /// A writer advances by what it managed to write and asks again, so this
    /// resumes a short write from exactly where it stopped, whether it stopped
    /// on a run boundary or inside one.
    pub fn run_at(&self, offset: usize) -> &[u8] {
        assert!(
            offset <= self.len,
            "offset {offset} past a {} byte file",
            self.len
        );
        let mut start = 0;
        for run in &self.runs {
            let bytes = run.as_slice();
            if offset < start + bytes.len() {
                return &bytes[offset - start..];
            }
            start += bytes.len();
        }
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SlabAllocator, has_memory_context, init_test_free_pool};

    /// Everything a writer would put on the wire for `file`, in order.
    fn written(file: &FileBytes) -> Vec<u8> {
        file.runs().flatten().copied().collect()
    }

    /// A file of one run per slice, with the allocator that owns their memory.
    /// The pool is installed once per test thread, since a test may build more
    /// than one file and a second context is refused.
    fn file(parts: &[&[u8]]) -> (FileBytes, SlabAllocator) {
        if !has_memory_context() {
            init_test_free_pool(8);
        }
        let mut allocator = SlabAllocator::new(false);
        let mut file = FileBytes::new();
        for part in parts {
            let mut run = allocator.get_slab_of_size(part.len(), false);
            run.as_mut_slice().copy_from_slice(part);
            file.push(run);
        }
        (file, allocator)
    }

    #[test]
    fn the_runs_concatenate_to_the_file() {
        let (file, _allocator) = file(&[b"PAR1", b"a page", b"another page"]);

        assert_eq!(file.len(), 22);
        assert_eq!(written(&file), b"PAR1a pageanother page");
    }

    /// Writing resumes from wherever it stopped, including partway into a run,
    /// which is what a short write leaves behind.
    #[test]
    fn a_write_resumes_from_any_offset() {
        let (file, _allocator) = file(&[b"abcd", b"efgh"]);

        assert_eq!(file.run_at(0), b"abcd");
        assert_eq!(file.run_at(2), b"cd");
        assert_eq!(file.run_at(4), b"efgh");
        assert_eq!(file.run_at(7), b"h");
        assert_eq!(file.run_at(8), b"");
    }

    #[test]
    fn appending_a_file_keeps_the_order_of_its_runs() {
        let (mut first, _allocator) = file(&[b"one"]);
        let (second, _second_allocator) = file(&[b"two", b"three"]);

        first.append(second);

        assert_eq!(written(&first), b"onetwothree");
        assert_eq!(first.len(), 11);
    }
}
