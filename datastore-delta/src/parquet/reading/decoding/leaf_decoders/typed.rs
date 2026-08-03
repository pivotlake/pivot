//! [`TypedLeafDecoder`] — the generic, type-parameterised implementation of
//! [`LeafDecoder`].
//!
//! This struct is parameterised over the traits that together describe how to
//! decode a particular Parquet column type:
//! - `DC: DictFromBytes` / `DS: DictFromVecBytes` - the two dictionary
//!   flavours; which one is built depends on whether the dictionary page
//!   arrives in one contiguous buffer or scattered across several (see
//!   [`DictStorage`]).
//! - `B: ArrayBuilder` — accumulates decoded values into an Arrow array.
//! - `P: DecodePlain` — reads plain-encoded values from raw page bytes.
//!
//! Concrete column decoders (e.g. `PrimitiveLeafDecoder`, `BytesViewDecoder`)
//! are type aliases over `TypedLeafDecoder` with the appropriate type
//! parameters.

use crate::parquet::reading::decoding::leaf_decoders::levels::decode_def_levels;
use crate::parquet::reading::decoding::leaf_decoders::rle::RleDecoder;
use crate::parquet::reading::decoding::leaf_decoders::{
    ArrayBuilder, DecodeDelta, DecodePlain, Dict, DictFromBytes, DictFromVecBytes, Error,
    LeafDecoder, Result,
};
use crate::parquet::types::filter_mask::RunningFilterMask;
use crate::parquet::types::page::{DataPage, DecompressedPage, DecompressedPageType};
use crate::parquet::types::thrift::general::Encoding;
use crate::parquet::types::thrift::headers::DataPageHeader;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use bytes::Bytes;
use dispatch::arrays::ValidityBuilder;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};
use std::marker::PhantomData;

/// Which encoding strategy to use for a data page's values.
pub enum ValueDecoder<P: DecodePlain> {
    Plain(P),
    Rle(RleDecoder),
    Delta(P::Delta),
}

/// Decode `n` values into `builder` (a dictionary page looks each index up in
/// `dict`). Shared by the required and nullable read loops.
fn decode_run<D, P>(
    decoder: &mut ValueDecoder<P>,
    builder: &mut P::Builder,
    dict: Option<&D>,
    n: usize,
) where
    D: Dict<Builder = P::Builder, Item = <P::Builder as ArrayBuilder>::Element>,
    P: DecodePlain,
{
    match decoder {
        ValueDecoder::Plain(p) => p.read(builder, n),
        ValueDecoder::Rle(r) => r.read(builder, dict.expect("No dict available!"), n),
        ValueDecoder::Delta(d) => d.read(builder, n),
    }
}

/// Advance past `n` values without decoding them (a filtered-out run).
fn skip_run<P: DecodePlain>(decoder: &mut ValueDecoder<P>, n: usize) {
    match decoder {
        ValueDecoder::Plain(p) => p.skip(n),
        ValueDecoder::Rle(r) => r.skip(n),
        ValueDecoder::Delta(d) => d.skip(n),
    }
}

/// A nullable page's definition levels: `present[i]` is whether logical row `i`
/// holds a value (vs a null), consumed run-wise via a cursor.
struct DefLevels {
    present: Vec<bool>,
    pos: usize,
}

impl DefLevels {
    /// The next run of equal present/null rows, capped at `max`.
    fn next_run(&mut self, max: usize) -> (bool, usize) {
        let present = self.present[self.pos];
        let run = self.present[self.pos..]
            .iter()
            .take_while(|&&p| p == present)
            .count()
            .min(max);
        self.pos += run;
        (present, run)
    }
}

/// State for the page currently being read.
///
/// Tracks the value decoder, the optional filter mask cursor, how many rows
/// remain in the page, and, for a nullable column, the page's definition
/// levels (which logical rows are null).
pub struct ReadPage<P: DecodePlain> {
    /// Plain or RLE decoder for this page's values.
    decoder: ValueDecoder<P>,
    /// Filter mask cursor; `None` when the full page is kept.
    running_filter_mask_opt: Option<RunningFilterMask>,
    /// Rows left to decode in this page.
    pub(crate) remaining: usize,
    /// Definition levels for a page with nulls; `None` for a required column or
    /// an all-present page.
    def_levels: Option<DefLevels>,
}

impl<P: DecodePlain> ReadPage<P> {
    /// Decodes up to `size` rows from this page into `builder`.
    ///
    /// When a [`RunningFilterMask`] is present, false runs are skipped and
    /// only true runs are decoded, so the actual number of values pushed may
    /// be less than `size`.
    pub fn read_into<D>(&mut self, dict: Option<&D>, builder: &mut P::Builder, size: usize)
    where
        D: Dict<Builder = P::Builder, Item = <P::Builder as ArrayBuilder>::Element>,
    {
        let current_len = builder.len();
        let mut read_left = size.min(self.remaining);
        while read_left > 0 {
            let (keep, next_run) = match self.running_filter_mask_opt.as_mut() {
                Some(m) => m.next_run(read_left),
                None => (true, read_left),
            };
            if keep {
                decode_run(&mut self.decoder, builder, dict, next_run);
                read_left -= next_run;
            } else {
                skip_run(&mut self.decoder, next_run);
            }
        }
        self.remaining -= builder.len() - current_len;
    }

    /// Like [`read_into`](Self::read_into) but for a page with nulls: each
    /// logical row is decoded as a value (present) or appended as a null, per
    /// the page's definition levels, and a validity bit is pushed for each
    /// kept row. The value stream holds only the present values.
    ///
    /// A filter mask (from a pushed-down predicate) and definition levels are
    /// combined: filter runs partition the rows into kept/skipped, and within
    /// each, definition-level runs partition them into present/null. Skipped
    /// rows still advance the value stream past their present values; only
    /// kept rows reach the builder and validity. `remaining` and `size` count
    /// *kept* rows (`DataPage::rows()` is the kept count when filtered).
    pub fn read_into_nullable<D>(
        &mut self,
        dict: Option<&D>,
        builder: &mut P::Builder,
        validity: &mut ValidityBuilder,
        size: usize,
    ) where
        D: Dict<Builder = P::Builder, Item = <P::Builder as ArrayBuilder>::Element>,
    {
        let start_len = builder.len();
        let mut kept_left = size.min(self.remaining);
        while kept_left > 0 {
            // A filter run covers kept rows (capped at `kept_left`) or a full
            // skipped run; the whole page is kept when there's no mask.
            let (keep, run) = match self.running_filter_mask_opt.as_mut() {
                Some(m) => m.next_run(kept_left),
                None => (true, kept_left),
            };
            let def = self
                .def_levels
                .as_mut()
                .expect("nullable page has def levels");
            let mut run_left = run;
            while run_left > 0 {
                let (present, n) = def.next_run(run_left);
                match (keep, present) {
                    // A kept value decodes; a kept null gets a zeroed slot
                    // that `validity` masks off. The slot must hold a valid
                    // element (zero is an empty inline view / zero primitive):
                    // branchless kernels read masked slots before applying
                    // validity, and byte-view arrays require every view to be
                    // safe to interpret.
                    (true, true) => decode_run(&mut self.decoder, builder, dict, n),
                    (true, false) => {
                        builder
                            .spare_mut(n)
                            .fill(<P::Builder as ArrayBuilder>::Element::default());
                    }
                    // Skipped present rows still consume their values; skipped
                    // nulls have no value in the stream.
                    (false, true) => skip_run(&mut self.decoder, n),
                    (false, false) => {}
                }
                if keep {
                    validity.append_n(n, present);
                }
                run_left -= n;
            }
            if keep {
                kept_left -= run;
            }
        }
        self.remaining -= builder.len() - start_len;
    }
}

/// Tracks whether a page slot holds decodable data or was fully filtered out.
#[allow(clippy::large_enum_variant)]
enum PageSlot {
    Skipped,
    Data(DataPage),
}

/// The dictionary, in whichever representation its page's buffer shape
/// allowed: a page held in one contiguous buffer builds the `Contiguous`
/// flavour, a page scattered across buffers the `Scattered` one.
///
/// The variant is picked once when the dictionary page arrives; readers match
/// it once per call and run monomorphic code from there, so lookups never
/// branch per element.
enum DictStorage<DC, DS> {
    Contiguous(DC),
    Scattered(DS),
}

impl<DC: DictFromBytes, DS: DictFromVecBytes> DictStorage<DC, DS> {
    fn build(mut data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self {
        if data.len() == 1 {
            let bytes = data.pop().expect("data holds one buffer");
            Self::Contiguous(DC::new_from_bytes(bytes, size, allocator))
        } else {
            Self::Scattered(DS::new_from_vec_bytes(data, size, allocator))
        }
    }
}

/// Generic column decoder parameterised by dictionary, builder, and plain
/// decoder types.
///
/// Accumulates [`DecompressedPage`]s and decodes them into Arrow arrays on
/// demand. See the [module docs](self) for how the type parameters fit
/// together.
pub struct TypedLeafDecoder<DC, DS, B, P>
where
    DC: DictFromBytes<Builder = B, Item = B::Element>,
    DS: DictFromVecBytes<Builder = B, Item = B::Element, EqConstant = DC::EqConstant>,
    B: ArrayBuilder,
    P: DecodePlain<Builder = B>,
{
    /// Indexed by page number. `None` means the page hasn't arrived yet.
    pages: Vec<Option<PageSlot>>,
    /// Index of the next page to decode.
    page_idx: usize,
    /// Maximum definition level for this column (0 = non-nullable).
    max_def_level: i16,
    /// The page currently being consumed, if any.
    read_page: Option<ReadPage<P>>,
    /// Dictionary built from a dictionary page, if one has been received.
    dict: Option<DictStorage<DC, DS>>,
    /// The pushed-down equality constant, in this dictionary flavour's own
    /// representation (see [`Dict::EqConstant`]). Drives row-group pruning
    /// and scan-side batch filtering once the dictionary is built.
    eq_const: Option<DC::EqConstant>,
    /// Whether the dictionary was scanned and found to exclude
    /// [`Self::eq_const`]. Stays `false` until a dictionary page proves the
    /// constant absent.
    dict_excludes_eq_constant: bool,
    phantom_data: PhantomData<B>,
}

impl<DC, DS, B, P> TypedLeafDecoder<DC, DS, B, P>
where
    DC: DictFromBytes<Builder = B, Item = B::Element>,
    DS: DictFromVecBytes<Builder = B, Item = B::Element, EqConstant = DC::EqConstant>,
    B: ArrayBuilder,
    P: DecodePlain<Builder = B>,
{
    /// Creates a new decoder for a column with the given maximum definition
    /// level. Use `0` for non-nullable columns.
    pub fn new(max_def_level: i16) -> Self {
        Self {
            pages: Vec::new(),
            page_idx: 0,
            max_def_level,
            read_page: None,
            dict: None,
            eq_const: None,
            dict_excludes_eq_constant: false,
            phantom_data: Default::default(),
        }
    }

    /// Selects the appropriate [`ValueDecoder`] (plain, RLE-dictionary, or
    /// delta) based on the page header's encoding.
    fn create_decoder(
        &self,
        header: DataPageHeader,
        data: Vec<Bytes>,
        mut position: ReaderPosition,
    ) -> Result<ValueDecoder<P>> {
        match header.encoding {
            Encoding::PLAIN => Ok(ValueDecoder::Plain(P::new(data, position))),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                if self.dict.is_none() {
                    return Err(Error::NoPagesReady);
                }
                let bit_width = {
                    let mut reader = MultiBufferReader::new(&data, &mut position);
                    reader.read_u8()
                };
                Ok(ValueDecoder::Rle(RleDecoder::new(
                    data, position, bit_width,
                )))
            }
            // The column type decides which delta encoding it can read, so a
            // page carrying the other one falls through as unsupported.
            e if e == P::Delta::ENCODING => P::Delta::new(data, position)
                .map(ValueDecoder::Delta)
                .ok_or(Error::UnsupportedEncoding(e)),
            r => Err(Error::UnsupportedEncoding(r)),
        }
    }

    /// Advances to the next buffered data page and prepares it for reading.
    ///
    /// Skipped pages are silently stepped over. Returns `Ok(None)` when there
    /// are no more pages to consume.
    fn create_next_read_page(&mut self) -> Result<Option<()>> {
        if self.page_idx >= self.pages.len() {
            return Ok(None);
        }

        let page = loop {
            let page = self.pages[self.page_idx]
                .take()
                .ok_or(Error::NoPagesReady)?;
            if let PageSlot::Data(d) = page {
                break d;
            }
            self.page_idx += 1;
        };

        let mut position = ReaderPosition::default();
        // Definition levels cover every logical row in the page, but
        // `page.rows()` is the *kept* count once a filter mask is applied: so
        // decode `num_values` of them, while `remaining` tracks kept rows.
        let kept = page.rows();
        let num_values = page.header.num_values as usize;

        // A nullable column's pages prefix the values with definition levels;
        // decode them (advancing `position` past them). An all-present page
        // yields `None` and decodes by the fast, non-null path below.
        let def_levels = if self.max_def_level > 0 {
            let mut reader = MultiBufferReader::new(&page.data, &mut position);
            let def_level_byte_len = reader.read_u32_le() as usize;
            decode_def_levels(
                &mut reader,
                num_values,
                def_level_byte_len,
                self.max_def_level,
            )
            .map(|present| DefLevels { present, pos: 0 })
        } else {
            None
        };

        self.read_page = Some(ReadPage {
            decoder: self.create_decoder(page.header, page.data, position)?,
            running_filter_mask_opt: page.filter_mask.map(RunningFilterMask::new),
            remaining: kept,
            def_levels,
        });

        Ok(Some(()))
    }
}

impl<DC, DS, B, P> LeafDecoder for TypedLeafDecoder<DC, DS, B, P>
where
    DC: DictFromBytes<Builder = B, Item = B::Element>,
    DS: DictFromVecBytes<Builder = B, Item = B::Element, EqConstant = DC::EqConstant>,
    B: ArrayBuilder,
    P: DecodePlain<Builder = B>,
    B::Element: PartialEq,
{
    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>) {
        self.eq_const = DC::eq_constant_from_scalar(value);
    }

    fn dict_excludes_eq_constant(&self) -> bool {
        self.dict_excludes_eq_constant
    }

    fn fast_filter_record_batch(&self, batch: RecordBatch, column: usize) -> RecordBatch {
        match (&self.eq_const, &self.dict) {
            (Some(needle), Some(DictStorage::Contiguous(dict))) => {
                dict.filter_record_batch_by_const(batch, column, needle)
            }
            (Some(needle), Some(DictStorage::Scattered(dict))) => {
                dict.filter_record_batch_by_const(batch, column, needle)
            }
            _ => batch,
        }
    }

    fn available(&self) -> usize {
        let (mut available, mut page_idx) = if let Some(s) = &self.read_page {
            (s.remaining, self.page_idx + 1)
        } else {
            (0, self.page_idx)
        };

        while page_idx < self.pages.len()
            && let Some(p) = &self.pages[page_idx]
        {
            available += match p {
                PageSlot::Data(d) => {
                    if self.dict.is_none()
                        && matches!(
                            d.header.encoding,
                            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                        )
                    {
                        return available;
                    }
                    d.rows()
                }
                PageSlot::Skipped => 0,
            };
            page_idx += 1;
        }
        available
    }

    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        match page.data {
            DecompressedPageType::Dict { header, data } => {
                let size = header.num_values as usize;
                if let Some(needle) = self.eq_const.as_ref() {
                    // Equality pushdown: scan the raw dictionary for the constant
                    // before materializing it. If absent, the row group is pruned
                    // — so skip building the dictionary entirely (no allocation,
                    // no copy of values we'd never read).
                    let present = DC::maybe_contains(&data, size, needle);
                    self.dict_excludes_eq_constant = !present;
                    if !present {
                        // Row group will be pruned. Install an *empty* dictionary
                        // instead of the real one: this skips the copy but keeps
                        // the invariant that a dict-encoded column has
                        // `dict.is_some()`, so `available()` and worker scheduling
                        // behave exactly as on the build path. (Leaving `dict`
                        // `None` makes `available()` report 0, parking every
                        // worker before the prune completes → lost-wakeup hang.)
                        // The dictionary is never read — the row group is pruned.
                        self.dict = Some(DictStorage::build(data, 0, allocator));
                        return;
                    }
                }
                self.dict = Some(DictStorage::build(data, size, allocator));
            }
            DecompressedPageType::Data(data) => {
                let idx = page.idx;
                if idx >= self.pages.len() {
                    self.pages.resize_with(idx + 1, || None);
                }
                self.pages[idx] = Some(PageSlot::Data(data));
            }
            DecompressedPageType::SkippedData { .. } => {
                let idx = page.idx;
                if idx >= self.pages.len() {
                    self.pages.resize_with(idx + 1, || None);
                }
                self.pages[idx] = Some(PageSlot::Skipped);
            }
        }
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let mut builder = B::with_capacity(allocator, size);
        match &self.dict {
            Some(DictStorage::Contiguous(d)) => d.register_onto(&mut builder),
            Some(DictStorage::Scattered(d)) => d.register_onto(&mut builder),
            None => {}
        }
        // Validity is built lazily on slab memory only once a page actually has
        // a null. A required column, and a nullable one whose pages are all
        // present (e.g. a variant's `value` leaf), never allocates a bitmap and
        // takes the same fast path as before.
        let mut validity: Option<ValidityBuilder> = None;

        while builder.len() < size {
            if self.read_page.is_none() {
                match self.create_next_read_page()? {
                    Some(()) => {}
                    None => break,
                }
            }

            let remaining = size - builder.len();
            let read_page = self.read_page.as_mut().unwrap();
            if read_page.def_levels.is_some() {
                // This page has nulls: ensure a bitmap exists (backfilling the
                // rows decoded so far as present), then scatter values and nulls.
                let v = validity.get_or_insert_with(|| {
                    let mut vb = ValidityBuilder::with_capacity(allocator, size);
                    vb.append_n(builder.len(), true);
                    vb
                });
                match &self.dict {
                    Some(DictStorage::Contiguous(d)) => {
                        read_page.read_into_nullable(Some(d), &mut builder, v, remaining)
                    }
                    Some(DictStorage::Scattered(d)) => {
                        read_page.read_into_nullable(Some(d), &mut builder, v, remaining)
                    }
                    None => read_page.read_into_nullable(None::<&DC>, &mut builder, v, remaining),
                }
            } else {
                let before = builder.len();
                match &self.dict {
                    Some(DictStorage::Contiguous(d)) => {
                        read_page.read_into(Some(d), &mut builder, remaining)
                    }
                    Some(DictStorage::Scattered(d)) => {
                        read_page.read_into(Some(d), &mut builder, remaining)
                    }
                    None => read_page.read_into(None::<&DC>, &mut builder, remaining),
                }
                if let Some(v) = &mut validity {
                    v.append_n(builder.len() - before, true);
                }
            }

            if read_page.remaining == 0 {
                self.read_page = None;
                self.page_idx += 1;
            }
        }

        let null_buffer = validity.map(|v| v.into_buffer());
        Ok(builder.into_array(null_buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::reading::decoding::leaf_decoders::primitive::PrimitiveLeafDecoder;
    use crate::parquet::test_utils::dummy_metadata;
    use crate::parquet::types::filter_mask::FilterMask;
    use crate::parquet::types::page::{DataPage, DecompressedPageType};
    use crate::parquet::types::thrift::general::Encoding;
    use crate::parquet::types::thrift::headers::PageHeader;
    use arrow_array::Int32Array;
    use arrow_array::types::Int32Type;
    use dispatch::memory::{SlabAllocator, init_test_free_pool};

    // -- Helpers --

    fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn int_scalar(value: i32) -> Scalar<ArrayRef> {
        Scalar::new(std::sync::Arc::new(Int32Array::from(vec![value])) as ArrayRef)
    }

    fn extract_i32s(arr: &ArrayRef) -> Vec<i32> {
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn data_page(data: Vec<u8>, num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn filtered_data_page(
        data: Vec<u8>,
        num_values: usize,
        idx: usize,
        mask: FilterMask,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: Some(mask),
            }),
        }
    }

    fn skipped_page(num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx,
            data: DecompressedPageType::SkippedData {
                header: header.data_page_header.unwrap(),
            },
        }
    }

    fn dict_page(data: Vec<u8>, num_values: usize) -> DecompressedPage {
        let header = PageHeader::for_dict_page(num_values as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(data)],
            },
        }
    }

    type Dec = PrimitiveLeafDecoder<Int32Type>;

    // -- available --

    /// No pages inserted → nothing available.
    #[test]
    fn test_available_empty() {
        let dec = Dec::new(0);

        assert_eq!(dec.available(), 0);
    }

    /// Single page with 3 values → 3 available.
    #[test]
    fn test_available_single_page() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0), &mut alloc);

        assert_eq!(dec.available(), 3);
    }

    /// Two pages sum their rows for availability.
    #[test]
    fn test_available_two_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[3, 4, 5]), 3, 1), &mut alloc);

        assert_eq!(dec.available(), 5);
    }

    /// Skipped pages contribute 0 rows to availability.
    #[test]
    fn test_available_skipped_page_contributes_zero() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(10, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 1), &mut alloc);

        assert_eq!(dec.available(), 2);
    }

    /// RLE-dict page without a dictionary → 0 available.
    #[test]
    fn test_available_dict_page_missing() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x00, 0x00])],
                filter_mask: None,
            }),
        };
        dec.insert_page(page, &mut alloc);

        assert_eq!(dec.available(), 0);
    }

    // -- Dictionary pruning (pushed-down equality constant) --

    /// No constant set → never prunes.
    #[test]
    fn test_dict_excludes_false_without_constant() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert!(!dec.dict_excludes_eq_constant());
    }

    /// Constant set but dictionary not yet loaded → not prunable yet.
    #[test]
    fn test_dict_excludes_false_before_dict_loaded() {
        let mut dec = Dec::new(0);
        dec.set_eq_constant(&int_scalar(20));

        assert!(!dec.dict_excludes_eq_constant());
    }

    /// Constant present in the dictionary → cannot prune.
    #[test]
    fn test_dict_excludes_false_when_present() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(&int_scalar(20));
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert!(!dec.dict_excludes_eq_constant());
    }

    /// Constant absent from the dictionary → the row group is prunable.
    #[test]
    fn test_dict_excludes_true_when_absent() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(&int_scalar(99));
        dec.insert_page(dict_page(encode_i32s(&[10, 20, 30]), 3), &mut alloc);

        assert!(dec.dict_excludes_eq_constant());
    }

    // -- Page insertion order --

    /// Pages inserted out of order are read in index order.
    #[test]
    fn test_out_of_order_insertion() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[30, 40]), 2, 1), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 0), &mut alloc);

        let result = dec.read(&mut alloc, 4).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20, 30, 40]);
    }

    // -- Skipped pages --

    /// A skipped page is stepped over; read produces values from the next data page.
    #[test]
    fn test_skipped_page_stepped_over() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 1), &mut alloc);

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20]);
    }

    /// Only skipped pages → no rows available.
    #[test]
    fn test_only_skipped_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0), &mut alloc);

        assert_eq!(dec.available(), 0);
    }

    // -- Incremental reads --

    /// Reading fewer rows than available leaves the rest for the next read.
    #[test]
    fn test_incremental_read_within_page() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(
            data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0),
            &mut alloc,
        );

        let r1 = dec.read(&mut alloc, 2).unwrap();
        let r2 = dec.read(&mut alloc, 3).unwrap();

        assert_eq!(extract_i32s(&r1), vec![10, 20]);
        assert_eq!(extract_i32s(&r2), vec![30, 40, 50]);
    }

    /// Incremental reads spanning multiple pages.
    #[test]
    fn test_incremental_read_across_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0), &mut alloc);
        dec.insert_page(data_page(encode_i32s(&[4, 5]), 2, 1), &mut alloc);

        let r1 = dec.read(&mut alloc, 4).unwrap();
        let r2 = dec.read(&mut alloc, 1).unwrap();

        assert_eq!(extract_i32s(&r1), vec![1, 2, 3, 4]);
        assert_eq!(extract_i32s(&r2), vec![5]);
    }

    // -- Filter mask --

    /// Filtered page keeps only the matching rows.
    #[test]
    fn test_filter_mask_keeps_matching_rows() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[1, 3]);
        dec.insert_page(
            filtered_data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, mask),
            &mut alloc,
        );

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![20, 40]);
    }

    /// All-false filter mask → 0 rows available.
    #[test]
    fn test_filter_mask_all_false() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[]);
        dec.insert_page(
            filtered_data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, mask),
            &mut alloc,
        );

        assert_eq!(dec.available(), 0);
    }

    // -- Dict-encoded --

    /// Dict page followed by RLE data page decodes correctly.
    #[test]
    fn test_dict_then_data() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(dict_page(encode_i32s(&[100, 200, 300]), 3), &mut alloc);

        // bit_width=2, 1 group of 8, indices [0,1,2,0,0,1,2,0]
        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page, &mut alloc);

        let result = dec.read(&mut alloc, 8).unwrap();

        assert_eq!(
            extract_i32s(&result),
            vec![100, 200, 300, 100, 100, 200, 300, 100]
        );
    }

    /// available() returns 0 until dict page arrives, then reports the full count.
    #[test]
    fn test_dict_page_unblocks_availability() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(None),
            column_idx: 0,
            idx: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page, &mut alloc);
        assert_eq!(dec.available(), 0);

        dec.insert_page(dict_page(encode_i32s(&[100, 200, 300]), 3), &mut alloc);

        assert_eq!(dec.available(), 8);
    }
}
