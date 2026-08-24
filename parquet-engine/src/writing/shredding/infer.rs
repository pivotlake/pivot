//! Chooses which paths inside a variant column are worth storing as typed
//! Parquet leaves.
//!
//! Shredding is an optimization, never a constraint: a row whose value at a
//! shredded path is missing, or is of another type, simply keeps that value in
//! the binary `value` fallback beside the typed leaf. So the choice here can
//! only be better or worse, never wrong — which is what lets each file decide
//! independently from its own rows.
//!
//! The rule is deliberately a simple one: a path earns a typed leaf when enough
//! rows carry it ([`MIN_PRESENCE`]) as one dominant type. The bar for "enough"
//! is low, because a leaf costs about what it holds, so the question is less
//! whether the rows agree on a path than whether it is really there at all.
//! Everything else stays in `value`, where it costs nothing beyond the bytes it
//! already occupied.
//!
//! The sampled documents are counted into a [`Candidate`] tree shaped like the
//! documents, which is also the shape of the answer — a `typed_value` type is a
//! tree of fields. So nothing here builds or splits a path string: the field
//! names are the tree's edges.

use std::collections::BTreeMap;

use arrow_schema::{DataType, Field, Fields};
use parquet_variant::Variant;
use parquet_variant_compute::VariantArray;

/// Rows read to decide a file's shredding schema. The decision only has to be
/// representative, and walking every row of a large file would cost more than
/// the shredding saves — so sample a bounded number, spread across the file
/// rather than taken from the head, where one unusual leading batch would
/// otherwise drive the whole layout.
const SAMPLE_ROWS: usize = 4096;

/// How far into a nested object to look for paths. Far enough to be beside the
/// point for real documents, but not unbounded, because the reader sets a
/// ceiling this has to live under.
///
/// A shredded path costs two schema levels per segment (the field's group, then
/// its `typed_value`), and the reader refuses a footer nesting deeper than 128
/// levels rather than risk overflowing the stack parsing it. Shredding past
/// about 62 segments would therefore write files that our own reader will not
/// load at all — and a file that doesn't load is far worse than a path that
/// stayed in the binary `value` — so this keeps a wide margin under that.
const MAX_DEPTH: usize = 60;

/// The share of sampled rows that must carry a path, as a shreddable type,
/// before it earns a typed leaf.
///
/// Low, because a rare leaf is close to free: the rows without the path spend
/// only a definition level, a run of which RLE-encodes to a few bytes, and the
/// typed leaf stores just the values that are there. So a path on a twentieth of
/// the rows still pays for itself on a query that reads it. The bar is here to
/// drop the genuinely one-off key, not to ask that the rows agree — which also
/// makes it the only thing bounding how many leaves a column spends, so a column
/// of many moderately-common paths shreds all of them.
const MIN_PRESENCE: f64 = 0.05;

/// The Arrow type `value` shreds into, or `None` for a value the writer has no
/// Parquet leaf type for (booleans, timestamps, decimals, and the nested
/// values) — those stay in the binary `value`.
///
/// The set is deliberately narrow, and every integer width unifies to `Int64`
/// (every float to `Float64`), so a path holding a mix of widths still shreds
/// into one leaf instead of splitting its rows across the fallback.
fn shred_data_type(value: &Variant) -> Option<DataType> {
    match value {
        Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_) => {
            Some(DataType::Int64)
        }
        Variant::Float(_) | Variant::Double(_) => Some(DataType::Float64),
        Variant::String(_) | Variant::ShortString(_) => Some(DataType::Utf8View),
        _ => None,
    }
}

/// One position inside the sampled documents: what turned up there, and — where
/// it held an object — the same again for each of its fields.
///
/// Shaped like the documents themselves, so counting a row is a walk down the
/// tree rather than a lookup by path: no path is ever assembled, a field name is
/// allocated only the first time that field is seen (not once per row), and the
/// finished tree converts straight into the `typed_value` type, which is a tree
/// too.
///
/// The maps are ordered, so a file's layout is a function of its rows alone and
/// two writers given the same rows agree.
#[derive(Default)]
struct Candidate {
    /// How often this position held a scalar of each shreddable type.
    scalars: BTreeMap<DataType, usize>,
    /// How often it held an object worth descending into.
    objects: usize,
    /// The fields seen under it, when it did.
    fields: BTreeMap<String, Candidate>,
}

/// What a position is stored as. Parquet gives it one type, but the rows need
/// not agree — `{"a": 1}` and `{"a": {"b": 2}}` in one file leave `a` both a
/// scalar and an object — so the shape it held most often wins and the rest of
/// the rows fall back to the binary `value`.
enum Shape<'a> {
    /// Stored as a typed leaf of this type, which this many rows had.
    Scalar(DataType, usize),
    /// Stored as a group of these fields' own leaves.
    Object(&'a BTreeMap<String, Candidate>),
}

impl Candidate {
    /// Count everything shreddable in `value`, which sits at `depth` (the
    /// document itself is at 0).
    ///
    /// Only objects are descended into: a bare scalar column has no paths to
    /// name, and a list's elements have no stable path either.
    fn observe(&mut self, value: &Variant, depth: usize) {
        if let Some(data_type) = shred_data_type(value) {
            *self.scalars.entry(data_type).or_default() += 1;
            return;
        }
        let Some(object) = value.as_object() else {
            return;
        };
        if depth == MAX_DEPTH {
            return;
        }
        self.objects += 1;
        for (name, field) in object.iter() {
            // `entry` would allocate the name on every row; only a field never
            // seen before needs one.
            if !self.fields.contains_key(name) {
                self.fields.insert(name.to_string(), Candidate::default());
            }
            let child = self
                .fields
                .get_mut(name)
                .expect("the field was present or just inserted");
            child.observe(&field, depth + 1);
        }
    }

    /// How this position is stored, or `None` when it held nothing shreddable.
    fn shape(&self) -> Option<Shape<'_>> {
        // The most common scalar type here, ties broken by type so the choice
        // doesn't depend on iteration order.
        let scalar = self
            .scalars
            .iter()
            .map(|(data_type, &count)| (count, data_type))
            .max();
        match scalar {
            Some((count, data_type)) if count >= self.objects => {
                Some(Shape::Scalar(data_type.clone(), count))
            }
            _ if self.objects > 0 => Some(Shape::Object(&self.fields)),
            _ => None,
        }
    }

    /// The type to shred this position into, or `None` when it earned no leaf.
    fn resolve_data_type(&self, min_count: usize) -> Option<DataType> {
        match self.shape()? {
            Shape::Scalar(data_type, count) => (count >= min_count).then_some(data_type),
            Shape::Object(fields) => resolve_fields(fields, min_count).map(DataType::Struct),
        }
    }
}

/// The struct type for `fields`, or `None` when none of them earned a leaf — an
/// empty group is not worth a level of nesting.
fn resolve_fields(fields: &BTreeMap<String, Candidate>, min_count: usize) -> Option<Fields> {
    let resolved: Vec<Field> = fields
        .iter()
        .filter_map(|(name, child)| {
            Some(Field::new(name, child.resolve_data_type(min_count)?, true))
        })
        .collect();
    (!resolved.is_empty()).then(|| resolved.into())
}

/// The `typed_value` type to shred `array` into, or `None` when no path is worth
/// it (the column then stays unshredded, which is a legal variant column).
pub(super) fn infer_shredding_type(arrays: &[VariantArray]) -> Option<DataType> {
    let mut root = Candidate::default();
    let mut sampled = 0usize;
    let rows: usize = arrays.iter().map(|array| array.len()).sum();
    for row in sample_rows(rows) {
        let (array, row) = locate(arrays, row);
        if array.is_null(row) {
            continue;
        }
        sampled += 1;
        root.observe(&array.value(row), 0);
    }
    if sampled == 0 {
        return None;
    }

    // A document is shredded by its fields and never as a whole, so `root`'s own
    // shape is the one never consulted: its fields are resolved directly, where
    // every node below them goes through `resolve_data_type`. A bare scalar
    // document has no fields, so a column of them stays unshredded.
    let min_count = (sampled as f64 * MIN_PRESENCE).ceil() as usize;
    Some(DataType::Struct(resolve_fields(&root.fields, min_count)?))
}

/// The rows to sample: every row of a small column, else [`SAMPLE_ROWS`] spread
/// evenly across the whole of a large one.
fn sample_rows(rows: usize) -> impl Iterator<Item = usize> {
    let stride = rows.div_ceil(SAMPLE_ROWS).max(1);
    (0..rows).step_by(stride)
}

/// The array holding row `row` of the file, and that row's index within it. A
/// file arrives as its row groups, and the sample spans the file, so a sampled
/// row has to be placed back in the group it came from.
fn locate(arrays: &[VariantArray], row: usize) -> (&VariantArray, usize) {
    let mut row = row;
    for array in arrays {
        if row < array.len() {
            return (array, row);
        }
        row -= array.len();
    }
    unreachable!("a sampled row is one of the file's rows")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, StringArray};
    use arrow_schema::{Field, Fields};
    use parquet_variant_compute::json_to_variant;
    use std::sync::Arc;

    /// A variant column built from one JSON document per row.
    fn variants(rows: &[&str]) -> VariantArray {
        let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
        json_to_variant(&json).unwrap()
    }

    /// The `typed_value` fields the inference chose, as `(name, type)`.
    fn shredded_fields(data_type: &DataType) -> Vec<(String, DataType)> {
        let DataType::Struct(fields) = data_type else {
            panic!("expected a struct shredding type, got {data_type:?}");
        };
        fields
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect()
    }

    /// Fields every row agrees on get a typed leaf of that type.
    #[test]
    fn shreds_the_fields_every_row_shares() {
        let array = variants(&[
            r#"{"id": 1, "name": "a"}"#,
            r#"{"id": 2, "name": "b"}"#,
            r#"{"id": 3, "name": "c"}"#,
        ]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![
                ("id".to_string(), DataType::Int64),
                ("name".to_string(), DataType::Utf8View),
            ]
        );
    }

    /// A field too few rows carry isn't worth a leaf. The bar is low, so it takes
    /// a genuinely rare field to fall under it: `rare` is on 2 rows of 100, well
    /// below the 5 that [`MIN_PRESENCE`] asks for.
    #[test]
    fn skips_a_field_too_few_rows_carry() {
        let mut rows = vec![r#"{"id": 1, "rare": 9}"#.to_string(); 2];
        rows.resize(100, r#"{"id": 1}"#.to_string());
        let array = variants(&rows.iter().map(String::as_str).collect::<Vec<_>>());

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("id".to_string(), DataType::Int64)]
        );
    }

    /// A field a twentieth of the rows carry does earn one, though: the leaf only
    /// costs what it holds, and the rows without it only cost a definition level.
    #[test]
    fn shreds_a_field_only_a_twentieth_of_rows_carry() {
        let mut rows = vec![r#"{"id": 1, "occasional": 9}"#.to_string(); 5];
        rows.resize(100, r#"{"id": 1}"#.to_string());
        let array = variants(&rows.iter().map(String::as_str).collect::<Vec<_>>());

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![
                ("id".to_string(), DataType::Int64),
                ("occasional".to_string(), DataType::Int64),
            ]
        );
    }

    /// Integer widths unify, so a field mixing them still shreds into one leaf.
    #[test]
    fn unifies_integer_widths_into_one_leaf() {
        let array = variants(&[r#"{"n": 1}"#, r#"{"n": 40000000000}"#, r#"{"n": 3}"#]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("n".to_string(), DataType::Int64)]
        );
    }

    /// A field whose rows disagree on the type shreds as the dominant one; the
    /// rest of the rows fall back to the binary `value` at read time.
    #[test]
    fn a_mixed_type_field_shreds_as_its_dominant_type() {
        let array = variants(&[
            r#"{"v": "a"}"#,
            r#"{"v": "b"}"#,
            r#"{"v": "c"}"#,
            r#"{"v": 4}"#,
        ]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("v".to_string(), DataType::Utf8View)]
        );
    }

    /// A field that is a scalar on some rows and an object on others can only be
    /// stored as one of them: the shape most rows had wins, and the others fall
    /// back to the binary `value`. Here the object does.
    #[test]
    fn a_field_that_is_mostly_an_object_shreds_as_one() {
        let array = variants(&[r#"{"a": {"b": 1}}"#, r#"{"a": {"b": 2}}"#, r#"{"a": 3}"#]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        let nested = DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int64, true)]));
        assert_eq!(shredded_fields(&inferred), vec![("a".to_string(), nested)]);
    }

    /// The same conflict the other way round: the scalar had more rows, so the
    /// field is a typed leaf and the object-valued row falls back.
    #[test]
    fn a_field_that_is_mostly_a_scalar_shreds_as_one() {
        let array = variants(&[r#"{"a": 1}"#, r#"{"a": 2}"#, r#"{"a": {"b": 3}}"#]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("a".to_string(), DataType::Int64)]
        );
    }

    /// The deepest field path in a shredding type, in segments.
    fn max_path_depth(data_type: &DataType) -> usize {
        match data_type {
            DataType::Struct(fields) => {
                1 + fields
                    .iter()
                    .map(|field| max_path_depth(field.data_type()))
                    .max()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// Nesting stops at [`MAX_DEPTH`] however deep the documents go, which is
    /// what keeps the schema a file gets asked to write under the 128 levels the
    /// reader will parse (a path costs two of them per segment).
    #[test]
    fn stops_descending_at_the_depth_limit() {
        // Nested well past the limit, with a shreddable scalar at every level so
        // that each one has something to keep.
        let deep = (0..MAX_DEPTH + 5).fold(r#"{"v": 1}"#.to_string(), |inner, _| {
            format!(r#"{{"v": 1, "n": {inner}}}"#)
        });
        let array = variants(&[&deep, &deep]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(max_path_depth(&inferred), MAX_DEPTH);
        assert!(2 * max_path_depth(&inferred) + 2 < 128);
    }

    /// A nested object's own fields are shredded, addressed by their full path.
    #[test]
    fn shreds_a_nested_path() {
        let array = variants(&[r#"{"user": {"id": 1}}"#, r#"{"user": {"id": 2}}"#]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        let expected =
            DataType::Struct(Fields::from(vec![Field::new("id", DataType::Int64, true)]));
        assert_eq!(
            shredded_fields(&inferred),
            vec![("user".to_string(), expected)]
        );
    }

    /// Rows with no shreddable field at all leave the column unshredded rather
    /// than inventing an empty typed_value.
    #[test]
    fn infers_nothing_when_no_path_is_worth_it() {
        let array = variants(&[r#"{"ok": true}"#, r#"{"ok": false}"#]);

        assert!(infer_shredding_type(std::slice::from_ref(&array)).is_none());
    }

    /// Documents that are bare scalars rather than objects have no fields to
    /// shred, so the column stays unshredded instead of becoming one typed leaf.
    #[test]
    fn infers_nothing_for_documents_that_are_bare_scalars() {
        let array = variants(&["1", "2", "3"]);

        assert!(infer_shredding_type(std::slice::from_ref(&array)).is_none());
    }

    /// A key containing a dot is one field, not a nested path — splitting it
    /// would shred something that isn't there.
    #[test]
    fn treats_a_dotted_key_as_one_field() {
        let array = variants(&[r#"{"a.b": 1}"#, r#"{"a.b": 2}"#]);

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("a.b".to_string(), DataType::Int64)]
        );
    }

    /// An object with many one-off keys (one keyed by request id, say) is what
    /// would blow a file's width up, and nothing caps the leaf count outright.
    /// It doesn't need one even at a bar this low: a key on one row in a hundred
    /// is far under it, so only the shared field earns a leaf.
    #[test]
    fn a_wide_one_off_object_shreds_only_its_shared_field() {
        let wide: String = (0..64).map(|i| format!(r#""k{i}": {i},"#)).collect();
        let mut rows = vec![format!(r#"{{{wide} "keep": 1}}"#)];
        rows.resize(100, r#"{"keep": 2}"#.to_string());
        let array = variants(&rows.iter().map(String::as_str).collect::<Vec<_>>());

        let inferred = infer_shredding_type(std::slice::from_ref(&array)).unwrap();

        assert_eq!(
            shredded_fields(&inferred),
            vec![("keep".to_string(), DataType::Int64)]
        );
    }

    /// The paths chosen depend on the rows alone, not on hash iteration order.
    #[test]
    fn infers_the_same_schema_twice() {
        let rows = [
            r#"{"a": 1, "b": "x", "c": 1.5}"#,
            r#"{"a": 2, "b": "y", "c": 2.5}"#,
        ];

        let first = infer_shredding_type(&[variants(&rows)]).unwrap();
        let second = infer_shredding_type(&[variants(&rows)]).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            shredded_fields(&first),
            vec![
                ("a".to_string(), DataType::Int64),
                ("b".to_string(), DataType::Utf8View),
                ("c".to_string(), DataType::Float64),
            ]
        );
    }
}
