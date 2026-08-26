//! The union of the layouts a compaction's input files gave a variant column:
//! the one layout their batches are all widened to before they are gathered
//! together.
//!
//! The union keeps every path any input typed, so that no typed value has to
//! move before the output layout is decided. Where inputs typed one path as
//! different kinds, it can keep only one: the kind that holds the most rows
//! across the inputs, counted from the footers' null counts. The rows of the
//! other kinds fold back into their leftover when their batch is widened.
//!
//! Nothing here decides what the output files type. That takes the rows, and
//! happens once they are collected ([`super::infer`]).

use std::collections::BTreeMap;

use arrow_schema::{DataType, Field, Fields};

/// One input's layout of a variant column: the column's Arrow type as its
/// file's footer describes it, and how many rows each of the column's leaves
/// holds, in the footer's leaf order.
pub(super) struct InputLayout<'a> {
    pub(super) column: &'a DataType,
    pub(super) rows_per_leaf: &'a [u64],
}

/// The union of `inputs`' layouts as a `typed_value` type, or `None` when no
/// input typed anything.
pub(super) fn union_layout(inputs: &[InputLayout<'_>]) -> Option<DataType> {
    let mut root = Votes::default();
    for input in inputs {
        let DataType::Struct(fields) = input.column else {
            continue;
        };
        let mut leaf = 0;
        for field in fields {
            if field.name() == "typed_value" {
                root.count(field.data_type(), input.rows_per_leaf, &mut leaf);
            } else {
                leaf += leaf_count(field.data_type());
            }
        }
    }
    resolve_fields(&root.fields)
}

/// The rows an input's leaves under one position hold, by the kind they hold
/// them as.
#[derive(Default)]
struct Votes {
    scalars: BTreeMap<DataType, u64>,
    fields: BTreeMap<String, Votes>,
}

impl Votes {
    /// Count the leaves under a `typed_value` of type `typed`, whose first
    /// leaf is `leaf`.
    fn count(&mut self, typed: &DataType, rows_per_leaf: &[u64], leaf: &mut usize) {
        let DataType::Struct(children) = typed else {
            if is_shreddable(typed) {
                *self.scalars.entry(typed.clone()).or_default() += rows_per_leaf[*leaf];
            }
            *leaf += 1;
            return;
        };
        for child in children {
            let DataType::Struct(pair) = child.data_type() else {
                *leaf += 1;
                continue;
            };
            for part in pair {
                if part.name() == "typed_value" {
                    self.fields.entry(child.name().clone()).or_default().count(
                        part.data_type(),
                        rows_per_leaf,
                        leaf,
                    );
                } else {
                    *leaf += leaf_count(part.data_type());
                }
            }
        }
    }

    /// The rows this position holds as an object: it has no leaf of its own,
    /// so the most any child holds stands in for it.
    fn object_rows(&self) -> u64 {
        self.fields
            .values()
            .map(|child| child.rows())
            .max()
            .unwrap_or(0)
    }

    fn rows(&self) -> u64 {
        self.scalars
            .values()
            .copied()
            .max()
            .unwrap_or(0)
            .max(self.object_rows())
    }

    /// The kind this position keeps: the scalar type with the most rows, unless
    /// more rows held an object. Ties break the way inference breaks them, by
    /// the type's own order, so the union and the output layout agree.
    fn resolve(&self) -> Option<DataType> {
        let scalar = self
            .scalars
            .iter()
            .map(|(data_type, &rows)| (rows, data_type))
            .max();
        match scalar {
            Some((rows, data_type)) if rows >= self.object_rows() => Some(data_type.clone()),
            _ => resolve_fields(&self.fields),
        }
    }
}

fn resolve_fields(fields: &BTreeMap<String, Votes>) -> Option<DataType> {
    let resolved: Vec<Field> = fields
        .iter()
        .filter_map(|(name, child)| Some(Field::new(name, child.resolve()?, true)))
        .collect();
    (!resolved.is_empty()).then(|| DataType::Struct(Fields::from(resolved)))
}

/// The scalar types the shredder writes. A leaf of any other type came from
/// another writer, and its position keeps only what this writer can rebuild.
fn is_shreddable(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int64 | DataType::Float64 | DataType::Utf8View
    )
}

fn leaf_count(data_type: &DataType) -> usize {
    match data_type {
        DataType::Struct(children) => children
            .iter()
            .map(|child| leaf_count(child.data_type()))
            .sum(),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::super::shred::{column_fields, typed_value_type};
    use super::*;

    fn object(fields: Vec<(&str, DataType)>) -> DataType {
        DataType::Struct(
            fields
                .into_iter()
                .map(|(name, data_type)| Field::new(name, data_type, true))
                .collect(),
        )
    }

    /// A file's column type for `shredding`, and the rows per leaf: `rows`
    /// for every typed leaf and zero for every fallback, so a layout's votes
    /// are all its rows.
    fn input(shredding: Option<&DataType>, rows: u64) -> (DataType, Vec<u64>) {
        let column = DataType::Struct(column_fields(shredding.map(typed_value_type).as_ref()));
        let mut rows_per_leaf = Vec::new();
        push_rows(&column, rows, &mut rows_per_leaf);
        (column, rows_per_leaf)
    }

    fn push_rows(data_type: &DataType, rows: u64, out: &mut Vec<u64>) {
        match data_type {
            DataType::Struct(fields) => {
                for field in fields {
                    match field.data_type() {
                        DataType::Struct(_) => push_rows(field.data_type(), rows, out),
                        _ => out.push(if field.name() == "typed_value" {
                            rows
                        } else {
                            0
                        }),
                    }
                }
            }
            _ => out.push(rows),
        }
    }

    fn union_of(inputs: &[(DataType, Vec<u64>)]) -> Option<DataType> {
        let layouts: Vec<InputLayout<'_>> = inputs
            .iter()
            .map(|(column, rows_per_leaf)| InputLayout {
                column,
                rows_per_leaf,
            })
            .collect();
        union_layout(&layouts)
    }

    #[test]
    fn keeps_every_path_any_input_typed() {
        let a = input(Some(&object(vec![("id", DataType::Int64)])), 10);
        let b = input(
            Some(&object(vec![
                ("n", DataType::Float64),
                ("user", object(vec![("name", DataType::Utf8View)])),
            ])),
            3,
        );

        let union = union_of(&[a, b]);

        assert_eq!(
            union,
            Some(object(vec![
                ("id", DataType::Int64),
                ("n", DataType::Float64),
                ("user", object(vec![("name", DataType::Utf8View)])),
            ]))
        );
    }

    #[test]
    fn a_path_typed_as_two_types_keeps_the_one_with_more_rows() {
        let as_int = input(Some(&object(vec![("x", DataType::Int64)])), 5);
        let as_text = input(Some(&object(vec![("x", DataType::Utf8View)])), 6);

        assert_eq!(
            union_of(&[as_int.clone(), as_text.clone()]),
            Some(object(vec![("x", DataType::Utf8View)]))
        );
        assert_eq!(
            union_of(&[
                as_int,
                input(Some(&object(vec![("x", DataType::Utf8View)])), 5)
            ]),
            Some(object(vec![("x", DataType::Utf8View)])),
            "a tie breaks by the type's order, as inference breaks it"
        );
    }

    #[test]
    fn a_path_typed_as_an_object_and_a_scalar_keeps_the_one_with_more_rows() {
        let as_object = input(
            Some(&object(vec![("p", object(vec![("q", DataType::Int64)]))])),
            8,
        );
        let as_scalar = input(Some(&object(vec![("p", DataType::Int64)])), 2);

        assert_eq!(
            union_of(&[as_object, as_scalar]),
            Some(object(vec![("p", object(vec![("q", DataType::Int64)]))]))
        );
    }

    #[test]
    fn unshredded_inputs_and_foreign_leaf_types_add_nothing() {
        let plain = input(None, 100);
        let foreign = input(Some(&object(vec![("flag", DataType::Boolean)])), 100);
        let ours = input(Some(&object(vec![("id", DataType::Int64)])), 1);

        assert_eq!(union_of(&[plain.clone(), foreign.clone()]), None);
        assert_eq!(
            union_of(&[plain, foreign, ours]),
            Some(object(vec![("id", DataType::Int64)]))
        );
    }
}
