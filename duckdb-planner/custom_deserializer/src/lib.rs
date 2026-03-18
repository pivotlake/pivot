use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Expr, Fields, parse_macro_input};

/// Derive `Deserialize` for enums and structs.
///
/// We need this custom derive because serde's built-in `#[derive(Deserialize)]`
/// only supports matching enum discriminants against literal values (strings or
/// integers written inline). DuckDB's JSON plan uses numeric type tags whose
/// values are defined as constants on C++ enums (e.g.
/// `LogicalOperatorType::LOGICAL_GET`), and serde has no way to reference an
/// external constant or expression as a discriminant. This macro bridges that
/// gap: each variant is annotated with `#[type_tag(...)]` pointing to the
/// DuckDB enum constant, and the generated `Deserialize` impl matches the
/// `"type"` field against that constant's numeric value at compile time.
///
/// **Enums** must be serialized as `{"type": <u8>, "data": {...}}`.
/// Each variant must be a single-element tuple variant annotated with
/// `#[type_tag(...)]` pointing to a DuckDB enum constant.
///
/// **Structs** are deserialized from a JSON object by field name.
/// No `#[type_tag(...)]` is needed for structs.
///
/// Fields (enum variants or struct fields) can optionally specify
/// `#[custom_deserialize(fn_name)]` to use a custom function instead of
/// `serde_json::from_value`. The function must have the signature
/// `fn(serde_json::Value) -> Result<T, String>`.
///
/// ```ignore
/// #[derive(CustomDeserializer)]
/// enum PlanNode {
///     #[type_tag(LogicalOperatorType::LOGICAL_GET)]
///     Input(Input),
///     #[type_tag(LogicalOperatorType::LOGICAL_PROJECTION)]
///     Projection(Projection),
///     #[type_tag(ExpressionType::VALUE_CONSTANT)]
///     Constant(ScalarValue),
/// }
///
/// #[derive(CustomDeserializer)]
/// struct Input {
///     table_name: String,
///     column_names: Vec<String>,
/// }
/// ```
#[proc_macro_derive(CustomDeserializer, attributes(type_tag, custom_deserialize, skip_deserialize))]
pub fn derive_deserialize_typed(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match &input.data {
        Data::Enum(_) => derive_enum(&input),
        Data::Struct(_) => derive_struct(&input),
        _ => panic!("CustomDeserializer can only be derived for enums and structs"),
    }
}

fn derive_enum(input: &DeriveInput) -> TokenStream {
    let enum_name = &input.ident;

    let variants = match &input.data {
        Data::Enum(e) => &e.variants,
        _ => unreachable!(),
    };

    let mut arms = Vec::new();
    for variant in variants {
        // Skip variants marked with #[skip_deserialize] — they are never
        // produced by deserialization and may contain types that don't
        // implement Deserialize.
        let is_skipped = variant.attrs.iter().any(|attr| attr.path().is_ident("skip_deserialize"));
        if is_skipped {
            continue;
        }

        let variant_name = &variant.ident;

        let inner_ty = match &variant.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => &f.unnamed[0].ty,
            _ => panic!("each variant must be a single-element tuple variant, e.g. Foo(Bar)"),
        };

        let disc: Option<Expr> = variant.attrs.iter().find_map(|attr| {
            if !attr.path().is_ident("type_tag") {
                return None;
            }
            attr.parse_args::<Expr>().ok()
        });

        let custom_deser: Option<Expr> = variant.attrs.iter().find_map(|attr| {
            if !attr.path().is_ident("custom_deserialize") {
                return None;
            }
            attr.parse_args::<Expr>().ok()
        });

        let deser_expr = match custom_deser {
            Some(func) => quote! {
                #func(data).map_err(serde::de::Error::custom)?
            },
            None => quote! {
                serde_json::from_value::<#inner_ty>(data).map_err(serde::de::Error::custom)?
            },
        };

        let condition = match disc {
            Some(d) => quote! {
                serde_json::Value::Number(ref n) if n.as_u64() == Some(#d as u64)
            },
            None => {
                let name_lower = variant_name.to_string().to_lowercase();
                quote! {
                    serde_json::Value::String(ref s) if s == #name_lower
                }
            }
        };

        arms.push(quote! {
            #condition => {
                Ok(#enum_name::#variant_name(#deser_expr))
            }
        });
    }

    let enum_name_str = enum_name.to_string();
    let expecting_msg = format!("a {} object with `type` and `data` fields", enum_name_str);
    let err_msg = format!("unsupported {} type: {{}}", enum_name_str);

    let expanded = quote! {
        impl<'de> serde::Deserialize<'de> for #enum_name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct __Visitor;

                impl<'de> serde::de::Visitor<'de> for __Visitor {
                    type Value = #enum_name;

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str(#expecting_msg)
                    }

                    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<#enum_name, A::Error> {
                        let mut op_type: Option<serde_json::Value> = None;
                        let mut data: Option<serde_json::Value> = None;

                        while let Some(key) = map.next_key::<String>()? {
                            match key.as_str() {
                                "type" => op_type = Some(map.next_value()?),
                                "data" => data = Some(map.next_value()?),
                                _ => { let _ = map.next_value::<serde_json::Value>()?; }
                            }
                        }

                        let op_type = op_type.ok_or_else(|| serde::de::Error::missing_field("type"))?;
                        let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;

                        match op_type {
                            #(#arms)*
                            _ => Err(serde::de::Error::custom(format!(#err_msg, op_type))),
                        }
                    }
                }

                deserializer.deserialize_map(__Visitor)
            }
        }
    };

    expanded.into()
}

fn derive_struct(input: &DeriveInput) -> TokenStream {
    let struct_name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("CustomDeserializer only supports structs with named fields"),
        },
        _ => unreachable!(),
    };

    let field_vars: Vec<_> = fields
        .iter()
        .map(|f| {
            let name = f.ident.as_ref().unwrap();
            let name_str = name.to_string();

            let skip_expr: Option<Option<Expr>> = f.attrs.iter().find_map(|attr| {
                if !attr.path().is_ident("skip_deserialize") {
                    return None;
                }
                // #[skip_deserialize] uses Default::default()
                // #[skip_deserialize(expr)] uses the given expression
                Some(attr.parse_args::<Expr>().ok())
            });

            if let Some(initializer) = skip_expr {
                return match initializer {
                    Some(expr) => quote! { let #name = #expr; },
                    None => quote! { let #name = Default::default(); },
                };
            }

            let custom_deser: Option<Expr> = f.attrs.iter().find_map(|attr| {
                if !attr.path().is_ident("custom_deserialize") {
                    return None;
                }
                attr.parse_args::<Expr>().ok()
            });

            match custom_deser {
                Some(func) => quote! {
                    let #name = {
                        let val = obj.remove(#name_str).unwrap_or(serde_json::Value::Null);
                        #func(val).map_err(serde::de::Error::custom)?
                    };
                },
                None => quote! {
                    let #name = {
                        let val = obj.remove(#name_str).unwrap_or(serde_json::Value::Null);
                        serde_json::from_value(val).map_err(serde::de::Error::custom)?
                    };
                },
            }
        })
        .collect();

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let struct_name_str = struct_name.to_string();

    let expanded = quote! {
        impl<'de> serde::Deserialize<'de> for #struct_name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)
                    .map_err(serde::de::Error::custom)?;
                let mut obj = match value {
                    serde_json::Value::Object(map) => map,
                    _ => return Err(serde::de::Error::custom(
                        format!("expected object for {}. Received: {}", #struct_name_str, value)
                    )),
                };

                #(#field_vars)*

                Ok(#struct_name {
                    #(#field_names),*
                })
            }
        }
    };

    expanded.into()
}
