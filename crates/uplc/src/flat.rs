use crate::{
    ast::{
        Constant, Data, DeBruijn, FakeNamedDeBruijn, Name, NamedDeBruijn, Program, Term, Type,
        Unique,
    },
    builtins::DefaultFunction,
    machine::runtime::Compressable,
    protocol::MAX_CONSTANT_TYPE_SIZE,
};
use num_bigint::BigInt;
use pallas_codec::flat::{
    Flat,
    de::{self, Decode, Decoder},
    en::{self, Encode, Encoder},
};
use pallas_primitives::Fragment;
use std::{collections::VecDeque, fmt::Debug, rc::Rc};

const BUILTIN_TAG_WIDTH: u32 = 7;
const CONST_TAG_WIDTH: u32 = 4;
const TERM_TAG_WIDTH: u32 = 4;

pub trait Binder<'b>: Encode + Decode<'b> {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error>;
    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error>;
    fn text(&self) -> String;
}

impl<'b, T> Flat<'b> for Program<T>
where
    T: Binder<'b> + Debug,
{
    /// Decodes a program and its filler, rejecting anything after them as
    /// Plutus' flat decoder does (`TooMuchSpace`).
    fn unflat(bytes: &'b [u8]) -> Result<Self, de::Error> {
        let mut d = Decoder::new(bytes);

        let program = d.decode()?;

        d.filler()?;

        // A filler ending inside a byte leaves `pos` on that byte.
        if d.pos < bytes.len() {
            return Err(de::Error::Message(format!(
                "{} unexpected bytes after the program",
                bytes.len() - d.pos
            )));
        }

        Ok(program)
    }
}

impl<'b, T> Program<T>
where
    T: Binder<'b> + Debug,
{
    pub fn from_cbor(bytes: &'b [u8], buffer: &'b mut Vec<u8>) -> Result<Self, de::Error> {
        let mut cbor_decoder = pallas_codec::minicbor::Decoder::new(bytes);

        let flat_bytes = cbor_decoder
            .bytes()
            .map_err(|err| de::Error::Message(err.to_string()))?;

        buffer.extend(flat_bytes);

        Self::unflat(buffer)
    }

    pub fn from_flat(bytes: &'b [u8]) -> Result<Self, de::Error> {
        Self::unflat(bytes)
    }

    pub fn from_hex(
        hex_str: &str,
        cbor_buffer: &'b mut Vec<u8>,
        flat_buffer: &'b mut Vec<u8>,
    ) -> Result<Self, de::Error> {
        let cbor_bytes = hex::decode(hex_str).map_err(|err| de::Error::Message(err.to_string()))?;

        cbor_buffer.extend(cbor_bytes);

        Self::from_cbor(cbor_buffer, flat_buffer)
    }

    /// Convert a program to cbor bytes.
    ///
    /// _note: The cbor bytes of a program are merely
    /// the flat bytes of the program encoded as cbor bytes._
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program.to_debruijn().unwrap().to_cbor().unwrap(),
    ///     vec![
    ///         0x46, 0x01, 0x00, 0x00,
    ///         0x20, 0x01, 0x01
    ///     ],
    /// );
    /// ```
    pub fn to_cbor(&self) -> Result<Vec<u8>, en::Error> {
        let flat_bytes = self.flat()?;

        let mut bytes = Vec::new();

        let mut cbor_encoder = pallas_codec::minicbor::Encoder::new(&mut bytes);

        cbor_encoder
            .bytes(&flat_bytes)
            .map_err(|err| en::Error::Message(err.to_string()))?;

        Ok(bytes)
    }

    /// Convert a program to a flat bytes.
    ///
    /// _**note**: Convenient so that people don't need to depend on the flat crate
    /// directly to call programs flat function._
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program
    ///         .to_debruijn()
    ///         .unwrap()
    ///         .to_flat()
    ///         .unwrap(),
    ///     vec![
    ///         0x01, 0x00, 0x00,
    ///         0x20, 0x01, 0x01
    ///     ],
    /// );
    /// ```
    pub fn to_flat(&self) -> Result<Vec<u8>, en::Error> {
        self.flat()
    }

    /// Convert a program to hex encoded cbor bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc::ast::{Program, Name, Term};
    ///
    /// let term = Term::var("x").lambda("x");
    /// let program = Program { version: (1, 0, 0), term };
    ///
    /// assert_eq!(
    ///     program.to_debruijn().unwrap().to_hex().unwrap(),
    ///     "46010000200101".to_string(),
    /// );
    /// ```
    pub fn to_hex(&self) -> Result<String, en::Error> {
        let bytes = self.to_cbor()?;

        let hex = hex::encode(bytes);

        Ok(hex)
    }
}

impl<'b, T> Encode for Program<T>
where
    T: Binder<'b> + Debug,
{
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        let (major, minor, patch) = self.version;

        major.encode(e)?;
        minor.encode(e)?;
        patch.encode(e)?;

        self.term.encode(e)?;

        Ok(())
    }
}

impl<'b, T> Decode<'b> for Program<T>
where
    T: Binder<'b>,
{
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let mut state_log: Vec<String> = vec![];
        let version = (decode_usize(d)?, decode_usize(d)?, decode_usize(d)?);
        let term_option = Term::decode_debug(d, &mut state_log);

        match term_option {
            Ok(term) => {
                let program = Program { version, term };
                program
                    .check_version_features()
                    .map_err(de::Error::Message)?;
                Ok(program)
            }
            Err(error) => Err(de::Error::Message(format!(
                "{} {error}",
                state_log.join("")
            ))),
        }
    }
}

impl<'b, T> Encode for Term<T>
where
    T: Binder<'b> + Debug,
{
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        match self {
            Term::Var(name) => {
                encode_term_tag(0, e)?;
                name.encode(e)?;
            }
            Term::Delay(term) => {
                encode_term_tag(1, e)?;
                term.encode(e)?;
            }
            Term::Lambda {
                parameter_name,
                body,
            } => {
                encode_term_tag(2, e)?;
                parameter_name.binder_encode(e)?;
                body.encode(e)?;
            }
            Term::Apply { function, argument } => {
                encode_term_tag(3, e)?;
                function.encode(e)?;
                argument.encode(e)?;
            }

            Term::Constant(constant) => {
                encode_term_tag(4, e)?;
                constant.encode(e)?;
            }

            Term::Force(term) => {
                encode_term_tag(5, e)?;
                term.encode(e)?;
            }

            Term::Error => {
                encode_term_tag(6, e)?;
            }
            Term::Builtin(builtin) => {
                encode_term_tag(7, e)?;

                builtin.encode(e)?;
            }
            Term::Constr { tag, fields } => {
                encode_term_tag(8, e)?;

                encode_word64(*tag, e);

                e.encode_list_with(fields, |term, e| (*term).encode(e))?;
            }
            Term::Case { constr, branches } => {
                encode_term_tag(9, e)?;

                constr.encode(e)?;

                e.encode_list_with(branches, |term, e| (*term).encode(e))?;
            }
        }

        Ok(())
    }
}

impl<'b, T> Decode<'b> for Term<T>
where
    T: Binder<'b>,
{
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        match decode_term_tag(d)? {
            0 => Ok(Term::Var(T::decode(d)?.into())),
            1 => Ok(Term::Delay(Rc::new(Term::decode(d)?))),
            2 => Ok(Term::Lambda {
                parameter_name: T::binder_decode(d)?.into(),
                body: Rc::new(Term::decode(d)?),
            }),
            3 => Ok(Term::Apply {
                function: Rc::new(Term::decode(d)?),
                argument: Rc::new(Term::decode(d)?),
            }),
            // Need size limit for Constant
            4 => Ok(Term::Constant(Constant::decode(d)?.into())),
            5 => Ok(Term::Force(Rc::new(Term::decode(d)?))),
            6 => Ok(Term::Error),
            7 => Ok(Term::Builtin(DefaultFunction::decode(d)?)),
            8 => {
                let tag = decode_word64(d)?;
                let fields = d.decode_list_with(Term::<T>::decode)?;

                Ok(Term::Constr { tag, fields })
            }
            9 => {
                let constr = (Term::<T>::decode(d)?).into();

                let branches = d.decode_list_with(Term::<T>::decode)?;

                Ok(Term::Case { constr, branches })
            }
            x => {
                let buffer_slice: Vec<u8> = d
                    .buffer
                    .to_vec()
                    .iter()
                    .skip(d.pos.saturating_sub(5))
                    .take(10)
                    .cloned()
                    .collect();

                Err(de::Error::UnknownTermConstructor(
                    x,
                    if d.pos > 5 { 5 } else { d.pos },
                    format!("{buffer_slice:02X?}"),
                    d.pos,
                    d.buffer.len(),
                ))
            }
        }
    }
}

impl<'b, T> Term<T>
where
    T: Binder<'b>,
{
    fn decode_debug(d: &mut Decoder, state_log: &mut Vec<String>) -> Result<Term<T>, de::Error> {
        match decode_term_tag(d)? {
            0 => {
                state_log.push("(var ".to_string());
                let var_option = T::decode(d);
                match var_option {
                    Ok(var) => {
                        state_log.push(format!("{})", var.text()));
                        Ok(Term::Var(var.into()))
                    }
                    Err(error) => {
                        state_log.push("parse error)".to_string());
                        Err(error)
                    }
                }
            }

            1 => {
                state_log.push("(delay ".to_string());
                let term_option = Term::decode_debug(d, state_log);
                match term_option {
                    Ok(term) => {
                        state_log.push(")".to_string());
                        Ok(Term::Delay(Rc::new(term)))
                    }
                    Err(error) => {
                        state_log.push(")".to_string());
                        Err(error)
                    }
                }
            }
            2 => {
                state_log.push("(lam ".to_string());

                let var_option = T::binder_decode(d);
                match var_option {
                    Ok(var) => {
                        state_log.push(var.text());
                        let term_option = Term::decode_debug(d, state_log);
                        match term_option {
                            Ok(term) => {
                                state_log.push(")".to_string());
                                Ok(Term::Lambda {
                                    parameter_name: var.into(),
                                    body: Rc::new(term),
                                })
                            }
                            Err(error) => {
                                state_log.push(")".to_string());
                                Err(error)
                            }
                        }
                    }
                    Err(error) => {
                        state_log.push(")".to_string());
                        Err(error)
                    }
                }
            }
            3 => {
                state_log.push("[ ".to_string());

                let function_term_option = Term::decode_debug(d, state_log);
                match function_term_option {
                    Ok(function) => {
                        state_log.push(" ".to_string());
                        let arg_term_option = Term::decode_debug(d, state_log);
                        match arg_term_option {
                            Ok(argument) => {
                                state_log.push("]".to_string());
                                Ok(Term::Apply {
                                    function: Rc::new(function),
                                    argument: Rc::new(argument),
                                })
                            }
                            Err(error) => {
                                state_log.push("]".to_string());
                                Err(error)
                            }
                        }
                    }
                    Err(error) => {
                        state_log.push(" not parsed]".to_string());
                        Err(error)
                    }
                }
            }
            // Need size limit for Constant
            4 => {
                state_log.push("(con ".to_string());

                let con_option = Constant::decode(d);
                match con_option {
                    Ok(constant) => {
                        state_log.push(format!("{})", constant.to_pretty()));
                        Ok(Term::Constant(constant.into()))
                    }
                    Err(error) => {
                        state_log.push("parse error)".to_string());
                        Err(error)
                    }
                }
            }
            5 => {
                state_log.push("(force ".to_string());
                let term_option = Term::decode_debug(d, state_log);
                match term_option {
                    Ok(term) => {
                        state_log.push(")".to_string());
                        Ok(Term::Force(Rc::new(term)))
                    }
                    Err(error) => {
                        state_log.push(")".to_string());
                        Err(error)
                    }
                }
            }
            6 => {
                state_log.push("(error)".to_string());
                Ok(Term::Error)
            }
            7 => {
                state_log.push("(builtin ".to_string());

                let builtin_option = DefaultFunction::decode(d);
                match builtin_option {
                    Ok(builtin) => {
                        state_log.push(format!("{builtin})"));
                        Ok(Term::Builtin(builtin))
                    }
                    Err(error) => {
                        state_log.push("parse error)".to_string());
                        Err(error)
                    }
                }
            }
            8 => {
                state_log.push("(constr ".to_string());

                let tag = decode_word64(d)?;

                let fields = d.decode_list_with_debug(
                    |d, state_log| Term::<T>::decode_debug(d, state_log),
                    state_log,
                )?;

                Ok(Term::Constr { tag, fields })
            }
            9 => {
                state_log.push("(case ".to_string());
                let constr = Term::<T>::decode_debug(d, state_log)?.into();

                let branches = d.decode_list_with_debug(
                    |d, state_log| Term::<T>::decode_debug(d, state_log),
                    state_log,
                )?;

                Ok(Term::Case { constr, branches })
            }
            x => {
                state_log.push("parse error".to_string());

                let buffer_slice: Vec<u8> = d
                    .buffer
                    .to_vec()
                    .iter()
                    .skip(d.pos.saturating_sub(5))
                    .take(10)
                    .cloned()
                    .collect();

                Err(de::Error::UnknownTermConstructor(
                    x,
                    if d.pos > 5 { 5 } else { d.pos },
                    format!("{buffer_slice:02X?}"),
                    d.pos,
                    d.buffer.len(),
                ))
            }
        }
    }
}

/// Integers are typically smaller so we save space
/// by encoding them in 7 bits and this allows it to be byte alignment agnostic.
/// Strings and bytestrings span multiple bytes so using bytestring is
/// the most effective encoding.
/// i.e. A 17 or greater length byte array loses efficiency being encoded as
/// a unsigned integer instead of a byte array
impl Encode for Constant {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        match self {
            Constant::Integer(i) => {
                encode_constant(&[0], e)?;
                i.encode(e)?;
            }

            Constant::ByteString(bytes) => {
                encode_constant(&[1], e)?;
                bytes.encode(e)?;
            }
            Constant::String(s) => {
                encode_constant(&[2], e)?;
                s.encode(e)?;
            }
            Constant::Unit => encode_constant(&[3], e)?,
            Constant::Bool(b) => {
                encode_constant(&[4], e)?;
                b.encode(e)?;
            }
            Constant::ProtoList(typ, list) => {
                let mut type_encode = vec![7, 5];

                encode_type(typ, &mut type_encode);

                encode_constant(&type_encode, e)?;

                e.encode_list_with(list, |item, e| encode_constant_value(item, e))?;
            }
            Constant::ProtoArray(typ, items) => {
                let mut type_encode = vec![7, 12];

                encode_type(typ, &mut type_encode);

                encode_constant(&type_encode, e)?;

                e.encode_list_with(items, |item, e| encode_constant_value(item, e))?;
            }
            Constant::ProtoPair(type1, type2, a, b) => {
                let mut type_encode = vec![7, 7, 6];

                encode_type(type1, &mut type_encode);

                encode_type(type2, &mut type_encode);

                encode_constant(&type_encode, e)?;
                encode_constant_value(a, e)?;
                encode_constant_value(b, e)?;
            }
            Constant::Data(data) => {
                encode_constant(&[8], e)?;

                let cbor = data
                    .encode_fragment()
                    .map_err(|err| en::Error::Message(err.to_string()))?;

                cbor.encode(e)?;
            }
            Constant::Bls12_381G1Element(_) => {
                encode_constant(&[9], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 G1 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381G2Element(_) => {
                encode_constant(&[10], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 G2 points are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Bls12_381MlResult(_) => {
                encode_constant(&[11], e)?;

                return Err(en::Error::Message(
                    "BLS12-381 ML results are not supported for flat encoding".to_string(),
                ));
            }
            Constant::Value(value) => {
                encode_constant(&[13], e)?;
                encode_value(value, e)?;
            }
        }

        Ok(())
    }
}

fn encode_value(value: &crate::ast::Value, e: &mut Encoder) -> Result<(), en::Error> {
    for (currency, inner) in value.iter() {
        e.bool(true);
        currency.encode(e)?;

        for (token, quantity) in inner {
            e.bool(true);
            token.encode(e)?;
            BigInt::from(*quantity).encode(e)?;
        }
        e.bool(false);
    }
    e.bool(false);

    Ok(())
}

fn encode_constant_value(x: &Constant, e: &mut Encoder) -> Result<(), en::Error> {
    match x {
        Constant::Integer(x) => x.encode(e),
        Constant::ByteString(b) => b.encode(e),
        Constant::String(s) => s.encode(e),
        Constant::Unit => Ok(()),
        Constant::Bool(b) => b.encode(e),
        Constant::ProtoList(_, list) | Constant::ProtoArray(_, list) => {
            e.encode_list_with(list, |item, e| encode_constant_value(item, e))?;
            Ok(())
        }
        Constant::ProtoPair(_, _, a, b) => {
            encode_constant_value(a, e)?;

            encode_constant_value(b, e)
        }
        Constant::Data(data) => {
            let cbor = data
                .encode_fragment()
                .map_err(|err| en::Error::Message(err.to_string()))?;

            cbor.encode(e)
        }
        Constant::Bls12_381G1Element(_) => Err(en::Error::Message(
            "BLS12-381 G1 points are not supported for flat encoding".to_string(),
        )),
        Constant::Bls12_381G2Element(_) => Err(en::Error::Message(
            "BLS12-381 G2 points are not supported for flat encoding".to_string(),
        )),
        Constant::Bls12_381MlResult(_) => Err(en::Error::Message(
            "BLS12-381 ML results are not supported for flat encoding".to_string(),
        )),
        Constant::Value(value) => encode_value(value, e),
    }
}

fn encode_type(typ: &Type, bytes: &mut Vec<u8>) {
    match typ {
        Type::Integer => bytes.push(0),
        Type::ByteString => bytes.push(1),
        Type::String => bytes.push(2),
        Type::Unit => bytes.push(3),
        Type::Bool => bytes.push(4),
        Type::List(sub_typ) => {
            bytes.extend(vec![7, 5]);
            encode_type(sub_typ, bytes);
        }
        Type::Pair(type1, type2) => {
            bytes.extend(vec![7, 7, 6]);
            encode_type(type1, bytes);
            encode_type(type2, bytes);
        }
        Type::Data => bytes.push(8),
        Type::Bls12_381G1Element => bytes.push(9),
        Type::Bls12_381G2Element => bytes.push(10),
        Type::Bls12_381MlResult => bytes.push(11),
        Type::Array(sub_typ) => {
            bytes.extend_from_slice(&[7, 12]);
            encode_type(sub_typ, bytes);
        }
        Type::Value => bytes.push(13),
    }
}

/// Decodes a constant as Plutus' `decodeKindedUni` does: its type tags must
/// encode exactly one type, with no tag left over.
impl Decode<'_> for Constant {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let tags = decode_constant(d)?;
        let mut rest = VecDeque::from(tags.clone());

        let typ = decode_type(&mut rest)?;

        if !rest.is_empty() {
            return Err(de::Error::Message(format!(
                "Constant type tags {tags:?} encode more than one type"
            )));
        }

        decode_constant_value(typ.into(), d)
    }
}

fn decode_value(d: &mut Decoder) -> Result<crate::ast::Value, de::Error> {
    let entries = d.decode_list_with(|d| {
        let currency = Vec::<u8>::decode(d)?;
        let inner = d.decode_list_with(|d| {
            let token = Vec::<u8>::decode(d)?;
            let quantity = BigInt::decode(d)?;
            Ok((token, quantity))
        })?;
        Ok((currency, inner))
    })?;

    crate::ast::Value::from_canonical_entries(entries)
        .map_err(|error| de::Error::Message(error.to_string()))
}

fn decode_constant_value(typ: Rc<Type>, d: &mut Decoder) -> Result<Constant, de::Error> {
    match typ.as_ref() {
        Type::Integer => Ok(Constant::Integer(BigInt::decode(d)?)),
        Type::ByteString => Ok(Constant::ByteString(Vec::<u8>::decode(d)?)),
        Type::String => Ok(Constant::String(String::decode(d)?)),
        Type::Unit => Ok(Constant::Unit),
        Type::Bool => Ok(Constant::Bool(decode_bool(d)?)),
        Type::List(sub_type) => {
            let list: Vec<Rc<Constant>> =
                d.decode_list_with(|d| decode_constant_value(sub_type.clone(), d).map(Rc::new))?;

            Ok(Constant::ProtoList(sub_type.as_ref().clone(), list.into()))
        }
        Type::Array(sub_type) => {
            let items: Vec<Rc<Constant>> =
                d.decode_list_with(|d| decode_constant_value(sub_type.clone(), d).map(Rc::new))?;

            Ok(Constant::ProtoArray(
                sub_type.as_ref().clone(),
                items.into(),
            ))
        }
        Type::Pair(type1, type2) => {
            let a = decode_constant_value(type1.clone(), d)?;
            let b = decode_constant_value(type2.clone(), d)?;

            Ok(Constant::ProtoPair(
                type1.as_ref().clone(),
                type2.as_ref().clone(),
                a.into(),
                b.into(),
            ))
        }
        Type::Data => {
            let cbor = Vec::<u8>::decode(d)?;

            let data =
                Data::decode_fragment(&cbor).map_err(|err| de::Error::Message(err.to_string()))?;

            Ok(Constant::Data(data))
        }
        Type::Bls12_381G1Element => {
            let p1 = Vec::<u8>::decode(d)?;

            let _p1 = blst::blst_p1::uncompress(&p1)
                .map_err(|err| de::Error::Message(format!("Failed to uncompress p1: {err}")))?;

            Err(de::Error::Message(
                "BLS12-381 G1 points are not supported for flat decoding.".to_string(),
            ))
        }
        Type::Bls12_381G2Element => {
            let p2 = Vec::<u8>::decode(d)?;

            let _p2 = blst::blst_p2::uncompress(&p2)
                .map_err(|err| de::Error::Message(format!("Failed to uncompress p2: {err}")))?;

            Err(de::Error::Message(
                "BLS12-381 G2 points are not supported for flat decoding.".to_string(),
            ))
        }
        Type::Bls12_381MlResult => Err(de::Error::Message(
            "BLS12-381 ML results are not supported for flat decoding".to_string(),
        )),
        Type::Value => Ok(Constant::Value(decode_value(d)?)),
    }
}

/// Decodes a bool, failing at the end of the input where pallas'
/// `Decoder::bool` panics.
fn decode_bool(d: &mut Decoder) -> Result<bool, de::Error> {
    Ok(d.bits8(1)? == 1)
}

fn decode_type(types: &mut VecDeque<u8>) -> Result<Type, de::Error> {
    match types.pop_front() {
        Some(4) => Ok(Type::Bool),
        Some(0) => Ok(Type::Integer),
        Some(2) => Ok(Type::String),
        Some(1) => Ok(Type::ByteString),
        Some(3) => Ok(Type::Unit),
        Some(8) => Ok(Type::Data),
        Some(9) => Ok(Type::Bls12_381G1Element),
        Some(10) => Ok(Type::Bls12_381G2Element),
        Some(11) => Ok(Type::Bls12_381MlResult),
        Some(13) => Ok(Type::Value),
        Some(7) => match types.pop_front() {
            Some(5) => Ok(Type::List(decode_type(types)?.into())),
            Some(12) => Ok(Type::Array(decode_type(types)?.into())),
            Some(7) => match types.pop_front() {
                Some(6) => {
                    let type1 = decode_type(types)?;
                    let type2 = decode_type(types)?;

                    Ok(Type::Pair(type1.into(), type2.into()))
                }
                Some(x) => Err(de::Error::Message(format!(
                    "Unknown constant type tag: {x}"
                ))),
                None => Err(de::Error::Message("Unexpected empty buffer".to_string())),
            },
            Some(x) => Err(de::Error::Message(format!(
                "Unknown constant type tag: {x}"
            ))),
            None => Err(de::Error::Message("Unexpected empty buffer".to_string())),
        },

        Some(x) => Err(de::Error::Message(format!(
            "Unknown constant type tag: {x}"
        ))),
        None => Err(de::Error::Message("Unexpected empty buffer".to_string())),
    }
}

impl Encode for Unique {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        isize::from(*self).encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for Unique {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(isize::decode(d)?.into())
    }
}

impl Encode for Name {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.unique.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for Name {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(Name {
            text: String::decode(d)?,
            unique: Unique::decode(d)?,
        })
    }
}

impl Binder<'_> for Name {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.encode(e)?;

        Ok(())
    }

    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Name::decode(d)
    }

    fn text(&self) -> String {
        self.text.clone()
    }
}

impl Encode for NamedDeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.index.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for NamedDeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(NamedDeBruijn {
            text: String::decode(d)?,
            index: DeBruijn::decode(d)?,
        })
    }
}

impl Binder<'_> for NamedDeBruijn {
    fn binder_encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        self.text.encode(e)?;
        self.index.encode(e)?;

        Ok(())
    }

    fn binder_decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(NamedDeBruijn {
            text: String::decode(d)?,
            index: DeBruijn::decode(d)?,
        })
    }

    fn text(&self) -> String {
        format!("{}_{}", self.text, self.index)
    }
}

impl Encode for DeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        encode_word64(self.word(), e);

        Ok(())
    }
}

impl Decode<'_> for DeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(DeBruijn::from_word(decode_word64(d)?))
    }
}

impl Binder<'_> for DeBruijn {
    fn binder_encode(&self, _: &mut Encoder) -> Result<(), en::Error> {
        Ok(())
    }

    fn binder_decode(_d: &mut Decoder) -> Result<Self, de::Error> {
        Ok(DeBruijn::new(0))
    }

    fn text(&self) -> String {
        format!("i_{self}")
    }
}

impl Encode for FakeNamedDeBruijn {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        let index: DeBruijn = self.clone().into();

        index.encode(e)?;

        Ok(())
    }
}

impl Decode<'_> for FakeNamedDeBruijn {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let index = DeBruijn::decode(d)?;

        Ok(index.into())
    }
}

impl Binder<'_> for FakeNamedDeBruijn {
    fn binder_encode(&self, _: &mut Encoder) -> Result<(), en::Error> {
        Ok(())
    }

    fn binder_decode(_d: &mut Decoder) -> Result<Self, de::Error> {
        let index = DeBruijn::new(0);

        Ok(index.into())
    }

    fn text(&self) -> String {
        format!("{}_{}", self.0.text, self.0.index)
    }
}

impl Encode for DefaultFunction {
    fn encode(&self, e: &mut Encoder) -> Result<(), en::Error> {
        e.bits(BUILTIN_TAG_WIDTH as i64, *self as u8);

        Ok(())
    }
}

impl Decode<'_> for DefaultFunction {
    fn decode(d: &mut Decoder) -> Result<Self, de::Error> {
        let builtin_tag = d.bits8(BUILTIN_TAG_WIDTH as usize)?;
        builtin_tag.try_into()
    }
}

fn encode_term_tag(tag: u8, e: &mut Encoder) -> Result<(), en::Error> {
    safe_encode_bits(TERM_TAG_WIDTH, tag, e)
}

/// Encode a natural as a flat word: 7-bit groups, least significant first.
/// Unlike Pallas' `Encoder::word`, this does not go through `usize`, so it
/// handles the full `Word64` range on 32-bit targets as well.
fn encode_word64(word: u64, e: &mut Encoder) {
    let mut word = word;

    loop {
        let mut group = (word & 0x7f) as u8;
        word >>= 7;

        if word != 0 {
            group |= 0x80;
        }

        e.bits(8, group);

        if word == 0 {
            break;
        }
    }
}

/// Decode a flat-encoded natural that must fit in 64 bits, as the reference
/// `Word64` decoder requires. Pallas' `Decoder::word` accumulates into a
/// `usize` without any overflow check, so an over-long encoding silently
/// aliases a smaller value: a constr tag of 2^64 decodes as 0 everywhere, and
/// any tag at or above 2^32 aliases on 32-bit targets.
fn decode_word64(d: &mut Decoder) -> Result<u64, de::Error> {
    let mut word: u64 = 0;
    let mut shift: u32 = 0;

    loop {
        let byte = d.bits8(8)?;
        let chunk = u64::from(byte & 0x7f);

        if shift > 63 || (shift == 63 && chunk > 1) {
            return Err(de::Error::Message(
                "flat-encoded natural does not fit in 64 bits".to_string(),
            ));
        }

        word |= chunk << shift;

        if byte & 0x80 == 0 {
            return Ok(word);
        }

        shift += 7;
    }
}

/// Like [`decode_word64`], but additionally refuses values that do not fit in
/// a `usize` (only possible on 32-bit targets) rather than truncating them.
fn decode_usize(d: &mut Decoder) -> Result<usize, de::Error> {
    let word = decode_word64(d)?;

    usize::try_from(word).map_err(|_| {
        de::Error::Message(format!(
            "flat-encoded natural {word} does not fit in a {}-bit word",
            usize::BITS
        ))
    })
}

fn decode_term_tag(d: &mut Decoder) -> Result<u8, de::Error> {
    d.bits8(TERM_TAG_WIDTH as usize)
}

fn safe_encode_bits(num_bits: u32, byte: u8, e: &mut Encoder) -> Result<(), en::Error> {
    if 2_u8.pow(num_bits) <= byte {
        Err(en::Error::Message(format!(
            "Overflow detected, cannot fit {byte} in {num_bits} bits."
        )))
    } else {
        e.bits(num_bits as i64, byte);
        Ok(())
    }
}

pub fn encode_constant(tag: &[u8], e: &mut Encoder) -> Result<(), en::Error> {
    e.encode_list_with(tag, encode_constant_tag)?;

    Ok(())
}

/// Decodes the type tags of a constant, rejecting more than the ledger
/// allows ([`MAX_CONSTANT_TYPE_SIZE`]) as soon as there are, so that a long
/// list is neither read in full nor decoded into a deeply nested type.
pub fn decode_constant(d: &mut Decoder) -> Result<Vec<u8>, de::Error> {
    let mut tags = Vec::new();

    while decode_bool(d)? {
        if tags.len() == MAX_CONSTANT_TYPE_SIZE {
            return Err(de::Error::Message(format!(
                "Constant type takes more than {MAX_CONSTANT_TYPE_SIZE} tags"
            )));
        }

        tags.push(decode_constant_tag(d)?);
    }

    Ok(tags)
}

pub fn encode_constant_tag(tag: &u8, e: &mut Encoder) -> Result<(), en::Error> {
    safe_encode_bits(CONST_TAG_WIDTH, *tag, e)
}

pub fn decode_constant_tag(d: &mut Decoder) -> Result<u8, de::Error> {
    d.bits8(CONST_TAG_WIDTH as usize)
}

#[cfg(test)]
mod tests {
    use super::{Constant, Program, Term, decode_bool, decode_value, encode_constant};
    use crate::{
        ast::{BigIntValueEntries, DeBruijn, Name, Type, Value},
        parser,
    };
    use indoc::indoc;
    use num_bigint::BigInt;
    use pallas_codec::flat::{
        Flat,
        de::Decoder,
        en::{Encode, Encoder},
    };

    #[test]
    fn flat_encode_integer() {
        let program = Program::<Name> {
            version: (11, 22, 33),
            term: Term::Constant(Constant::Integer(11.into()).into()),
        };

        let expected_bytes = vec![
            0b00001011, 0b00010110, 0b00100001, 0b01001000, 0b00000101, 0b10000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_encode_list_list_integer() {
        let program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant(
                Constant::ProtoList(
                    Type::List(Type::Integer.into()),
                    vec![
                        Constant::ProtoList(
                            Type::Integer,
                            vec![Constant::Integer(7.into()).into()].into(),
                        )
                        .into(),
                        Constant::ProtoList(
                            Type::Integer,
                            vec![Constant::Integer(5.into()).into()].into(),
                        )
                        .into(),
                    ]
                    .into(),
                )
                .into(),
            ),
        };

        let expected_bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11010110, 0b11110101, 0b10000011,
            0b00001110, 0b01100001, 0b01000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_encode_pair_pair_integer_bool_integer() {
        let program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant(
                Constant::ProtoPair(
                    Type::Pair(Type::Integer.into(), Type::Bool.into()),
                    Type::Integer,
                    Constant::ProtoPair(
                        Type::Integer,
                        Type::Bool,
                        Constant::Integer(11.into()).into(),
                        Constant::Bool(true).into(),
                    )
                    .into(),
                    Constant::Integer(11.into()).into(),
                )
                .into(),
            ),
        };

        let expected_bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11011110, 0b11010111, 0b10111101,
            0b10100001, 0b01001000, 0b00000101, 0b10100010, 0b11000001,
        ];

        let actual_bytes = program.to_flat().unwrap();

        assert_eq!(actual_bytes, expected_bytes)
    }

    #[test]
    fn flat_decode_list_list_integer() {
        let bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11010110, 0b11110101, 0b10000011,
            0b00001110, 0b01100001, 0b01000001,
        ];

        let expected_program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant(
                Constant::ProtoList(
                    Type::List(Type::Integer.into()),
                    vec![
                        Constant::ProtoList(
                            Type::Integer,
                            vec![Constant::Integer(7.into()).into()].into(),
                        )
                        .into(),
                        Constant::ProtoList(
                            Type::Integer,
                            vec![Constant::Integer(5.into()).into()].into(),
                        )
                        .into(),
                    ]
                    .into(),
                )
                .into(),
            ),
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn flat_decode_pair_pair_integer_bool_integer() {
        let bytes = vec![
            0b00000001, 0b00000000, 0b00000000, 0b01001011, 0b11011110, 0b11010111, 0b10111101,
            0b10100001, 0b01001000, 0b00000101, 0b10100010, 0b11000001,
        ];

        let expected_program = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant(
                Constant::ProtoPair(
                    Type::Pair(Type::Integer.into(), Type::Bool.into()),
                    Type::Integer,
                    Constant::ProtoPair(
                        Type::Integer,
                        Type::Bool,
                        Constant::Integer(11.into()).into(),
                        Constant::Bool(true).into(),
                    )
                    .into(),
                    Constant::Integer(11.into()).into(),
                )
                .into(),
            ),
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn flat_decode_integer() {
        let bytes = vec![
            0b00001011, 0b00010110, 0b00100001, 0b01001000, 0b00000101, 0b10000001,
        ];

        let expected_program = Program {
            version: (11, 22, 33),
            term: Term::Constant(Constant::Integer(11.into()).into()),
        };

        let actual_program: Program<Name> = Program::unflat(&bytes).unwrap();

        assert_eq!(actual_program, expected_program)
    }

    #[test]
    fn unflat_string_escape() {
        let cbor = "490000004901015c0001";

        let program =
            Program::<DeBruijn>::from_hex(cbor, &mut Vec::new(), &mut Vec::new()).unwrap();

        assert_eq!(
            program.to_pretty().as_str(),
            indoc! { r#"
              (program
                0.0.0
                (con string "\\")
              )"#}
        );
    }

    #[test]
    fn uplc_parser_string_escape() {
        let source = indoc! { r#"
            (program
              0.0.0
              (con string "\n\t\\\"\'\r")
            )"#};

        let program = parser::program(source).unwrap();

        assert_eq!(program.to_pretty(), source);
    }
    #[test]
    fn flat_roundtrips_value_universe_tag_and_nested_payload() {
        let value = Value::from_canonical_entries(vec![
            (
                vec![0x01],
                vec![
                    (vec![0x00], BigInt::from(i128::MIN)),
                    (vec![0xff], BigInt::from(7)),
                ],
            ),
            (
                vec![0x02],
                vec![
                    (vec![], BigInt::from(-42)),
                    (vec![0x01], BigInt::from(i128::MAX)),
                ],
            ),
        ])
        .unwrap();
        let expected = Program::<Name> {
            version: (1, 0, 0),
            term: Term::Constant(Constant::Value(value).into()),
        };

        let bytes = expected.to_flat().unwrap();
        let actual = Program::<Name>::unflat(&bytes).unwrap();

        assert_eq!(actual, expected);
    }

    fn encode_raw_value(entries: BigIntValueEntries) -> Vec<u8> {
        let mut encoder = Encoder::new();
        for (currency, tokens) in entries {
            encoder.bool(true);
            currency.encode(&mut encoder).unwrap();
            for (token, quantity) in tokens {
                encoder.bool(true);
                token.encode(&mut encoder).unwrap();
                quantity.encode(&mut encoder).unwrap();
            }
            encoder.bool(false);
        }
        encoder.bool(false);

        // Encoding an empty bytestring flushes the final partial byte. Its payload
        // is intentionally left after the Value; the decoder below consumes only
        // the Value payload.
        Vec::<u8>::new().encode(&mut encoder).unwrap();
        encoder.buffer
    }

    #[test]
    fn flat_rejects_malformed_value_payloads() {
        let malformed = vec![
            vec![
                (vec![1], vec![(vec![0], BigInt::from(1))]),
                (vec![0], vec![(vec![0], BigInt::from(1))]),
            ],
            vec![
                (vec![0], vec![(vec![0], BigInt::from(1))]),
                (vec![0], vec![(vec![1], BigInt::from(1))]),
            ],
            vec![(vec![0], vec![])],
            vec![(
                vec![0],
                vec![(vec![1], BigInt::from(1)), (vec![0], BigInt::from(1))],
            )],
            vec![(
                vec![0],
                vec![(vec![0], BigInt::from(1)), (vec![0], BigInt::from(2))],
            )],
            vec![(vec![0], vec![(vec![0], BigInt::from(0))])],
            vec![(vec![0; 33], vec![(vec![0], BigInt::from(1))])],
            vec![(vec![0], vec![(vec![0; 33], BigInt::from(1))])],
            vec![(
                vec![0],
                vec![(vec![0], BigInt::from(i128::MAX) + BigInt::from(1))],
            )],
        ];

        for entries in malformed {
            let bytes = encode_raw_value(entries);
            assert!(decode_value(&mut Decoder::new(&bytes)).is_err());
        }

        let mut truncated = encode_raw_value(vec![(vec![0], vec![(vec![0], BigInt::from(1))])]);
        truncated.truncate(truncated.len() - 2);
        assert!(decode_value(&mut Decoder::new(&truncated)).is_err());
    }

    /// Flat bytes for `(program 1.1.0 (constr <tag>))`, with the tag written
    /// as the given 7-bit groups (least significant first).
    fn constr_with_tag_groups(groups: &[u8]) -> Vec<u8> {
        let mut e = Encoder::new();
        for word in [1_usize, 1, 0] {
            word.encode(&mut e).unwrap();
        }
        e.bits(4, 8);
        for (i, group) in groups.iter().enumerate() {
            let more = if i + 1 < groups.len() { 0x80 } else { 0 };
            e.bits(8, group | more);
        }
        // empty field list, then padding
        e.bits(1, 0);
        pallas_codec::flat::filler::Filler::FillerEnd
            .encode(&mut e)
            .unwrap();
        e.buffer
    }

    #[test]
    fn constr_tag_must_fit_in_64_bits() {
        // 2^64 - 1 is the largest tag the reference implementation accepts.
        let max =
            constr_with_tag_groups(&[0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 1]);
        let program = Program::<DeBruijn>::unflat(&max).unwrap();
        assert_eq!(
            program.term,
            Term::Constr {
                tag: u64::MAX,
                fields: vec![]
            }
        );
        assert_eq!(program.to_flat().unwrap(), max);

        // 2^32 + 1 must neither alias nor be rejected on 32-bit targets.
        let wide = constr_with_tag_groups(&[1, 0, 0, 0, 0x10]);
        let program = Program::<DeBruijn>::unflat(&wide).unwrap();
        assert_eq!(
            program.term,
            Term::Constr {
                tag: (1 << 32) + 1,
                fields: vec![]
            }
        );
        assert_eq!(program.to_flat().unwrap(), wide);

        // 2^64 used to alias to 0.
        let overflow = constr_with_tag_groups(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
        assert!(Program::<DeBruijn>::unflat(&overflow).is_err());

        // An 11th group can only overflow, even when it is zero.
        let overlong = constr_with_tag_groups(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(Program::<DeBruijn>::unflat(&overlong).is_err());

        // A redundant (but in-range) encoding is still accepted.
        let padded = constr_with_tag_groups(&[5, 0]);
        assert_eq!(
            Program::<DeBruijn>::unflat(&padded).unwrap().term,
            Term::Constr {
                tag: 5,
                fields: vec![]
            }
        );
    }

    #[test]
    fn constr_and_case_rejected_before_uplc_1_1() {
        for source in [
            "(constr 0)",
            "(case (constr 0) (con integer 1))",
            "(lam x [(force (delay (constr 1 x))) x])",
        ] {
            let mut program: Program<DeBruijn> =
                parser::program(&format!("(program 1.1.0 {source})"))
                    .unwrap()
                    .try_into()
                    .unwrap();
            assert!(Program::<DeBruijn>::unflat(&program.to_flat().unwrap()).is_ok());

            program.version = (1, 0, 0);
            assert!(
                Program::<DeBruijn>::unflat(&program.to_flat().unwrap()).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn bytes_after_the_program_rejected() {
        let program: Program<DeBruijn> = parser::program("(program 1.0.0 (con integer 1))")
            .unwrap()
            .try_into()
            .unwrap();
        let bytes = program.to_flat().unwrap();

        assert_eq!(Program::<DeBruijn>::unflat(&bytes).unwrap(), program);

        for trailing in [&[0x00][..], &[0x01], &[0x00, 0x00, 0x00, 0x00]] {
            let bytes = [&bytes[..], trailing].concat();

            assert!(
                Program::<DeBruijn>::unflat(&bytes).is_err(),
                "{}",
                hex::encode(&bytes)
            );
        }
    }

    #[test]
    fn array_constant_round_trips() {
        let source = "(program 1.1.0 (con (array (list integer)) [[1, 2], [], [3]]))";
        let program: Program<DeBruijn> = parser::program(source).unwrap().try_into().unwrap();

        assert!(matches!(
            program.term,
            Term::Constant(ref c) if matches!(c.as_ref(), Constant::ProtoArray(Type::List(_), items) if items.len() == 3)
        ));

        let bytes = program.to_flat().unwrap();
        // Decodes to the same array in Plutus.
        assert_eq!(hex::encode(&bytes), "0101004bf2f58302822c1801");

        let decoded = Program::<DeBruijn>::unflat(&bytes).unwrap();
        assert_eq!(decoded, program);

        let printed: Program<Name> = (&decoded).try_into().unwrap();
        let reparsed: Program<DeBruijn> = parser::program(&printed.to_pretty())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(reparsed, program);
    }

    /// Flat bytes for `(program 1.1.0 (con <type> <value>))`, with the type
    /// given as its raw tags and the value written by `value`.
    fn constant_with_tags(tags: &[u8], value: impl FnOnce(&mut Encoder)) -> Vec<u8> {
        let mut e = Encoder::new();
        for word in [1_usize, 1, 0] {
            word.encode(&mut e).unwrap();
        }
        e.bits(4, 4);
        encode_constant(tags, &mut e).unwrap();
        value(&mut e);
        pallas_codec::flat::filler::Filler::FillerEnd
            .encode(&mut e)
            .unwrap();
        e.buffer
    }

    #[test]
    fn constant_type_tags_must_encode_one_type() {
        // Units have no value, and a list or array value here is empty.
        for (tags, is_list, leftover) in [
            (&[3][..], false, &[3][..]),
            (&[7, 7, 6, 3, 3], false, &[3]),
            (&[7, 5, 3], true, &[0]),
            (&[7, 12, 3], true, &[3, 3, 3]),
            (&[7, 5, 7, 7, 6, 3, 3], true, &[7]),
        ] {
            let value = |e: &mut Encoder| {
                if is_list {
                    e.bits(1, 0);
                }
            };

            let valid = constant_with_tags(tags, value);
            assert!(Program::<DeBruijn>::unflat(&valid).is_ok(), "{tags:?}");

            let invalid = constant_with_tags(&[tags, leftover].concat(), value);
            assert!(
                Program::<DeBruijn>::unflat(&invalid).is_err(),
                "{tags:?} {leftover:?}"
            );
        }
    }

    #[test]
    fn constant_types_limited_to_32_tags() {
        // A list type takes 2 more tags than its element type, and a pair
        // type 3 more than its two, so a type takes an odd number of tags.
        let lists = |depth: usize| [&[7, 5].repeat(depth)[..], &[3]].concat();
        let empty = |e: &mut Encoder| {
            e.bits(1, 0);
        };

        assert_eq!(lists(15).len(), 31);
        assert!(Program::<DeBruijn>::unflat(&constant_with_tags(&lists(15), empty)).is_ok());

        // Rejected before the tags are decoded into a type.
        for depth in [16, 100_000] {
            assert!(
                Program::<DeBruijn>::unflat(&constant_with_tags(&lists(depth), empty)).is_err()
            );
        }
    }

    #[test]
    fn truncated_bools_rejected() {
        let mut d = Decoder::new(&[0b1000_0000]);
        assert!(decode_bool(&mut d).unwrap());
        for _ in 0..7 {
            assert!(!decode_bool(&mut d).unwrap());
        }
        assert!(decode_bool(&mut d).is_err());

        // A `bool` constant whose value is past the end of the input, which
        // used to panic in pallas' `Decoder::bool`.
        let bytes = hex::decode("010000183682d528").unwrap();
        assert!(Program::<DeBruijn>::unflat(&bytes).is_err());
    }
}
