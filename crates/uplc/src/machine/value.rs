use super::{
    Error,
    runtime::{BuiltinRuntime, BuiltinSemantics},
};
use crate::{
    ast::{Constant, Data, DebugDepth, ListSpine, NamedDeBruijn, Term, Type},
    builtins::DefaultFunction,
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use pallas_primitives::conway;
use std::{
    cell::{Cell, RefCell},
    mem::size_of,
    ops::{Deref, DerefMut},
    rc::Rc,
};

/// Number of bindings per environment chunk. Extending a shared environment
/// copies at most one chunk, so applications cost O(ENV_CHUNK) instead of
/// O(depth), while most lookups stay within the first one or two chunks.
const ENV_CHUNK: usize = 8;

/// A persistent environment of values, indexed by de Bruijn index (1 is the
/// most recent binding).
///
/// Bindings live in a chain of chunks, newest first. Every chunk but the
/// newest is full, so the chain is depth / ENV_CHUNK long. Closures share
/// the chunks they capture; only the newest chunk is ever copied on write.
#[derive(Clone, Debug, Default)]
pub struct Env(Option<Rc<EnvChunk>>);

#[derive(Debug)]
pub struct EnvChunk {
    values: Vec<Value>,
    parent: Env,
}

impl Drop for EnvChunk {
    fn drop(&mut self) {
        drop_nested(&mut self.values, std::mem::take(&mut self.parent));
    }
}

impl Clone for EnvChunk {
    fn clone(&self) -> Self {
        let mut values = Vec::with_capacity(ENV_CHUNK);
        values.extend_from_slice(&self.values);

        EnvChunk {
            values,
            parent: self.parent.clone(),
        }
    }
}

impl Env {
    pub fn push(&mut self, value: Value) {
        match &mut self.0 {
            Some(chunk) if chunk.values.len() < ENV_CHUNK => {
                Rc::make_mut(chunk).values.push(value);
            }
            _ => {
                let mut values = Vec::with_capacity(ENV_CHUNK);
                values.push(value);

                let parent = Env(self.0.take());

                self.0 = Some(Rc::new(EnvChunk { values, parent }));
            }
        }
    }

    /// The value bound at de Bruijn index `index`, if any.
    #[inline]
    pub fn get(&self, mut index: usize) -> Option<&Value> {
        if index == 0 {
            return None;
        }

        let mut chunk = self.0.as_deref()?;

        loop {
            let len = chunk.values.len();

            if index <= len {
                return Some(&chunk.values[len - index]);
            }

            index -= len;
            chunk = chunk.parent.0.as_deref()?;
        }
    }

    /// Whether both environments are the very same bindings.
    fn same(&self, other: &Env) -> bool {
        match (&self.0, &other.0) {
            (Some(left), Some(right)) => Rc::ptr_eq(left, right),
            (None, None) => true,
            _ => false,
        }
    }

    /// Bindings from the most recent to the oldest.
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        let mut chunk = self.0.as_deref();

        std::iter::from_fn(move || {
            let current = chunk?;
            chunk = current.parent.0.as_deref();
            Some(current.values.iter().rev())
        })
        .flatten()
    }
}

impl PartialEq for Env {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

/// Values can nest as deep as an evaluation makes them (closures capturing
/// closures, constructors holding constructors), so cloning, comparing,
/// formatting and dropping them do not recurse (see [`Values`] and
/// [`EnvChunk`] for dropping).
#[derive(Clone)]
pub enum Value {
    Con(Rc<Constant>),
    Delay(Rc<Term<NamedDeBruijn>>, Env),
    Lambda {
        parameter_name: Rc<NamedDeBruijn>,
        body: Rc<Term<NamedDeBruijn>>,
        env: Env,
    },
    Builtin {
        fun: DefaultFunction,
        runtime: BuiltinRuntime,
    },
    /// Fields are shared so that copying a constructor value (variable
    /// lookup, environment capture) is O(1) instead of a deep copy of the
    /// whole structure.
    Constr {
        tag: u64,
        fields: Rc<Values>,
    },
}

/// The values a value holds: the fields of a constr, or the arguments of a
/// builtin. It dereferences to a `Vec<Value>`.
#[derive(Clone, Default, PartialEq)]
pub struct Values(Vec<Value>);

impl Values {
    pub fn into_vec(mut self) -> Vec<Value> {
        std::mem::take(&mut self.0)
    }
}

impl From<Vec<Value>> for Values {
    fn from(values: Vec<Value>) -> Self {
        Values(values)
    }
}

impl FromIterator<Value> for Values {
    fn from_iter<I: IntoIterator<Item = Value>>(values: I) -> Self {
        Values(values.into_iter().collect())
    }
}

impl Deref for Values {
    type Target = Vec<Value>;

    fn deref(&self) -> &Vec<Value> {
        &self.0
    }
}

impl DerefMut for Values {
    fn deref_mut(&mut self) -> &mut Vec<Value> {
        &mut self.0
    }
}

impl std::fmt::Debug for Values {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Drop for Values {
    fn drop(&mut self) {
        drop_nested(&mut self.0, Env::default());
    }
}

/// How many collections of values (environment chunks, constr fields and
/// builtin arguments) may be dropped inside one another before deeper ones
/// are queued instead, so that dropping a deeply nested value or a long
/// environment uses bounded stack while dropping a shallow one costs next to
/// nothing more.
const MAX_NESTED_VALUE_DROPS: usize = 512;

std::thread_local! {
    /// How many collections of values are being dropped inside one another on
    /// this thread.
    static VALUE_DROP_DEPTH: Cell<usize> = const { Cell::new(0) };

    /// Whether `DEFERRED_VALUE_DROPS` may hold anything.
    static VALUE_DROPS_DEFERRED: Cell<bool> = const { Cell::new(false) };

    /// What collections dropped too deep held, left for the outermost drop.
    static DEFERRED_VALUE_DROPS: RefCell<Vec<(Vec<Value>, Env)>> =
        const { RefCell::new(Vec::new()) };
}

/// Drops what a collection being dropped holds: its values and, for an
/// environment chunk, the environment it extends.
#[inline]
fn drop_nested(values: &mut Vec<Value>, parent: Env) {
    if values.is_empty() && parent.0.is_none() {
        return;
    }

    let depth = VALUE_DROP_DEPTH.get();

    if depth >= MAX_NESTED_VALUE_DROPS {
        defer_drop(std::mem::take(values), parent);

        return;
    }

    VALUE_DROP_DEPTH.set(depth + 1);
    values.clear();
    drop(parent);

    if depth == 0 && VALUE_DROPS_DEFERRED.get() {
        drop_deferred();
    }

    VALUE_DROP_DEPTH.set(depth);
}

#[cold]
fn defer_drop(values: Vec<Value>, parent: Env) {
    // Only fails while the thread is exiting, when its locals are being
    // destroyed; what is left is then dropped right here.
    let _ = DEFERRED_VALUE_DROPS.try_with(|deferred| {
        deferred.borrow_mut().push((values, parent));
    });

    VALUE_DROPS_DEFERRED.set(true);
}

#[cold]
fn drop_deferred() {
    while let Some(nested) = DEFERRED_VALUE_DROPS
        .try_with(|deferred| deferred.borrow_mut().pop())
        .ok()
        .flatten()
    {
        drop(nested);
    }

    VALUE_DROPS_DEFERRED.set(false);
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        let mut pending = vec![(self, other)];

        // Pairs up the values of two sequences to compare next, if their
        // lengths match.
        fn pair_up<'a>(
            pending: &mut Vec<(&'a Value, &'a Value)>,
            left: impl Iterator<Item = &'a Value>,
            mut right: impl Iterator<Item = &'a Value>,
        ) -> bool {
            for left in left {
                let Some(right) = right.next() else {
                    return false;
                };

                pending.push((left, right));
            }

            right.next().is_none()
        }

        while let Some(pair) = pending.pop() {
            let equal = match pair {
                (Value::Con(left), Value::Con(right)) => left == right,
                (Value::Delay(left_body, left_env), Value::Delay(right_body, right_env)) => {
                    (Rc::ptr_eq(left_body, right_body) || left_body == right_body)
                        && (left_env.same(right_env)
                            || pair_up(&mut pending, left_env.iter(), right_env.iter()))
                }
                (
                    Value::Lambda {
                        parameter_name: left_name,
                        body: left_body,
                        env: left_env,
                    },
                    Value::Lambda {
                        parameter_name: right_name,
                        body: right_body,
                        env: right_env,
                    },
                ) => {
                    left_name == right_name
                        && (Rc::ptr_eq(left_body, right_body) || left_body == right_body)
                        && (left_env.same(right_env)
                            || pair_up(&mut pending, left_env.iter(), right_env.iter()))
                }
                (
                    Value::Builtin {
                        fun: left_fun,
                        runtime: left,
                    },
                    Value::Builtin {
                        fun: right_fun,
                        runtime: right,
                    },
                ) => {
                    left_fun == right_fun
                        && left.fun == right.fun
                        && left.forces == right.forces
                        && pair_up(&mut pending, left.args().iter(), right.args().iter())
                }
                (
                    Value::Constr {
                        tag: left_tag,
                        fields: left,
                    },
                    Value::Constr {
                        tag: right_tag,
                        fields: right,
                    },
                ) => {
                    left_tag == right_tag
                        && (Rc::ptr_eq(left, right)
                            || pair_up(&mut pending, left.iter(), right.iter()))
                }
                _ => false,
            };

            if !equal {
                return false;
            }
        }

        true
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some(_depth) = DebugDepth::enter() else {
            return f.write_str("..");
        };

        match self {
            Value::Con(constant) => f.debug_tuple("Con").field(constant).finish(),
            Value::Delay(body, env) => f.debug_tuple("Delay").field(body).field(env).finish(),
            Value::Lambda {
                parameter_name,
                body,
                env,
            } => f
                .debug_struct("Lambda")
                .field("parameter_name", parameter_name)
                .field("body", body)
                .field("env", env)
                .finish(),
            Value::Builtin { fun, runtime } => f
                .debug_struct("Builtin")
                .field("fun", fun)
                .field("runtime", runtime)
                .finish(),
            Value::Constr { tag, fields } => f
                .debug_struct("Constr")
                .field("tag", tag)
                .field("fields", fields)
                .finish(),
        }
    }
}

impl Value {
    pub fn integer(n: BigInt) -> Self {
        let constant = Constant::Integer(n);

        Value::Con(constant.into())
    }

    pub fn bool(n: bool) -> Self {
        let constant = Constant::Bool(n);

        Value::Con(constant.into())
    }

    pub fn byte_string(n: Vec<u8>) -> Self {
        let constant = Constant::ByteString(n);

        Value::Con(constant.into())
    }

    pub fn string(n: String) -> Self {
        let constant = Constant::String(n);

        Value::Con(constant.into())
    }

    pub fn list(typ: Type, n: impl Into<ListSpine>) -> Self {
        let constant = Constant::ProtoList(typ, n.into());

        Value::Con(constant.into())
    }

    pub fn data(d: Data) -> Self {
        let constant = Constant::Data(d);

        Value::Con(constant.into())
    }
    pub fn from_value(value: crate::ast::Value) -> Self {
        Value::Con(Constant::Value(value).into())
    }

    pub(super) fn unwrap_integer(&self) -> Result<&BigInt, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Integer(integer) = inner else {
            return Err(Error::TypeMismatch(Type::Integer, inner.into()));
        };

        Ok(integer)
    }

    pub(super) fn unwrap_byte_string(&self) -> Result<&Vec<u8>, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ByteString(byte_string) = inner else {
            return Err(Error::TypeMismatch(Type::ByteString, inner.into()));
        };

        Ok(byte_string)
    }

    pub(super) fn unwrap_string(&self) -> Result<&String, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::String(string) = inner else {
            return Err(Error::TypeMismatch(Type::String, inner.into()));
        };

        Ok(string)
    }

    pub(super) fn unwrap_bool(&self) -> Result<&bool, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bool(condition) = inner else {
            return Err(Error::TypeMismatch(Type::Bool, inner.into()));
        };

        Ok(condition)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn unwrap_pair(
        &self,
    ) -> Result<(&Type, &Type, &Rc<Constant>, &Rc<Constant>), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoPair(t1, t2, first, second) = inner else {
            return Err(Error::PairTypeMismatch(inner.into()));
        };

        Ok((t1, t2, first, second))
    }

    pub(super) fn unwrap_list(&self) -> Result<(&Type, &ListSpine), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(t, list) = inner else {
            return Err(Error::ListTypeMismatch(inner.into()));
        };

        Ok((t, list))
    }

    pub(super) fn unwrap_array(&self) -> Result<(&Type, &ListSpine), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoArray(t, items) = inner else {
            return Err(Error::ArrayTypeMismatch(inner.into()));
        };

        Ok((t, items))
    }

    pub(super) fn unwrap_data(&self) -> Result<&Data, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Data(data) = inner else {
            return Err(Error::TypeMismatch(Type::Data, inner.into()));
        };

        Ok(data)
    }
    pub(super) fn unwrap_value(&self) -> Result<&crate::ast::Value, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Value(value) = inner else {
            return Err(Error::TypeMismatch(Type::Value, inner.into()));
        };

        Ok(value)
    }

    pub(super) fn unwrap_unit(&self) -> Result<(), Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Unit = inner else {
            return Err(Error::TypeMismatch(Type::Unit, inner.into()));
        };

        Ok(())
    }

    /// Like `unwrap_constant`, but hands out the shared allocation itself so
    /// the caller can store it (e.g. as a list element) without a deep clone.
    pub(super) fn unwrap_constant_rc(&self) -> Result<Rc<Constant>, Error> {
        let Value::Con(item) = self else {
            return Err(Error::NotAConstant(self.clone()));
        };

        Ok(Rc::clone(item))
    }

    pub(super) fn unwrap_constant(&self) -> Result<&Constant, Error> {
        let Value::Con(item) = self else {
            return Err(Error::NotAConstant(self.clone()));
        };

        Ok(item.as_ref())
    }

    pub(super) fn unwrap_data_list(&self) -> Result<&ListSpine, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(Type::Data, list) = inner else {
            return Err(Error::TypeMismatch(
                Type::List(Type::Data.into()),
                inner.into(),
            ));
        };

        Ok(list)
    }

    pub(super) fn unwrap_int_list(&self) -> Result<&ListSpine, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::ProtoList(Type::Integer, list) = inner else {
            return Err(Error::TypeMismatch(
                Type::List(Type::Integer.into()),
                inner.into(),
            ));
        };

        Ok(list)
    }

    pub(super) fn unwrap_bls12_381_g1_element(&self) -> Result<&blst::blst_p1, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381G1Element(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381G1Element, inner.into()));
        };

        Ok(element)
    }

    pub(super) fn unwrap_bls12_381_g2_element(&self) -> Result<&blst::blst_p2, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381G2Element(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381G2Element, inner.into()));
        };

        Ok(element)
    }

    pub(super) fn unwrap_bls12_381_ml_result(&self) -> Result<&blst::blst_fp12, Error> {
        let inner = self.unwrap_constant()?;

        let Constant::Bls12_381MlResult(element) = inner else {
            return Err(Error::TypeMismatch(Type::Bls12_381MlResult, inner.into()));
        };

        Ok(element)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Value::Con(i) if matches!(i.as_ref(), Constant::Integer(_)))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Con(b) if matches!(b.as_ref(), Constant::Bool(_)))
    }

    /// The size Plutus gives an integer that counts bytes when costing it
    /// (`NumBytesCostedAsNumWords`): the number of 8-byte words that many
    /// bytes take up, saturating. Whether the integer is a valid number of
    /// bytes is up to the builtin, once it has been costed.
    pub fn cost_as_size(&self) -> Result<i64, Error> {
        let bytes = self.unwrap_integer()?.magnitude();

        Ok(match bytes.to_u64() {
            Some(bytes) => bytes.div_ceil(8) as i64,
            None => ((bytes - 1u8) / 8u8 + 1u8).to_i64().unwrap_or(i64::MAX),
        })
    }

    pub fn to_ex_mem(&self) -> i64 {
        self.to_ex_mem_with_semantics(BuiltinSemantics::C)
    }

    pub fn to_ex_mem_with_semantics(&self, semantics: BuiltinSemantics) -> i64 {
        match self {
            Value::Con(c) => Self::constant_to_ex_mem(c, semantics),
            Value::Delay(_, _) => 1,
            Value::Lambda { .. } => 1,
            Value::Builtin { .. } => 1,
            Value::Constr { .. } => 1,
        }
    }

    fn constant_to_ex_mem(constant: &Constant, semantics: BuiltinSemantics) -> i64 {
        let mut stack = vec![constant];
        let mut total = 0;

        while let Some(constant) = stack.pop() {
            match constant {
                Constant::Integer(i) => total += Self::integer_to_ex_mem(i),
                Constant::ByteString(b) => total += Self::byte_string_to_ex_mem(b),
                Constant::String(s) => {
                    total += if semantics.costs_strings_by_utf8_bytes() {
                        Self::utf8_text_to_ex_mem(s)
                    } else {
                        s.chars().count() as i64
                    };
                }
                Constant::Unit | Constant::Bool(_) => total += 1,
                Constant::ProtoList(_, items) => {
                    stack.extend(items.iter().map(|item| item.as_ref()))
                }
                // Plutus sizes an array by its length alone.
                Constant::ProtoArray(_, items) => total += items.len() as i64,
                Constant::ProtoPair(_, _, l, r) => {
                    stack.push(l.as_ref());
                    stack.push(r.as_ref());
                }
                Constant::Data(item) => total += Self::data_to_ex_mem_inner(item),
                Constant::Bls12_381G1Element(_) => total += size_of::<blst::blst_p1>() as i64 / 8,
                Constant::Bls12_381G2Element(_) => total += size_of::<blst::blst_p2>() as i64 / 8,
                Constant::Bls12_381MlResult(_) => total += size_of::<blst::blst_fp12>() as i64 / 8,
                Constant::Value(value) => total += value.total_size() as i64,
            }
        }

        total
    }

    fn utf8_text_to_ex_mem(s: &str) -> i64 {
        (s.len() as i64) / 4
    }

    fn integer_to_ex_mem(i: &BigInt) -> i64 {
        if i.is_zero() {
            1
        } else {
            ((i.bits() as i64 - 1) / 64) + 1
        }
    }

    fn byte_string_to_ex_mem(b: &[u8]) -> i64 {
        if b.is_empty() {
            1
        } else {
            ((b.len() as i64 - 1) / 8) + 1
        }
    }

    pub fn data_to_ex_mem(&self, data: &Data) -> i64 {
        Self::data_to_ex_mem_inner(data)
    }

    fn data_to_ex_mem_inner(data: &Data) -> i64 {
        // The order nodes are visited in does not matter for a sum.
        let mut stack: Vec<&Data> = vec![data];
        let mut total = 0;

        while let Some(item) = stack.pop() {
            // each time we deconstruct a data we add 4 memory units
            total += 4;
            match item {
                Data::Constr(c) => {
                    // note currently tag is not factored into cost of memory
                    stack.extend(c.fields.iter());
                }
                Data::Map(m) => {
                    for (k, v) in m.iter() {
                        stack.push(k);
                        stack.push(v);
                    }
                }
                Data::BigInt(i) => {
                    total += pallas_bigint_to_ex_mem(i);
                }
                Data::BoundedBytes(b) => {
                    total += Self::byte_string_to_ex_mem(b.deref());
                }
                Data::Array(a) => {
                    stack.extend(a.iter());
                }
            }
        }
        total
    }
    pub(super) fn data_node_count(&self) -> Result<i64, Error> {
        let data = self.unwrap_data()?;
        let mut stack = vec![data];
        let mut count = 0_i64;

        while let Some(item) = stack.pop() {
            count += 1;
            match item {
                Data::Constr(constr) => stack.extend(constr.fields.iter()),
                Data::Map(entries) => {
                    for (key, value) in entries.iter() {
                        stack.push(key);
                        stack.push(value);
                    }
                }
                Data::Array(items) => stack.extend(items.iter()),
                Data::BigInt(_) | Data::BoundedBytes(_) => {}
            }
        }

        Ok(count)
    }

    pub fn expect_type(&self, r#type: Type) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if constant_type == r#type {
            Ok(())
        } else {
            Err(Error::TypeMismatch(r#type, constant_type))
        }
    }

    pub fn expect_list(&self) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if matches!(constant_type, Type::List(_)) {
            Ok(())
        } else {
            Err(Error::ListTypeMismatch(constant_type))
        }
    }

    pub fn expect_pair(&self) -> Result<(), Error> {
        let constant: Constant = self.clone().try_into()?;

        let constant_type = Type::from(&constant);

        if matches!(constant_type, Type::Pair(_, _)) {
            Ok(())
        } else {
            Err(Error::PairTypeMismatch(constant_type))
        }
    }
}

impl TryFrom<Value> for Type {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let constant: Constant = value.try_into()?;

        let constant_type = Type::from(&constant);

        Ok(constant_type)
    }
}

impl TryFrom<&Value> for Type {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let constant: Constant = value.try_into()?;

        let constant_type = Type::from(&constant);

        Ok(constant_type)
    }
}

impl TryFrom<Value> for Constant {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Con(constant) => Ok(constant.as_ref().clone()),
            rest => Err(Error::NotAConstant(rest)),
        }
    }
}

impl TryFrom<&Value> for Constant {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Con(constant) => Ok(constant.as_ref().clone()),
            rest => Err(Error::NotAConstant(rest.clone())),
        }
    }
}

pub fn integer_log2(i: BigInt) -> i64 {
    integer_log2_ref(&i)
}

/// The base-2 logarithm of the magnitude of `i`, rounded down; 0 for 0.
pub(super) fn integer_log2_ref(i: &BigInt) -> i64 {
    match i.bits() {
        0 => 0,
        bits => bits as i64 - 1,
    }
}

/// The memory size of a Data integer, without converting it to a BigInt
/// unless it is a negative bignum.
fn pallas_bigint_to_ex_mem(n: &conway::BigInt) -> i64 {
    let bits = match n {
        conway::BigInt::Int(i) => {
            let magnitude = i128::from(*i).unsigned_abs();
            (u128::BITS - magnitude.leading_zeros()) as u64
        }
        conway::BigInt::BigUInt(bytes) => {
            let bytes: &[u8] = bytes;
            match bytes.iter().position(|b| *b != 0) {
                None => 0,
                Some(first) => {
                    (8 - bytes[first].leading_zeros()) as u64 + 8 * (bytes.len() - first - 1) as u64
                }
            }
        }
        conway::BigInt::BigNInt(_) => from_pallas_bigint(n).bits(),
    };

    if bits == 0 {
        1
    } else {
        ((bits as i64 - 1) / 64) + 1
    }
}

pub fn from_pallas_bigint(n: &conway::BigInt) -> BigInt {
    match n {
        conway::BigInt::Int(i) => i128::from(*i).into(),
        conway::BigInt::BigUInt(bytes) => BigInt::from_bytes_be(num_bigint::Sign::Plus, bytes),
        conway::BigInt::BigNInt(bytes) => BigInt::from_bytes_be(num_bigint::Sign::Minus, bytes) - 1,
    }
}

pub fn to_pallas_bigint(n: &BigInt) -> conway::BigInt {
    if let Some(i) = n.to_i128()
        && let Ok(i) = i.try_into()
    {
        let pallas_int: pallas_codec::utils::Int = i;
        return conway::BigInt::Int(pallas_int);
    }

    if n.is_positive() {
        let (_, bytes) = n.to_bytes_be();
        conway::BigInt::BigUInt(bytes.into())
    } else {
        // Note that this would break if n == 0
        // BUT n == 0 always fits into 64bits and hence would end up in the first branch.
        let n: BigInt = n + 1;
        let (_, bytes) = n.to_bytes_be();
        conway::BigInt::BigNInt(bytes.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Data, Type},
        data::tests::arb_plutus_data,
        machine::{
            runtime::BuiltinSemantics,
            value::{
                Value, from_pallas_bigint, integer_log2, pallas_bigint_to_ex_mem, to_pallas_bigint,
            },
        },
    };
    use num_bigint::BigInt;
    use pallas_primitives::{PlutusData, conway};
    use proptest::prelude::*;
    use std::rc::Rc;

    #[test]
    fn sizes_cost_as_words_of_their_magnitude() {
        let two_to = |n: usize| BigInt::from(1) << n;

        for (size, words) in [
            (BigInt::from(0), 0),
            (BigInt::from(1), 1),
            (BigInt::from(8), 1),
            (BigInt::from(9), 2),
            (BigInt::from(-9), 2),
            (BigInt::from(u64::MAX), 1 << 61),
            (two_to(64), 1 << 61),
            (two_to(64) + 1, (1 << 61) + 1),
            (two_to(65), 1 << 62),
            (two_to(66) - 8, i64::MAX),
            // Saturating.
            (two_to(66), i64::MAX),
            (-two_to(70), i64::MAX),
        ] {
            assert_eq!(
                Value::integer(size.clone()).cost_as_size().unwrap(),
                words,
                "{size}"
            );
        }
    }

    #[test]
    fn to_ex_mem_bigint() {
        let value = Value::Con(Constant::Integer(1.into()).into());

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(Constant::Integer(42.into()).into());

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(
            Constant::Integer(BigInt::parse_bytes("18446744073709551615".as_bytes(), 10).unwrap())
                .into(),
        );

        assert_eq!(value.to_ex_mem(), 1);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("999999999999999999999999999999".as_bytes(), 10).unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105726".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105727".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105728".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("170141183460469231731687303715884105729".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 2);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("340282366920938463463374607431768211458".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 3);

        let value = Value::Con(
            Constant::Integer(
                BigInt::parse_bytes("999999999999999999999999999999999999999999".as_bytes(), 10)
                    .unwrap(),
            )
            .into(),
        );

        assert_eq!(value.to_ex_mem(), 3);

        let value =
            Value::Con(Constant::Integer(BigInt::parse_bytes("999999999999999999999999999999999999999999999999999999999999999999999999999999999999".as_bytes(), 10).unwrap()).into());

        assert_eq!(value.to_ex_mem(), 5);
    }

    #[test]
    fn data_integer_ex_mem_matches_integer_ex_mem() {
        let mut samples: Vec<BigInt> = vec![0.into(), 1.into(), (-1).into()];

        for shift in [7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 200] {
            let power: BigInt = BigInt::from(1) << shift;
            for n in [&power - 1, power.clone(), &power + 1] {
                samples.push(-&n);
                samples.push(n);
            }
        }

        for n in samples {
            let pallas = to_pallas_bigint(&n);
            assert_eq!(from_pallas_bigint(&pallas), n);
            assert_eq!(
                pallas_bigint_to_ex_mem(&pallas),
                Value::integer(n.clone()).to_ex_mem(),
                "{n}"
            );
        }

        // Non-canonical big naturals with leading zero bytes.
        let padded = conway::BigInt::BigUInt(vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0].into());
        assert_eq!(pallas_bigint_to_ex_mem(&padded), 2);
        let zero = conway::BigInt::BigUInt(vec![0, 0].into());
        assert_eq!(pallas_bigint_to_ex_mem(&zero), 1);
    }

    #[test]
    fn integer_log2_oracle() {
        // Values come from the Haskell implementation
        assert_eq!(integer_log2(0.into()), 0);
        assert_eq!(integer_log2(1.into()), 0);
        assert_eq!(integer_log2(42.into()), 5);
        assert_eq!(
            integer_log2(BigInt::parse_bytes("18446744073709551615".as_bytes(), 10).unwrap()),
            63
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("999999999999999999999999999999".as_bytes(), 10).unwrap()
            ),
            99
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105726".as_bytes(), 10)
                    .unwrap()
            ),
            126
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105727".as_bytes(), 10)
                    .unwrap()
            ),
            126
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("170141183460469231731687303715884105728".as_bytes(), 10)
                    .unwrap()
            ),
            127
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("340282366920938463463374607431768211458".as_bytes(), 10)
                    .unwrap()
            ),
            128
        );
        assert_eq!(
            integer_log2(
                BigInt::parse_bytes("999999999999999999999999999999999999999999".as_bytes(), 10)
                    .unwrap()
            ),
            139
        );
        assert_eq!(
            integer_log2(BigInt::parse_bytes("999999999999999999999999999999999999999999999999999999999999999999999999999999999999".as_bytes(), 10).unwrap()),
            279
        );
    }

    #[test]
    fn to_ex_mem_counts_nested_constants_iteratively() {
        let nested = Constant::ProtoPair(
            Type::Integer,
            Type::List(Type::String.into()),
            Rc::new(Constant::Integer((1_i128 << 64).into())),
            Rc::new(Constant::ProtoList(
                Type::String,
                vec![
                    Constant::String("abcd".to_string()).into(),
                    Constant::String("é".to_string()).into(),
                    Constant::ByteString(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).into(),
                ]
                .into(),
            )),
        );

        let value = Value::Con(nested.into());

        assert_eq!(value.to_ex_mem_with_semantics(BuiltinSemantics::C), 9);
        assert_eq!(value.to_ex_mem_with_semantics(BuiltinSemantics::D), 5);
    }

    /// Data sizing as it was computed from `PlutusData` before `Data` existed.
    fn pallas_data_ex_mem(data: &PlutusData) -> (i64, i64) {
        let mut stack = vec![data];
        let (mut size, mut nodes) = (0, 0);

        while let Some(item) = stack.pop() {
            size += 4;
            nodes += 1;
            match item {
                PlutusData::Constr(c) => stack.extend(c.fields.iter()),
                PlutusData::Map(m) => {
                    for (k, v) in m.iter() {
                        stack.push(k);
                        stack.push(v);
                    }
                }
                PlutusData::BigInt(i) => size += pallas_bigint_to_ex_mem(i),
                PlutusData::BoundedBytes(b) => size += Value::byte_string_to_ex_mem(b),
                PlutusData::Array(a) => stack.extend(a.iter()),
            }
        }

        (size, nodes)
    }

    proptest! {
        #[test]
        fn sizes_data_like_pallas(pallas in arb_plutus_data()) {
            let value = Value::data(Data::from(&pallas));
            let (size, nodes) = pallas_data_ex_mem(&pallas);

            prop_assert_eq!(value.to_ex_mem(), size);
            prop_assert_eq!(value.data_node_count().unwrap(), nodes);
        }
    }
}
