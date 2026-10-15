use std::{fmt::Display, ptr::NonNull, rc::Rc};

use crate::ast::{Constant, NamedDeBruijn, Term, Type};
use num_traits::ToPrimitive;

pub mod cost_model;
mod discharge;
mod error;
pub mod eval_result;
pub mod runtime;
pub mod value;

use cost_model::{ExBudget, StepKind};
pub use error::Error;
use pallas_primitives::conway::Language;

use self::{
    cost_model::CostModel,
    runtime::{
        BuiltinRuntime, BuiltinSemantics, DIJKSTRA_PROTOCOL_VERSION, VAN_ROSSEM_PROTOCOL_VERSION,
    },
    value::{Env, Value},
};

enum MachineState {
    Return(Value),
    Compute(Env, TermRef),
    Done(Term<NamedDeBruijn>),
}

/// A pending continuation. The machine keeps them on a stack, with the next
/// one to resume on top.
enum Frame {
    AwaitArg(Value),
    AwaitFunTerm(Env, Rc<Term<NamedDeBruijn>>),
    AwaitFunValue(Value),
    Force,
    /// The `constr` term being evaluated, the index of the next field to
    /// compute, and the fields computed so far.
    Constr(Env, TermRef, Vec<Value>),
    /// The `case` term whose scrutinee is being evaluated.
    Cases(Env, TermRef),
}

/// A term to evaluate. The fields of `constr` and the branches of `case`
/// are stored inline rather than behind their own `Rc`, so evaluation
/// addresses them through the `Rc` that owns them instead of cloning each
/// one into a new allocation whenever it is entered.
#[derive(Clone)]
struct TermRef {
    owner: Rc<Term<NamedDeBruijn>>,
    term: NonNull<Term<NamedDeBruijn>>,
}

impl TermRef {
    #[inline(always)]
    fn get(&self) -> &Term<NamedDeBruijn> {
        // SAFETY: `term` points into the allocation kept alive by `owner`.
        // Terms behind an `Rc` are never mutated while shared, and the
        // machine never takes `&mut` to them, so the pointee is stable.
        unsafe { self.term.as_ref() }
    }

    /// The `n`th inline child of this `constr` or `case` term.
    #[inline(always)]
    fn child(&self, n: usize) -> TermRef {
        let (Term::Constr {
            fields: children, ..
        }
        | Term::Case {
            branches: children, ..
        }) = self.get()
        else {
            unreachable!("only constr and case terms have inline children")
        };

        TermRef {
            owner: self.owner.clone(),
            term: NonNull::from(&children[n]),
        }
    }
}

impl From<Rc<Term<NamedDeBruijn>>> for TermRef {
    #[inline(always)]
    fn from(owner: Rc<Term<NamedDeBruijn>>) -> Self {
        let term = NonNull::from(owner.as_ref());

        TermRef { owner, term }
    }
}

pub const TERM_COUNT: usize = 9;
// Builtin tags are sparse; this is the highest tag plus one for debug counters.
pub const BUILTIN_COUNT: usize = 101;

#[derive(Debug, Clone)]
pub enum Trace {
    Log(String),
    Label(String),
}

impl Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trace::Log(log) => f.write_str(log),
            Trace::Label(label) => f.write_str(label),
        }
    }
}

impl Trace {
    pub fn unwrap_log(self) -> Option<String> {
        match self {
            Trace::Log(log) => Some(log),
            _ => None,
        }
    }

    pub fn unwrap_label(self) -> Option<String> {
        match self {
            Trace::Label(label) => Some(label),
            _ => None,
        }
    }
}

pub struct Machine {
    costs: CostModel,
    pub ex_budget: ExBudget,
    slippage: u32,
    unbudgeted_steps: [u32; 10],
    pub traces: Vec<Trace>,
    pub spend_counter: Option<[i64; (TERM_COUNT + BUILTIN_COUNT) * 2]>,
    semantics: BuiltinSemantics,
    language: Language,
    protocol_major_version: u16,
    frames: Vec<Frame>,
}

impl Machine {
    /// A machine for `version` under the latest supported protocol version.
    pub fn new(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        Self::build(
            version,
            VAN_ROSSEM_PROTOCOL_VERSION,
            costs,
            initial_budget,
            slippage,
            None,
        )
    }

    pub fn new_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        Self::build(
            version,
            protocol_major_version,
            costs,
            initial_budget,
            slippage,
            None,
        )
    }

    pub fn new_debug(
        version: Language,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        Self::build(
            version,
            VAN_ROSSEM_PROTOCOL_VERSION,
            costs,
            initial_budget,
            slippage,
            Some([0; (TERM_COUNT + BUILTIN_COUNT) * 2]),
        )
    }

    pub fn new_debug_with_protocol(
        version: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
    ) -> Machine {
        Self::build(
            version,
            protocol_major_version,
            costs,
            initial_budget,
            slippage,
            Some([0; (TERM_COUNT + BUILTIN_COUNT) * 2]),
        )
    }

    fn build(
        language: Language,
        protocol_major_version: u16,
        costs: CostModel,
        initial_budget: ExBudget,
        slippage: u32,
        spend_counter: Option<[i64; (TERM_COUNT + BUILTIN_COUNT) * 2]>,
    ) -> Machine {
        let semantics =
            BuiltinSemantics::for_language_and_protocol(&language, protocol_major_version);

        Machine {
            costs,
            ex_budget: initial_budget,
            slippage,
            unbudgeted_steps: [0; 10],
            traces: vec![],
            spend_counter,
            semantics,
            language,
            protocol_major_version,
            frames: vec![],
        }
    }

    pub fn run(&mut self, term: Term<NamedDeBruijn>) -> Result<Term<NamedDeBruijn>, Error> {
        self.check_term(&term)?;

        let startup_budget = self.costs.machine_costs.get(StepKind::StartUp);

        self.spend_budget(startup_budget)?;

        let result = self.evaluate(term);

        // A failed evaluation leaves its pending frames on the stack; drop them
        // now instead of keeping what they hold alive with the machine.
        self.frames.clear();

        result
    }

    fn evaluate(&mut self, term: Term<NamedDeBruijn>) -> Result<Term<NamedDeBruijn>, Error> {
        use MachineState::*;

        let mut state = Compute(Env::default(), Rc::new(term).into());

        loop {
            state = match state {
                Compute(env, t) => self.compute(env, t)?,
                Return(value) => self.return_compute(value)?,
                Done(t) => {
                    return Ok(t);
                }
            };
        }
    }

    /// Reject, before any budget is spent, terms the ledger would refuse to
    /// evaluate: open terms (the ledger scope-checks every script it runs) and,
    /// before protocol version 11 (Van Rossem), builtins and constant types the
    /// ledger language does not have yet (see `DefaultFunction::introduced_in`).
    /// From Van Rossem onwards everything this machine knows about is available
    /// in every ledger language. Both are whole-program well-formedness rules:
    /// occurrences in latent code (under lambdas, delays or untaken branches)
    /// are rejected too, except that before protocol version 12 (Dijkstra) the
    /// ledger's scope check does not look inside `constr` and `case` (Plutus'
    /// `checkScope NoCaseConstr`), so a free variable there only fails the
    /// program if it is evaluated.
    fn check_term(&self, term: &Term<NamedDeBruijn>) -> Result<(), Error> {
        let check_availability = self.protocol_major_version < VAN_ROSSEM_PROTOCOL_VERSION;
        let scope_case_constr = self.protocol_major_version >= DIJKSTRA_PROTOCOL_VERSION;

        // Each term is paired with the number of lambdas enclosing it, or with
        // `None` when its variables are not scope-checked.
        let mut stack = vec![(term, Some(0_u64))];

        while let Some((term, depth)) = stack.pop() {
            match term {
                Term::Var(name) => {
                    let index = name.index.word();

                    if depth.is_some_and(|depth| index == 0 || index > depth) {
                        return Err(Error::OpenTermEvaluated(term.clone()));
                    }
                }
                Term::Lambda { body, .. } => {
                    stack.push((body.as_ref(), depth.map(|depth| depth + 1)))
                }
                Term::Delay(body) | Term::Force(body) => stack.push((body.as_ref(), depth)),
                Term::Apply { function, argument } => {
                    stack.push((function.as_ref(), depth));
                    stack.push((argument.as_ref(), depth));
                }
                Term::Constr { fields, .. } => {
                    let depth = depth.filter(|_| scope_case_constr);

                    stack.extend(fields.iter().map(|field| (field, depth)));
                }
                Term::Case { constr, branches } => {
                    let depth = depth.filter(|_| scope_case_constr);

                    stack.push((constr.as_ref(), depth));
                    stack.extend(branches.iter().map(|branch| (branch, depth)));
                }
                Term::Constant(constant) if check_availability => {
                    if Constant::contains_value(constant) {
                        return Err(Error::ValueConstantNotAvailable);
                    }
                    if Constant::contains_array(constant) {
                        return Err(Error::ArrayConstantNotAvailable);
                    }
                }
                Term::Builtin(fun) if check_availability => {
                    if !fun.is_available_in(&self.language, self.protocol_major_version) {
                        return Err(Error::BuiltinNotAvailable(*fun));
                    }
                }
                Term::Constant(_) | Term::Builtin(_) | Term::Error => {}
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn compute(&mut self, env: Env, term: TermRef) -> Result<MachineState, Error> {
        match term.get() {
            Term::Var(name) => {
                self.step_and_maybe_spend(StepKind::Var)?;

                let val = self.lookup_var(name.as_ref(), &env)?;

                Ok(MachineState::Return(val))
            }
            Term::Delay(body) => {
                self.step_and_maybe_spend(StepKind::Delay)?;

                Ok(MachineState::Return(Value::Delay(body.clone(), env)))
            }
            Term::Lambda {
                parameter_name,
                body,
            } => {
                self.step_and_maybe_spend(StepKind::Lambda)?;

                Ok(MachineState::Return(Value::Lambda {
                    parameter_name: parameter_name.clone(),
                    body: body.clone(),
                    env,
                }))
            }
            Term::Apply { function, argument } => {
                self.step_and_maybe_spend(StepKind::Apply)?;

                self.frames
                    .push(Frame::AwaitFunTerm(env.clone(), argument.clone()));

                Ok(MachineState::Compute(env, function.clone().into()))
            }
            Term::Constant(x) => {
                self.step_and_maybe_spend(StepKind::Constant)?;

                Ok(MachineState::Return(Value::Con(x.clone())))
            }
            Term::Force(body) => {
                self.step_and_maybe_spend(StepKind::Force)?;

                self.frames.push(Frame::Force);

                Ok(MachineState::Compute(env, body.clone().into()))
            }
            Term::Error => Err(Error::EvaluationFailure),
            Term::Builtin(fun) => {
                self.step_and_maybe_spend(StepKind::Builtin)?;

                let runtime: BuiltinRuntime = (*fun).into();

                Ok(MachineState::Return(Value::Builtin { fun: *fun, runtime }))
            }
            Term::Constr { tag, fields } => {
                self.step_and_maybe_spend(StepKind::Constr)?;

                match fields.first() {
                    Some(_) => {
                        let field = term.child(0);
                        let resolved_fields = Vec::with_capacity(fields.len());

                        self.frames
                            .push(Frame::Constr(env.clone(), term, resolved_fields));

                        Ok(MachineState::Compute(env, field))
                    }
                    None => Ok(MachineState::Return(Value::Constr {
                        tag: *tag,
                        fields: Rc::new(vec![]),
                    })),
                }
            }
            Term::Case { constr, .. } => {
                self.step_and_maybe_spend(StepKind::Case)?;

                let constr = constr.clone();

                self.frames.push(Frame::Cases(env.clone(), term));

                Ok(MachineState::Compute(env, constr.into()))
            }
        }
    }

    #[inline(always)]
    fn return_compute(&mut self, value: Value) -> Result<MachineState, Error> {
        let Some(frame) = self.frames.pop() else {
            if self.unbudgeted_steps[9] > 0 {
                self.spend_unbudgeted_steps()?;
            }

            let term = discharge::value_as_term(value);

            return Ok(MachineState::Done(term));
        };

        match frame {
            Frame::Force => self.force_evaluate(value),
            Frame::AwaitFunTerm(arg_env, arg) => {
                self.frames.push(Frame::AwaitArg(value));

                Ok(MachineState::Compute(arg_env, arg.into()))
            }
            Frame::AwaitArg(fun) => self.apply_evaluate(fun, value),
            Frame::AwaitFunValue(arg) => self.apply_evaluate(value, arg),
            Frame::Constr(env, term, mut resolved_fields) => {
                let Term::Constr { tag, fields } = term.get() else {
                    unreachable!("FrameConstr always holds a constr term")
                };

                resolved_fields.push(value);

                let next = resolved_fields.len();

                match fields.get(next) {
                    Some(_) => {
                        let field = term.child(next);

                        self.frames
                            .push(Frame::Constr(env.clone(), term, resolved_fields));

                        Ok(MachineState::Compute(env, field))
                    }
                    None => Ok(MachineState::Return(Value::Constr {
                        tag: *tag,
                        fields: Rc::new(resolved_fields),
                    })),
                }
            }
            Frame::Cases(env, term) => self.case_evaluate(env, term, value),
        }
    }

    #[inline(always)]
    fn case_evaluate(
        &mut self,
        env: Env,
        term: TermRef,
        value: Value,
    ) -> Result<MachineState, Error> {
        let Term::Case { branches, .. } = term.get() else {
            unreachable!("FrameCases always holds a case term")
        };

        match value {
            // A tag beyond the platform's index range has no branch, as in
            // the reference machine (which checks it against Int's maxBound).
            Value::Constr { tag, fields } => match usize::try_from(tag)
                .ok()
                .filter(|&index| index < branches.len())
            {
                Some(index) => {
                    self.transfer_arg_stack(Rc::unwrap_or_clone(fields));

                    Ok(MachineState::Compute(env, term.child(index)))
                }
                None => Err(Error::MissingCaseBranch(
                    branches.to_vec(),
                    Value::Constr { tag, fields },
                )),
            },
            Value::Con(constant) => {
                if !self.semantics.is_van_rossem() {
                    return Err(Error::NonConstrScrutinized(Value::Con(constant)));
                }

                let (tag, fields, max_branches) = match constant.as_ref() {
                    Constant::Unit => (0, vec![], 1),
                    Constant::Bool(false) => (0, vec![], 2),
                    Constant::Bool(true) => (1, vec![], 2),
                    Constant::Integer(integer) => {
                        let Some(tag) = integer.to_usize() else {
                            return Err(Error::MissingCaseBranch(
                                branches.to_vec(),
                                Value::Con(constant),
                            ));
                        };

                        (tag, vec![], usize::MAX)
                    }
                    Constant::ProtoList(_, items) if items.is_empty() => (1, vec![], 2),
                    Constant::ProtoList(item_type, items) => {
                        let head = items[0].clone();
                        let tail = Constant::ProtoList(
                            item_type.clone(),
                            items.skip(1).expect("list is non-empty"),
                        );

                        (0, vec![Value::Con(head), Value::Con(tail.into())], 2)
                    }
                    Constant::ProtoPair(_, _, first, second) => (
                        0,
                        vec![Value::Con(first.clone()), Value::Con(second.clone())],
                        1,
                    ),
                    _ => return Err(Error::NonConstrScrutinized(Value::Con(constant))),
                };

                if branches.len() > max_branches {
                    return Err(Error::MissingCaseBranch(
                        branches.to_vec(),
                        Value::Con(constant),
                    ));
                }

                match branches.get(tag) {
                    Some(_) => {
                        self.transfer_arg_stack(fields);

                        Ok(MachineState::Compute(env, term.child(tag)))
                    }
                    None => Err(Error::MissingCaseBranch(
                        branches.to_vec(),
                        Value::Con(constant),
                    )),
                }
            }
            v => Err(Error::NonConstrScrutinized(v)),
        }
    }

    #[inline(always)]
    fn force_evaluate(&mut self, value: Value) -> Result<MachineState, Error> {
        match value {
            Value::Delay(body, env) => Ok(MachineState::Compute(env, body.into())),
            Value::Builtin { fun, mut runtime } => {
                if runtime.needs_force() {
                    runtime.consume_force();

                    let res = if runtime.is_ready() {
                        self.eval_builtin_app(runtime)?
                    } else {
                        Value::Builtin { fun, runtime }
                    };

                    Ok(MachineState::Return(res))
                } else {
                    let term = discharge::value_as_term(Value::Builtin { fun, runtime });

                    Err(Error::BuiltinTermArgumentExpected(term))
                }
            }
            rest => Err(Error::NonPolymorphicInstantiation(rest)),
        }
    }

    #[inline(always)]
    fn apply_evaluate(&mut self, function: Value, argument: Value) -> Result<MachineState, Error> {
        match function {
            Value::Lambda { body, mut env, .. } => {
                env.push(argument);

                Ok(MachineState::Compute(env, body.into()))
            }
            Value::Builtin { fun, runtime } => {
                if runtime.is_arrow() && !runtime.needs_force() {
                    let mut runtime = runtime;

                    runtime.push(argument)?;

                    let res = if runtime.is_ready() {
                        self.eval_builtin_app(runtime)?
                    } else {
                        Value::Builtin { fun, runtime }
                    };

                    Ok(MachineState::Return(res))
                } else {
                    let term = discharge::value_as_term(Value::Builtin { fun, runtime });

                    Err(Error::UnexpectedBuiltinTermArgument(term))
                }
            }
            rest => Err(Error::NonFunctionalApplication(rest, argument)),
        }
    }

    fn eval_builtin_app(&mut self, runtime: BuiltinRuntime) -> Result<Value, Error> {
        runtime.unlift_arguments(self.semantics)?;

        let cost = runtime.to_ex_budget(&self.costs.builtin_costs, self.semantics)?;

        self.spend_budget(cost)?;

        if let Some(counter) = &mut self.spend_counter {
            let i = (runtime.fun as usize + TERM_COUNT) * 2;

            counter[i] += cost.mem;
            counter[i + 1] += cost.cpu;
        }

        runtime.call(self.semantics, &mut self.traces)
    }

    fn lookup_var(&mut self, name: &NamedDeBruijn, env: &Env) -> Result<Value, Error> {
        env.get(usize::from(name.index))
            .cloned()
            .ok_or_else(|| Error::OpenTermEvaluated(Term::Var(name.clone().into())))
    }

    /// Pushes the fields of a scrutinised constructor as pending arguments,
    /// the first field on top.
    fn transfer_arg_stack(&mut self, args: Vec<Value>) {
        self.frames
            .extend(args.into_iter().rev().map(Frame::AwaitFunValue));
    }

    fn step_and_maybe_spend(&mut self, step: StepKind) -> Result<(), Error> {
        let index = step as u8;
        self.unbudgeted_steps[index as usize] += 1;
        self.unbudgeted_steps[9] += 1;

        if self.unbudgeted_steps[9] >= self.slippage {
            self.spend_unbudgeted_steps()?;
        }

        Ok(())
    }

    fn spend_unbudgeted_steps(&mut self) -> Result<(), Error> {
        for i in 0..self.unbudgeted_steps.len() - 1 {
            let mut unspent_step_budget =
                self.costs.machine_costs.get(StepKind::try_from(i as u8)?);

            unspent_step_budget.occurrences(self.unbudgeted_steps[i] as i64);

            self.spend_budget(unspent_step_budget)?;

            self.unbudgeted_steps[i] = 0;

            if let Some(counter) = &mut self.spend_counter {
                counter[i * 2] = counter[i * 2].saturating_add(unspent_step_budget.mem);
                counter[i * 2 + 1] = counter[i * 2 + 1].saturating_add(unspent_step_budget.cpu);
            }
        }

        self.unbudgeted_steps[9] = 0;

        Ok(())
    }

    fn spend_budget(&mut self, spend_budget: ExBudget) -> Result<(), Error> {
        // Saturate like the reference `SatInt` budgeting: a builtin costed at
        // `maxBound` must exhaust the budget rather than overflow it.
        self.ex_budget.mem = self.ex_budget.mem.saturating_sub(spend_budget.mem);
        self.ex_budget.cpu = self.ex_budget.cpu.saturating_sub(spend_budget.cpu);

        if self.ex_budget.mem < 0 || self.ex_budget.cpu < 0 {
            Err(Error::OutOfExError(self.ex_budget))
        } else {
            Ok(())
        }
    }
}

impl From<&Constant> for Type {
    fn from(constant: &Constant) -> Self {
        match constant {
            Constant::Integer(_) => Type::Integer,
            Constant::ByteString(_) => Type::ByteString,
            Constant::String(_) => Type::String,
            Constant::Unit => Type::Unit,
            Constant::Bool(_) => Type::Bool,
            Constant::ProtoList(t, _) => Type::List(Rc::new(t.clone())),
            Constant::ProtoPair(t1, t2, _, _) => {
                Type::Pair(Rc::new(t1.clone()), Rc::new(t2.clone()))
            }
            Constant::Data(_) => Type::Data,
            Constant::ProtoArray(t, _) => Type::Array(Rc::new(t.clone())),
            Constant::Value(_) => Type::Value,
            Constant::Bls12_381G1Element(_) => Type::Bls12_381G1Element,
            Constant::Bls12_381G2Element(_) => Type::Bls12_381G2Element,
            Constant::Bls12_381MlResult(_) => Type::Bls12_381MlResult,
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use std::rc::Rc;

    use super::{
        Error, Machine,
        cost_model::{CostModel, ExBudget},
        runtime::Compressable,
    };
    use crate::{
        ast::{Constant, NamedDeBruijn, Program, Term},
        builtins::DefaultFunction,
    };
    use pallas_primitives::conway::Language;

    #[test]
    fn open_terms_rejected_before_evaluation() {
        use super::runtime::{DIJKSTRA_PROTOCOL_VERSION, VAN_ROSSEM_PROTOCOL_VERSION};
        use crate::ast::DeBruijn;

        let var = |index: u64| {
            Term::Var(
                NamedDeBruijn {
                    text: "x".to_string(),
                    index: DeBruijn::from_word(index),
                }
                .into(),
            )
        };
        let lambda = |body: Term<NamedDeBruijn>| Term::Lambda {
            parameter_name: NamedDeBruijn {
                text: "y".to_string(),
                index: DeBruijn::new(0),
            }
            .into(),
            body: body.into(),
        };
        let unit = || Term::Constant(Constant::Unit.into());
        let untaken_branch = |branch: Term<NamedDeBruijn>| Term::Case {
            constr: Term::Constr {
                tag: 0,
                fields: vec![],
            }
            .into(),
            branches: vec![unit(), branch],
        };
        let eval = |term: Term<NamedDeBruijn>, protocol_major_version: u16| {
            Program {
                version: (1, 1, 0),
                term,
            }
            .eval_version_with_protocol(
                ExBudget::max(),
                &Language::PlutusV3,
                protocol_major_version,
            )
            .result()
        };

        // A free variable that evaluation would never reach, outside of any
        // constr or case: under a delay that is not forced, in an unused
        // argument, or under a lambda that is never applied.
        for term in [
            Term::Delay(var(1).into()),
            lambda(unit()).apply(Term::Delay(var(1).into())),
            lambda(var(1).apply(var(2))),
            lambda(var(1).apply(var(1 << 32))),
        ] {
            for protocol_major_version in [VAN_ROSSEM_PROTOCOL_VERSION, DIJKSTRA_PROTOCOL_VERSION] {
                assert!(
                    matches!(
                        eval(term.clone(), protocol_major_version),
                        Err(Error::OpenTermEvaluated(_))
                    ),
                    "{} at {protocol_major_version}",
                    term.to_pretty()
                );
            }
        }

        // Before Dijkstra the ledger does not scope-check inside constr and
        // case, so a free variable there, with any index, only fails the
        // program if evaluation reaches it (as checked against the Haskell
        // ledger API at plutus 1b03c165).
        for term in [
            untaken_branch(var(1)),
            untaken_branch(var(0)),
            untaken_branch(var(1 << 32)),
            untaken_branch(var(u64::MAX)),
            Term::Case {
                constr: Term::Constr {
                    tag: 0,
                    fields: vec![lambda(var(2))],
                }
                .into(),
                branches: vec![lambda(unit())],
            },
            lambda(unit()).apply(lambda(Term::Constr {
                tag: 0,
                fields: vec![var(4)],
            })),
        ] {
            assert_eq!(
                eval(term.clone(), VAN_ROSSEM_PROTOCOL_VERSION),
                Ok(unit()),
                "{}",
                term.to_pretty()
            );
            assert!(
                matches!(
                    eval(term.clone(), DIJKSTRA_PROTOCOL_VERSION),
                    Err(Error::OpenTermEvaluated(_))
                ),
                "{}",
                term.to_pretty()
            );
        }

        // Evaluating one fails all the same.
        for term in [
            Term::Case {
                constr: Term::Constr {
                    tag: 1,
                    fields: vec![],
                }
                .into(),
                branches: vec![unit(), var(1 << 32)],
            },
            Term::Case {
                constr: var(1).into(),
                branches: vec![unit()],
            },
        ] {
            assert!(
                matches!(
                    eval(term.clone(), VAN_ROSSEM_PROTOCOL_VERSION),
                    Err(Error::OpenTermEvaluated(_))
                ),
                "{}",
                term.to_pretty()
            );
        }

        // Closed, for comparison.
        assert!(eval(lambda(var(1)), DIJKSTRA_PROTOCOL_VERSION).is_ok());
    }

    #[test]
    fn van_rossem_features_unavailable_before_van_rossem() {
        let program = |term: Term<NamedDeBruijn>| Program {
            version: (1, 1, 0),
            term,
        };

        let builtin = Term::Builtin(DefaultFunction::LookupCoin);
        let constant: Term<NamedDeBruijn> =
            Term::Constant(Constant::Value(crate::ast::Value::empty()).into());
        // Availability is a whole-program rule: latent occurrences that
        // evaluation never reaches must be rejected too.
        let latent: Term<NamedDeBruijn> = Term::Lambda {
            parameter_name: NamedDeBruijn {
                text: "x".to_string(),
                index: 0.into(),
            }
            .into(),
            body: Term::Builtin(DefaultFunction::ScaleValue).into(),
        };
        // Public AST callers can construct type-inconsistent constants even
        // though Flat decoding cannot. Inspect both declared types and actual
        // elements so this cannot bypass protocol availability checks.
        let array_builtin = Term::Builtin(DefaultFunction::IndexArray);
        let array_constant: Term<NamedDeBruijn> = Term::Constant(
            Constant::ProtoArray(
                crate::ast::Type::Integer,
                vec![Rc::new(Constant::Integer(1.into()))].into(),
            )
            .into(),
        );
        let inconsistent: Term<NamedDeBruijn> = Term::Constant(
            Constant::ProtoList(
                crate::ast::Type::Integer,
                vec![Rc::new(Constant::Value(crate::ast::Value::empty()))].into(),
            )
            .into(),
        );

        for (language, protocol) in [
            (Language::PlutusV3, 10),
            (Language::PlutusV2, 10),
            (Language::PlutusV1, 10),
        ] {
            for term in [array_builtin.clone(), array_constant.clone()] {
                let result = program(term)
                    .eval_version_with_protocol(ExBudget::default(), &language, protocol)
                    .result();
                assert!(
                    matches!(
                        result,
                        Err(Error::BuiltinNotAvailable(_) | Error::ArrayConstantNotAvailable)
                    ),
                    "expected arrays to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
                );
            }

            let result = program(builtin.clone())
                .eval_version_with_protocol(ExBudget::default(), &language, protocol)
                .result();
            assert!(
                matches!(result, Err(Error::BuiltinNotAvailable(_))),
                "expected lookupCoin to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
            );

            let result = program(constant.clone())
                .eval_version_with_protocol(ExBudget::default(), &language, protocol)
                .result();
            assert!(
                matches!(result, Err(Error::ValueConstantNotAvailable)),
                "expected Value constant to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
            );

            let result = program(inconsistent.clone())
                .eval_version_with_protocol(ExBudget::default(), &language, protocol)
                .result();
            assert!(
                matches!(result, Err(Error::ValueConstantNotAvailable)),
                "expected nested Value constant to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
            );

            let result = program(latent.clone())
                .eval_version_with_protocol(ExBudget::default(), &language, protocol)
                .result();
            assert!(
                matches!(result, Err(Error::BuiltinNotAvailable(_))),
                "expected latent scaleValue to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
            );

            // Direct calls through the public runtime API are gated too.
            let semantics =
                super::runtime::BuiltinSemantics::for_language_and_protocol(&language, protocol);
            let result = DefaultFunction::UnValueData.call(
                semantics,
                &[super::value::Value::Con(
                    Constant::Data(crate::ast::Data::map(vec![])).into(),
                )],
                &mut vec![],
            );
            assert!(
                matches!(result, Err(Error::BuiltinNotAvailable(_))),
                "expected direct unValueData call to be unavailable for {language:?} at protocol {protocol}, got {result:?}"
            );
        }

        // From Van Rossem onwards every builtin and constant type is available
        // in every ledger language.
        for language in [Language::PlutusV1, Language::PlutusV2, Language::PlutusV3] {
            for term in [
                builtin.clone(),
                constant.clone(),
                latent.clone(),
                array_builtin.clone(),
                array_constant.clone(),
            ] {
                let result = program(term)
                    .eval_version_with_protocol(ExBudget::default(), &language, 11)
                    .result();
                assert!(result.is_ok(), "{language:?}: {result:?}");
            }
        }
    }

    #[test]
    fn exp_mod_integer_modulus_bound_matches_plutus() {
        let modulus = |m: BigInt| {
            let program = crate::parser::program(&format!(
                "(program 1.1.0 [(builtin expModInteger) (con integer 2) (con integer 3) (con integer {m})])"
            ))
            .unwrap();

            Program::<NamedDeBruijn>::try_from(program)
                .unwrap()
                .eval_version_with_protocol(
                    ExBudget::max(),
                    &Language::PlutusV3,
                    super::VAN_ROSSEM_PROTOCOL_VERSION,
                )
        };
        let two = BigInt::from(2);

        // Expected results and costs from the reference evaluator (plutus
        // 1b03c165, semantics variant E). A modulus beyond 2^8191 - 1 fails, but
        // only after being charged for: the cost comes before the denotation.
        let largest = modulus(two.pow(8191) - 1);
        assert_eq!(
            largest.result().unwrap(),
            Term::Constant(Constant::Integer(8.into()).into())
        );
        assert_eq!(
            largest.cost(),
            ExBudget {
                mem: 928,
                cpu: 901_087_765
            }
        );

        for (m, cost) in [
            (
                two.pow(8191),
                ExBudget {
                    mem: 228,
                    cpu: 900_975_765,
                },
            ),
            (
                two.pow(8192),
                ExBudget {
                    mem: 229,
                    cpu: 914_865_470,
                },
            ),
        ] {
            let result = modulus(m);
            assert!(
                matches!(result.result(), Err(Error::OutsideNaturalBounds(..))),
                "{:?}",
                result.result()
            );
            assert_eq!(result.cost(), cost);
        }
    }

    #[test]
    fn scope_checked_like_the_ledger_at_van_rossem() {
        use super::runtime::VAN_ROSSEM_PROTOCOL_VERSION;
        use crate::{ast::DeBruijn, data::Data};

        // Plutus V3 scripts as the Haskell ledger API serialises them (flat,
        // without the CBOR wrapper), each a lambda over its script context, with
        // what the ledger gives them at protocol version 11 applied to
        // `Constr 0 []`: whether they return unit, and their cost (plutus
        // 1b03c165, `mkTermToEvaluate` and `evaluateTerm counting`). A free
        // variable fails the script only if it is evaluated, before the step
        // that evaluates it is charged.
        let cases = [
            // case (constr 0 [\y -> x]) [\f -> ()], with x free
            (
                "free var under a lambda in a constr field",
                "010100298009001a4931",
                true,
                128100,
                900,
            ),
            (
                "free var as a constr field",
                "01010029800801249301",
                false,
                100,
                100,
            ),
            (
                "free var in an untaken case branch",
                "010100298005268011",
                true,
                96100,
                700,
            ),
            (
                "free var in the taken case branch",
                "010100298015268011",
                false,
                100,
                100,
            ),
            (
                "free var as the case scrutinee",
                "01010029002a4c01",
                false,
                100,
                100,
            ),
            (
                "index 2^32 in an untaken case branch",
                "01010029800526840404040081",
                true,
                96100,
                700,
            ),
            (
                "index 2^64-1 in an untaken case branch",
                "0101002980052687fffffffffffffffff809",
                true,
                96100,
                700,
            ),
            (
                "index 0 in an untaken case branch",
                "010100298005268001",
                true,
                96100,
                700,
            ),
            (
                "free var in a constr under an unapplied lambda",
                "010100232498a00200a1",
                true,
                112100,
                800,
            ),
        ];

        for (name, flat, returns_unit, cpu, mem) in cases {
            let bytes = hex::decode(flat).unwrap();
            let program = Program::<DeBruijn>::from_flat(&bytes).unwrap();

            assert_eq!(program.to_flat().unwrap(), bytes, "{name}");

            let result = Program::<NamedDeBruijn>::from(program)
                .apply_data(Data::constr(0, vec![]))
                .eval_version_with_protocol(
                    ExBudget::max(),
                    &Language::PlutusV3,
                    VAN_ROSSEM_PROTOCOL_VERSION,
                );
            let cost = result.cost();

            if returns_unit {
                assert_eq!(
                    result.result(),
                    Ok(Term::Constant(Constant::Unit.into())),
                    "{name}"
                );
            } else {
                assert!(
                    matches!(result.result(), Err(Error::OpenTermEvaluated(_))),
                    "{name}"
                );
            }
            assert_eq!((cost.cpu, cost.mem), (cpu, mem), "{name}");
        }
    }

    #[test]
    fn data_like_the_ledger_before_and_at_van_rossem() {
        use super::runtime::{PLOMIN_PROTOCOL_VERSION, VAN_ROSSEM_PROTOCOL_VERSION};
        use crate::{ast::DeBruijn, builtins::DefaultFunction};

        // Plutus V3 programs (flat, without the CBOR wrapper) with what the
        // Haskell ledger API makes of them at protocol versions 10 and 11: the
        // result, or none if evaluation fails, and the cost (plutus 1b03c165,
        // `mkTermToEvaluate` and `evaluateTerm counting`).
        //
        // Before protocol version 11, `constrData` takes any integer as the
        // index, which the Data builtins then handle like any other; from it,
        // only a 64-bit word. `serialiseData` encodes the value, choosing tags
        // and integer encodings afresh, and `equalsData` compares integers by
        // value, however the Data constants they are given are encoded.
        let cases = [
            (
                "constrData 2^64",
                "0101003374a904040404040404040400225eb801",
                [
                    (
                        Some("(con data (Constr 18446744073709551616 []))"),
                        102251,
                        632,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "constrData -1",
                "0101003374a9000a5eb801",
                [
                    (Some("(con data (Constr -1 []))"), 102251, 632),
                    (None, 100, 100),
                ],
            ),
            (
                "constrData 2^64-1",
                "0101003374a907f7fffffffffffffff81a5eb801",
                [
                    (
                        Some("(con data (Constr 18446744073709551615 []))"),
                        102251,
                        632,
                    ),
                    (
                        Some("(con data (Constr 18446744073709551615 []))"),
                        102251,
                        632,
                    ),
                ],
            ),
            (
                "serialiseData constrData 2^64",
                "010100376666e9520808080808080808080044bd7001",
                [
                    (
                        Some("(con bytestring #d86682c24901000000000000000080)"),
                        1943005,
                        840,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData constrData -1",
                "010100376666e9520014bd7001",
                [
                    (Some("(con bytestring #d866822080)"), 1943005, 840),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData constrData -2^64",
                "010100376666e9520ffffffffffffffffff034bd7001",
                [
                    (
                        Some("(con bytestring #d866823bffffffffffffffff80)"),
                        1943005,
                        840,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData constrData -2^64-1",
                "010100376666e9520818080808080808080044bd7001",
                [
                    (
                        Some("(con bytestring #d86682c34901000000000000000080)"),
                        1943005,
                        840,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData constrData 2^200 [I 1]",
                "010100376666e952080808080808080808080808080808080808080808080808080808080204bd708101010001",
                [
                    (
                        Some(
                            "(con bytestring #d86682c2581a01000000000000000000000000000000000000000000000000009f01ff)",
                        ),
                        3009565,
                        850,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData constrData 2^520",
                "010100376666e95208080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080084bd7001",
                [
                    (
                        Some(
                            "(con bytestring #d86682c2584201000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080)",
                        ),
                        1943005,
                        840,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "serialiseData listData [constrData 2^64]",
                "01010037666e9ccd5d019ba5482020202020202020200112f5c097ae01",
                [
                    (
                        Some("(con bytestring #9fd86682c24901000000000000000080ff)"),
                        3014467,
                        1612,
                    ),
                    (None, 100, 100),
                ],
            ),
            (
                "fstPair unConstrData constrData 2^64",
                "01010035573a6ea8cdd2a410101010101010101000897ae001",
                [
                    (Some("(con integer 18446744073709551616)"), 364734, 1296),
                    (None, 100, 100),
                ],
            ),
            (
                "fstPair unConstrData constrData -5",
                "01010035573a6ea8cdd2a401297ae001",
                [(Some("(con integer -5)"), 364734, 1296), (None, 100, 100)],
            ),
            (
                "equalsData constrData 2^64 itself",
                "0101003375e66e9520808080808080808080044bd7019ba5482020202020202020200112f5c1",
                [(Some("(con bool True)"), 1259666, 1465), (None, 100, 100)],
            ),
            (
                "equalsData constrData 2^64 vs 0",
                "0101003375e66e9520808080808080808080044bd7019ba5480012f5c1",
                [(Some("(con bool False)"), 1259666, 1465), (None, 100, 100)],
            ),
            (
                "equalsData constrData -1 vs 2^64-1",
                "0101003375e66e9520014bd7019ba5483fbfffffffffffffffc0d2f5c1",
                [(Some("(con bool False)"), 1259666, 1465), (None, 100, 100)],
            ),
            (
                "chooseData constrData 2^64",
                "010100333333574866e9520808080808080808080044bd702400490022400c9004240141",
                [(Some("(con integer 1)"), 404626, 1964), (None, 100, 100)],
            ),
            (
                "serialiseData constrData 127",
                "010100376666e9520fe014bd7001",
                [
                    (Some("(con bytestring #d9057880)"), 1943005, 840),
                    (Some("(con bytestring #d9057880)"), 1943005, 840),
                ],
            ),
            (
                "serialiseData constrData 128",
                "010100376666e952080024bd7001",
                [
                    (Some("(con bytestring #d86682188080)"), 1943005, 840),
                    (Some("(con bytestring #d86682188080)"), 1943005, 840),
                ],
            ),
            (
                "serialiseData 102-form index 0",
                "0101003766980105d8668200800001",
                [
                    (Some("(con bytestring #d87980)"), 1856854, 408),
                    (Some("(con bytestring #d87980)"), 1856854, 408),
                ],
            ),
            (
                "serialiseData 102-form index 6",
                "0101003766980105d8668206800001",
                [
                    (Some("(con bytestring #d87f80)"), 1856854, 408),
                    (Some("(con bytestring #d87f80)"), 1856854, 408),
                ],
            ),
            (
                "serialiseData 102-form index 127",
                "0101003766980106d86682187f800001",
                [
                    (Some("(con bytestring #d9057880)"), 1856854, 408),
                    (Some("(con bytestring #d9057880)"), 1856854, 408),
                ],
            ),
            (
                "serialiseData 102-form indefinite pair",
                "0101003766980105d8668200800001",
                [
                    (Some("(con bytestring #d87980)"), 1856854, 408),
                    (Some("(con bytestring #d87980)"), 1856854, 408),
                ],
            ),
            (
                "serialiseData bignum 1",
                "0101003766980103c241010001",
                [
                    (Some("(con bytestring #01)"), 2070166, 410),
                    (Some("(con bytestring #01)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData bignum 1 padded",
                "0101003766980104c24200010001",
                [
                    (Some("(con bytestring #01)"), 2070166, 410),
                    (Some("(con bytestring #01)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData bignum 0 empty",
                "0101003766980102c2400001",
                [
                    (Some("(con bytestring #00)"), 2070166, 410),
                    (Some("(con bytestring #00)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData negative bignum -1",
                "0101003766980103c341000001",
                [
                    (Some("(con bytestring #20)"), 2070166, 410),
                    (Some("(con bytestring #20)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData negative bignum -1 empty",
                "0101003766980102c3400001",
                [
                    (Some("(con bytestring #20)"), 2070166, 410),
                    (Some("(con bytestring #20)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData bignum 2^64",
                "010100376698010bc2490100000000000000000001",
                [
                    (
                        Some("(con bytestring #c249010000000000000000)"),
                        2283478,
                        412,
                    ),
                    (
                        Some("(con bytestring #c249010000000000000000)"),
                        2283478,
                        412,
                    ),
                ],
            ),
            (
                "serialiseData bignum 2^64 padded",
                "010100376698010cc24a000100000000000000000001",
                [
                    (
                        Some("(con bytestring #c249010000000000000000)"),
                        2283478,
                        412,
                    ),
                    (
                        Some("(con bytestring #c249010000000000000000)"),
                        2283478,
                        412,
                    ),
                ],
            ),
            (
                "serialiseData bignum 2^64-1",
                "010100376698010ac248ffffffffffffffff0001",
                [
                    (Some("(con bytestring #1bffffffffffffffff)"), 2070166, 410),
                    (Some("(con bytestring #1bffffffffffffffff)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData non-minimal int 1",
                "0101003766980101010001",
                [
                    (Some("(con bytestring #01)"), 2070166, 410),
                    (Some("(con bytestring #01)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData indefinite bytes",
                "010100376698010241010001",
                [
                    (Some("(con bytestring #4101)"), 2070166, 410),
                    (Some("(con bytestring #4101)"), 2070166, 410),
                ],
            ),
            (
                "serialiseData indefinite map",
                "0101003766980104bf0102ff0001",
                [
                    (Some("(con bytestring #a10102)"), 3989974, 428),
                    (Some("(con bytestring #a10102)"), 3989974, 428),
                ],
            ),
            (
                "equalsData bignum -1 vs -1",
                "0101003375e98103c34100004c0101200001",
                [
                    (Some("(con bool True)"), 1114643, 601),
                    (Some("(con bool True)"), 1114643, 601),
                ],
            ),
            (
                "equalsData bignum -1 empty vs -1",
                "0101003375e98102c340004c0101200001",
                [
                    (Some("(con bool True)"), 1114643, 601),
                    (Some("(con bool True)"), 1114643, 601),
                ],
            ),
            (
                "equalsData bignum -6 vs -5",
                "0101003375e98103c34105004c0101240001",
                [
                    (Some("(con bool False)"), 1114643, 601),
                    (Some("(con bool False)"), 1114643, 601),
                ],
            ),
            (
                "equalsData bignum -6 vs -6",
                "0101003375e98103c34105004c0101250001",
                [
                    (Some("(con bool True)"), 1114643, 601),
                    (Some("(con bool True)"), 1114643, 601),
                ],
            ),
            (
                "equalsData bignum 1 padded vs 1",
                "0101003375e98104c2420001004c0101010001",
                [
                    (Some("(con bool True)"), 1114643, 601),
                    (Some("(con bool True)"), 1114643, 601),
                ],
            ),
            (
                "equalsData 102-form 0 vs 121",
                "0101003375e98105d866820080004c0103d879800001",
                [
                    (Some("(con bool True)"), 1087364, 601),
                    (Some("(con bool True)"), 1087364, 601),
                ],
            ),
            (
                "unIData negative bignum -1 empty",
                "010100375a980102c3400001",
                [
                    (Some("(con integer -1)"), 68844, 432),
                    (Some("(con integer -1)"), 68844, 432),
                ],
            ),
        ];

        for (name, flat, outcomes) in cases {
            let bytes = hex::decode(flat).unwrap();
            let program =
                Program::<NamedDeBruijn>::from(Program::<DeBruijn>::from_flat(&bytes).unwrap());

            for (protocol_version, (expected, cpu, mem)) in
                [PLOMIN_PROTOCOL_VERSION, VAN_ROSSEM_PROTOCOL_VERSION]
                    .into_iter()
                    .zip(outcomes)
            {
                let result = program.clone().eval_version_with_protocol(
                    ExBudget::max(),
                    &Language::PlutusV3,
                    protocol_version,
                );
                let cost = result.cost();

                match expected {
                    Some(expected) => {
                        let expected =
                            crate::parser::program(&format!("(program 1.1.0 {expected})")).unwrap();
                        let expected = Program::<NamedDeBruijn>::try_from(expected).unwrap().term;

                        assert_eq!(
                            result.result(),
                            Ok(expected),
                            "{name} at {protocol_version}"
                        );
                    }
                    None => assert!(
                        matches!(
                            result.result(),
                            Err(Error::IntegerOutOfWord64Range(
                                DefaultFunction::ConstrData,
                                _
                            ))
                        ),
                        "{name} at {protocol_version}"
                    ),
                }
                assert_eq!(
                    (cost.cpu, cost.mem),
                    (cpu, mem),
                    "{name} at {protocol_version}"
                );
            }
        }
    }

    #[test]
    fn builtin_arguments_unlift_like_plutus() {
        let eval = |language: &Language, term: &str| {
            let version = match language {
                Language::PlutusV3 => "1.1.0",
                _ => "1.0.0",
            };
            let program = crate::parser::program(&format!("(program {version} {term})")).unwrap();

            Program::<NamedDeBruijn>::try_from(program)
                .unwrap()
                .eval_version_with_protocol(
                    ExBudget::max(),
                    language,
                    super::VAN_ROSSEM_PROTOCOL_VERSION,
                )
        };
        let bound = BigInt::from(2).pow(262143);
        let scalar_bound = BigInt::from(2).pow(4095);
        let bytes = |n: usize| format!("#{}", "00".repeat(n));
        let miller_loop = "[(builtin bls12_381_millerLoop) (con bls12_381_G1_element 0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb) (con bls12_381_G2_element 0x93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8)]";
        let g1 = "(con bls12_381_G1_element 0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb)";

        // Expected outcomes (`Err` naming the error) and costs from the
        // reference evaluator (plutus 1b03c165): semantics variant E for
        // PlutusV3 and D for PlutusV2. An argument that does not unlift fails
        // the application once the builtin is saturated, before it is costed;
        // a costing of 100/100 is the machine's startup cost alone.
        let v3 = Language::PlutusV3;
        let v2 = Language::PlutusV2;
        type Case<'a> = (&'a Language, String, Result<(), &'a str>, i64, i64);

        let cases: Vec<Case> = vec![
            (
                &v3,
                format!("[(builtin addInteger) (con integer {}) (con integer 1)]", &bound - 1),
                Ok(()),
                1_901_208,
                4697,
            ),
            (
                &v3,
                format!("[(builtin addInteger) (con integer {bound}) (con integer 1)]"),
                Err("IntegerOutOfBounds"),
                100,
                100,
            ),
            (
                &v3,
                format!("[(builtin lessThanInteger) (con integer {}) (con integer 1)]", -&bound - 1),
                Err("IntegerOutOfBounds"),
                100,
                100,
            ),
            (
                &v3,
                format!("[(builtin equalsInteger) (con integer {bound}) (con integer 1)]"),
                Ok(()),
                132_433,
                601,
            ),
            (
                &v2,
                format!("[(builtin addInteger) (con integer {bound}) (con integer 1)]"),
                Err("IntegerOutOfBounds"),
                100,
                100,
            ),
            (
                &v3,
                format!("[(builtin appendByteString) (con bytestring {}) (con bytestring #)]", bytes(65536)),
                Ok(()),
                1_498_489,
                8793,
            ),
            (
                &v3,
                format!("[(builtin appendByteString) (con bytestring {}) (con bytestring #)]", bytes(65537)),
                Err("ByteStringTooLong"),
                100,
                100,
            ),
            (
                &v3,
                format!("[(builtin lengthOfByteString) (con bytestring {})]", bytes(65537)),
                Ok(()),
                70_200,
                410,
            ),
            (
                &v3,
                "[(builtin consByteString) (con integer 256) (con bytestring #)]".to_string(),
                Err("ByteStringConsNotAByte"),
                100,
                100,
            ),
            (
                &v2,
                "[(builtin consByteString) (con integer 256) (con bytestring #)]".to_string(),
                Ok(()),
                152_288,
                602,
            ),
            (
                &v3,
                "[(builtin sliceByteString) (con integer 9223372036854775808) (con integer 1) (con bytestring #00)]".to_string(),
                Err("IntegerOutOfInt64Range"),
                100,
                100,
            ),
            (
                &v2,
                "[(builtin shiftByteString) (con bytestring #00) (con integer 9223372036854775808)]".to_string(),
                Err("IntegerOutOfInt64Range"),
                100,
                100,
            ),
            (
                &v3,
                "[(force (builtin indexArray)) (con (array integer) [1]) (con integer 9223372036854775808)]".to_string(),
                Err("IntegerOutOfInt64Range"),
                100,
                100,
            ),
            (
                &v3,
                "[(builtin constrData) (con integer 18446744073709551615) (con (list data) [])]".to_string(),
                Ok(()),
                102_251,
                632,
            ),
            (
                &v3,
                "[(builtin constrData) (con integer 18446744073709551616) (con (list data) [])]".to_string(),
                Err("IntegerOutOfWord64Range"),
                100,
                100,
            ),
            (
                &v3,
                "[(builtin replicateByte) (con integer 1) (con integer 256)]".to_string(),
                Err("OutsideByteBounds"),
                100,
                100,
            ),
            (
                &v3,
                "[(builtin integerToByteString) (con bool True) (con integer -1) (con integer 0)]".to_string(),
                Err("IntegerToByteStringNegativeSize"),
                1_322_707,
                101,
            ),
            (
                &v3,
                format!("[(builtin writeBits) (con bytestring {}) (con (list integer) [0]) (con bool True)]", bytes(4096)),
                Ok(()),
                412_093,
                1312,
            ),
            (
                &v3,
                format!("[(builtin writeBits) (con bytestring {}) (con (list integer) [0]) (con bool True)]", bytes(4097)),
                Err("ByteStringTooLong"),
                300_093,
                613,
            ),
            (
                &v3,
                "[(builtin writeBits) (con bytestring #00) (con (list bool) []) (con bool True)]".to_string(),
                Err("TypeMismatch"),
                100,
                100,
            ),
            (
                &v3,
                format!("[(builtin bls12_381_G1_scalarMul) (con integer {}) {g1}]", &scalar_bound - 1),
                Ok(()),
                77_080_658,
                618,
            ),
            (
                &v3,
                format!("[(builtin bls12_381_G1_scalarMul) (con integer {scalar_bound}) {g1}]"),
                Err("MsmScalarOutOfBounds"),
                77_000_658,
                118,
            ),
            // The second argument is evaluated, and charged for, before the
            // first one fails to unlift.
            (
                &v3,
                format!("[(builtin bls12_381_mulMlResult) (con unit ()) {miller_loop}]"),
                Err("TypeMismatch"),
                254_006_373,
                172,
            ),
            (
                &v3,
                "[(builtin addInteger) (lam x x) (con integer 1)]".to_string(),
                Err("NotAConstant"),
                100,
                100,
            ),
            // A partial application is never unlifted.
            (
                &v3,
                "[(lam f (con integer 1)) [(builtin addInteger) (con string \"x\")]]".to_string(),
                Ok(()),
                96_100,
                700,
            ),
            // Polymorphic constants are checked by the builtin, once costed.
            (
                &v3,
                "[(force (builtin headList)) (con integer 1)]".to_string(),
                Err("ListTypeMismatch"),
                83_250,
                132,
            ),
            (
                &v3,
                "[(force (builtin indexArray)) (con (list integer) [1]) (con integer 0)]".to_string(),
                Err("ArrayTypeMismatch"),
                232_110,
                132,
            ),
        ];

        for (language, term, expected, cpu, mem) in cases {
            let term_prefix = &term[..term.len().min(80)];
            let result = eval(language, &term);

            match (expected, result.result()) {
                (Ok(()), Ok(_)) => {}
                (Err(name), Err(e)) if format!("{e:?}").starts_with(name) => {}
                (expected, actual) => {
                    panic!("{term_prefix}: expected {expected:?}, got {actual:?}")
                }
            }
            assert_eq!(result.cost(), ExBudget { mem, cpu }, "{term_prefix}");
        }
    }

    #[test]
    fn add_big_ints() {
        let program: Program<NamedDeBruijn> = Program {
            version: (0, 0, 0),
            term: Term::Apply {
                function: Term::Apply {
                    function: Term::Builtin(DefaultFunction::AddInteger).into(),
                    argument: Term::Constant(Constant::Integer(i128::MAX.into()).into()).into(),
                }
                .into(),
                argument: Term::Constant(Constant::Integer(i128::MAX.into()).into()).into(),
            },
        };

        let eval_result = program.eval(ExBudget::default());

        let term = eval_result.result().unwrap();

        assert_eq!(
            term,
            Term::Constant(
                Constant::Integer(
                    Into::<BigInt>::into(i128::MAX) + Into::<BigInt>::into(i128::MAX)
                )
                .into()
            )
        );
    }

    #[test]
    fn divide_integer() {
        let make_program = |fun: DefaultFunction, n: i32, m: i32| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Apply {
                function: Term::Apply {
                    function: Term::Builtin(fun).into(),
                    argument: Term::Constant(Constant::Integer(n.into()).into()).into(),
                }
                .into(),
                argument: Term::Constant(Constant::Integer(m.into()).into()).into(),
            },
        };

        let test_data = vec![
            (DefaultFunction::DivideInteger, 8, 3, 2),
            (DefaultFunction::DivideInteger, 8, -3, -3),
            (DefaultFunction::DivideInteger, -8, 3, -3),
            (DefaultFunction::DivideInteger, -8, -3, 2),
            (DefaultFunction::QuotientInteger, 8, 3, 2),
            (DefaultFunction::QuotientInteger, 8, -3, -2),
            (DefaultFunction::QuotientInteger, -8, 3, -2),
            (DefaultFunction::QuotientInteger, -8, -3, 2),
            (DefaultFunction::RemainderInteger, 8, 3, 2),
            (DefaultFunction::RemainderInteger, 8, -3, 2),
            (DefaultFunction::RemainderInteger, -8, 3, -2),
            (DefaultFunction::RemainderInteger, -8, -3, -2),
            (DefaultFunction::ModInteger, 8, 3, 2),
            (DefaultFunction::ModInteger, 8, -3, -1),
            (DefaultFunction::ModInteger, -8, 3, 1),
            (DefaultFunction::ModInteger, -8, -3, -2),
        ];

        for (fun, n, m, result) in test_data {
            let eval_result = make_program(fun, n, m).eval(ExBudget::default());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant(Constant::Integer(result.into()).into())
            );
        }
    }

    #[test]
    fn case_constr_case_0() {
        let make_program =
            |fun: DefaultFunction, tag: u64, n: i32, m: i32| Program::<NamedDeBruijn> {
                version: (0, 0, 0),
                term: Term::Case {
                    constr: Term::Constr {
                        tag,
                        fields: vec![
                            Term::Constant(Constant::Integer(n.into()).into()),
                            Term::Constant(Constant::Integer(m.into()).into()),
                        ],
                    }
                    .into(),
                    branches: vec![Term::Builtin(fun), Term::subtract_integer()],
                },
            };

        let test_data = vec![
            (DefaultFunction::AddInteger, 0, 8, 3, 11),
            (DefaultFunction::AddInteger, 1, 8, 3, 5),
        ];

        for (fun, tag, n, m, result) in test_data {
            let eval_result = make_program(fun, tag, n, m).eval(ExBudget::max());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant(Constant::Integer(result.into()).into())
            );
        }
    }

    #[test]
    fn case_constr_tag_beyond_branches_is_missing_branch() {
        for tag in [3, u32::MAX as u64 + 1, i64::MAX as u64 + 1, u64::MAX] {
            let program = Program::<NamedDeBruijn> {
                version: (1, 1, 0),
                term: Term::Case {
                    constr: Term::Constr {
                        tag,
                        fields: vec![],
                    }
                    .into(),
                    branches: vec![Term::integer(5.into()); 3],
                },
            };

            assert!(
                matches!(
                    program.eval(ExBudget::max()).result(),
                    Err(Error::MissingCaseBranch(..))
                ),
                "{tag}"
            );
        }
    }

    #[test]
    fn case_constr_case_1() {
        let make_program = |tag: u64| Program::<NamedDeBruijn> {
            version: (0, 0, 0),
            term: Term::Case {
                constr: Term::Constr {
                    tag,
                    fields: vec![],
                }
                .into(),
                branches: vec![
                    Term::integer(5.into()),
                    Term::integer(10.into()),
                    Term::integer(15.into()),
                ],
            },
        };

        let test_data = vec![(0, 5), (1, 10), (2, 15)];

        for (tag, result) in test_data {
            let eval_result = make_program(tag).eval(ExBudget::max());

            assert_eq!(
                eval_result.result().unwrap(),
                Term::Constant(Constant::Integer(result.into()).into())
            );
        }
    }

    #[test]
    fn bls_g1_add_associative() {
        let a = blst::blst_p1::uncompress(&[
            0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac,
            0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2,
            0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00,
            0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f,
        ])
        .unwrap();

        let b = blst::blst_p1::uncompress(&[
            0x95, 0x0d, 0xfd, 0x33, 0xda, 0x26, 0x82, 0x26, 0x0c, 0x76, 0x03, 0x8d, 0xfb, 0x8b,
            0xad, 0x6e, 0x84, 0xae, 0x9d, 0x59, 0x9a, 0x3c, 0x15, 0x18, 0x15, 0x94, 0x5a, 0xc1,
            0xe6, 0xef, 0x6b, 0x10, 0x27, 0xcd, 0x91, 0x7f, 0x39, 0x07, 0x47, 0x9d, 0x20, 0xd6,
            0x36, 0xce, 0x43, 0x7a, 0x41, 0xf5,
        ])
        .unwrap();

        let c = blst::blst_p1::uncompress(&[
            0xb9, 0x62, 0xfd, 0x0c, 0xc8, 0x10, 0x48, 0xe0, 0xcf, 0x75, 0x57, 0xbf, 0x3e, 0x4b,
            0x6e, 0xdc, 0x5a, 0xb4, 0xbf, 0xb3, 0xdc, 0x87, 0xf8, 0x3a, 0xf4, 0x28, 0xb6, 0x30,
            0x07, 0x27, 0xb1, 0x39, 0xc4, 0x04, 0xab, 0x15, 0x9b, 0xdf, 0x2e, 0xae, 0xa3, 0xf6,
            0x49, 0x90, 0x34, 0x21, 0x53, 0x7f,
        ])
        .unwrap();

        let term: Term<NamedDeBruijn> = Term::bls12_381_g1_equal()
            .apply(
                Term::bls12_381_g1_add().apply(Term::bls12_381_g1(a)).apply(
                    Term::bls12_381_g1_add()
                        .apply(Term::bls12_381_g1(b))
                        .apply(Term::bls12_381_g1(c)),
                ),
            )
            .apply(
                Term::bls12_381_g1_add()
                    .apply(
                        Term::bls12_381_g1_add()
                            .apply(Term::bls12_381_g1(a))
                            .apply(Term::bls12_381_g1(b)),
                    )
                    .apply(Term::bls12_381_g1(c)),
            );

        let program = Program {
            version: (1, 0, 0),
            term,
        };

        let eval_result = program.eval(Default::default());

        let final_term = eval_result.result().unwrap();

        assert_eq!(final_term, Term::bool(true))
    }

    #[test]
    fn bls_g2_add_associative() {
        let a = blst::blst_p1::uncompress(&[
            0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac,
            0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2,
            0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00,
            0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f,
        ])
        .unwrap();

        let b = blst::blst_p1::uncompress(&[
            0x95, 0x0d, 0xfd, 0x33, 0xda, 0x26, 0x82, 0x26, 0x0c, 0x76, 0x03, 0x8d, 0xfb, 0x8b,
            0xad, 0x6e, 0x84, 0xae, 0x9d, 0x59, 0x9a, 0x3c, 0x15, 0x18, 0x15, 0x94, 0x5a, 0xc1,
            0xe6, 0xef, 0x6b, 0x10, 0x27, 0xcd, 0x91, 0x7f, 0x39, 0x07, 0x47, 0x9d, 0x20, 0xd6,
            0x36, 0xce, 0x43, 0x7a, 0x41, 0xf5,
        ])
        .unwrap();

        let c = blst::blst_p1::uncompress(&[
            0xb9, 0x62, 0xfd, 0x0c, 0xc8, 0x10, 0x48, 0xe0, 0xcf, 0x75, 0x57, 0xbf, 0x3e, 0x4b,
            0x6e, 0xdc, 0x5a, 0xb4, 0xbf, 0xb3, 0xdc, 0x87, 0xf8, 0x3a, 0xf4, 0x28, 0xb6, 0x30,
            0x07, 0x27, 0xb1, 0x39, 0xc4, 0x04, 0xab, 0x15, 0x9b, 0xdf, 0x2e, 0xae, 0xa3, 0xf6,
            0x49, 0x90, 0x34, 0x21, 0x53, 0x7f,
        ])
        .unwrap();

        let term: Term<NamedDeBruijn> = Term::bls12_381_g1_equal()
            .apply(
                Term::bls12_381_g1_add().apply(Term::bls12_381_g1(a)).apply(
                    Term::bls12_381_g1_add()
                        .apply(Term::bls12_381_g1(b))
                        .apply(Term::bls12_381_g1(c)),
                ),
            )
            .apply(
                Term::bls12_381_g1_add()
                    .apply(
                        Term::bls12_381_g1_add()
                            .apply(Term::bls12_381_g1(a))
                            .apply(Term::bls12_381_g1(b)),
                    )
                    .apply(Term::bls12_381_g1(c)),
            );

        let program = Program {
            version: (1, 0, 0),
            term,
        };

        let eval_result = program.eval(Default::default());

        let final_term = eval_result.result().unwrap();

        assert_eq!(final_term, Term::bool(true))
    }

    #[test]
    fn failed_run_drops_pending_frames() {
        // `[(lam x x) (error)]` fails while the lambda still awaits its argument.
        let term: Term<NamedDeBruijn> = Term::Apply {
            function: Term::Lambda {
                parameter_name: NamedDeBruijn {
                    text: "x".to_string(),
                    index: 0.into(),
                }
                .into(),
                body: Term::Var(
                    NamedDeBruijn {
                        text: "x".to_string(),
                        index: 1.into(),
                    }
                    .into(),
                )
                .into(),
            }
            .into(),
            argument: Term::Error.into(),
        };

        let mut machine = Machine::new(
            Language::PlutusV3,
            CostModel::default(),
            ExBudget::default(),
            200,
        );

        assert!(matches!(machine.run(term), Err(Error::EvaluationFailure)));
        assert!(machine.frames.is_empty());
    }

    #[test]
    fn nested_constr_and_case_and_list_spines() {
        // Constr fields and case branches are entered in place, closures
        // escape from inside them, and list spines are consed onto, shared
        // and split. Small enough to run under miri.
        let eval = |src: &str| {
            let program: Program<NamedDeBruijn> =
                crate::parser::program(src).unwrap().try_into().unwrap();

            program.eval(ExBudget::max()).result().unwrap()
        };

        let int = |i: i32| Term::Constant(Constant::Integer(i.into()).into());

        assert_eq!(
            eval(
                "(program 1.1.0
                  [ (lam f
                      (case (constr 1 (con integer 7) f)
                        (lam a (lam g (con integer 0)))
                        (lam a (lam g
                          [ g (constr 0 a (case (constr 0) (con integer 2))) ]))))
                    (lam p (case p (lam x (lam y [(builtin addInteger) x y])))) ])"
            ),
            int(9)
        );

        assert_eq!(
            eval(
                "(program 1.1.0
                  [ (lam cons
                      [ (lam l
                          [ (lam a
                              [ (lam b
                                  [ [ (builtin addInteger)
                                      [ (force (builtin headList))
                                        [ (force (builtin tailList)) b ] ] ]
                                    [ (force (builtin headList))
                                      [ [ (force (builtin dropList)) (con integer 2) ] a ] ] ])
                                [ [ cons (con integer 30) ] l ] ])
                            [ [ cons (con integer 20) ] [ [ cons (con integer 10) ] l ] ] ])
                        (con (list integer) [1, 2]) ])
                    (force (builtin mkCons)) ])"
            ),
            int(2)
        );
    }
}
