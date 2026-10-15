use super::{Error, Trace, cost_model::ExBudget, discharge, value::Value};
use crate::ast::{Constant, NamedDeBruijn, Term};
use pallas_primitives::conway::Language;

#[derive(Debug)]
pub struct EvalResult {
    outcome: Result<Outcome, Error>,
    pub remaining_budget: ExBudget,
    pub initial_budget: ExBudget,
    pub traces: Vec<Trace>,
    pub debug_cost: Option<Vec<i64>>,
}

impl EvalResult {
    pub fn new(
        result: Result<Term<NamedDeBruijn>, Error>,
        remaining_budget: ExBudget,
        initial_budget: ExBudget,
        traces: Vec<Trace>,
        debug_cost: Option<Vec<i64>>,
    ) -> EvalResult {
        EvalResult {
            outcome: result.map(Outcome::Term),
            remaining_budget,
            initial_budget,
            traces,
            debug_cost,
        }
    }

    /// The result of an evaluation that returned `value`, which is discharged
    /// into a term only when asked for.
    pub(crate) fn returning(
        value: Result<Value, Error>,
        remaining_budget: ExBudget,
        initial_budget: ExBudget,
        traces: Vec<Trace>,
        debug_cost: Option<Vec<i64>>,
    ) -> EvalResult {
        EvalResult {
            outcome: value.map(Outcome::Value),
            remaining_budget,
            initial_budget,
            traces,
            debug_cost,
        }
    }

    pub fn cost(&self) -> ExBudget {
        self.initial_budget - self.remaining_budget
    }

    pub fn traces(&self) -> Vec<Trace> {
        self.traces.clone()
    }

    pub fn logs(&self) -> Vec<String> {
        self.traces
            .iter()
            .cloned()
            .filter_map(Trace::unwrap_log)
            .collect()
    }

    pub fn labels(&self) -> Vec<String> {
        self.traces
            .iter()
            .cloned()
            .filter_map(Trace::unwrap_label)
            .collect()
    }

    /// Determine whether script execution failed, taking the Plutus language
    /// version into account.
    ///
    /// This mirrors the Haskell reference implementation's `processLogsAndErrors`
    /// in `PlutusLedgerApi.Common.Eval`:
    ///
    /// - **PlutusV1/V2**: A script succeeds if it terminates without error.
    ///   Any non-error return value (not just `Bool(true)` or `Unit`) is
    ///   considered success.
    ///
    /// - **PlutusV3**: A script succeeds only if it returns `Bool(true)` or
    ///   `Unit` (the strict check, same as `failed(false)`).
    pub fn failed(&self, allow_bool_in_v3: bool, language: &Language) -> bool {
        let errored = match &self.outcome {
            Err(_) => true,
            Ok(outcome) => matches!(outcome, Outcome::Term(Term::Error)),
        };

        let expected_return = match language {
            // PlutusV1/V2: success = execution didn't error; any non-error result is acceptable
            Language::PlutusV1 | Language::PlutusV2 => true,
            // PlutusV3: strict check — must return Bool(true) or Unit.
            Language::PlutusV3 => self.constant().is_some_and(|constant| {
                matches!(constant, Constant::Unit)
                    || (allow_bool_in_v3 && matches!(constant, Constant::Bool(true)))
            }),
        };

        errored || !expected_return
    }

    pub fn debug_cost(&self) -> Option<Vec<i64>> {
        self.debug_cost.clone()
    }

    /// The constant the evaluation returned, if it returned one.
    pub fn constant(&self) -> Option<&Constant> {
        match self.outcome.as_ref().ok()? {
            Outcome::Value(Value::Con(constant)) | Outcome::Term(Term::Constant(constant)) => {
                Some(constant.as_ref())
            }
            _ => None,
        }
    }

    /// The error the evaluation failed with, if it failed.
    pub fn error(&self) -> Option<&Error> {
        self.outcome.as_ref().err()
    }

    #[allow(clippy::result_unit_err)]
    pub fn unwrap_constant(self) -> Result<Constant, ()> {
        self.constant().cloned().ok_or(())
    }

    /// The term the evaluation returned, or the error it failed with.
    ///
    /// A value the evaluation returned is discharged into a term (Plutus'
    /// `dischargeCekValue`) each time. Values share their parts, so a small
    /// value can discharge into a term exponentially larger: like the ledger,
    /// prefer [`EvalResult::error`] and [`EvalResult::constant`] when they say
    /// enough.
    pub fn result(&self) -> Result<Term<NamedDeBruijn>, Error> {
        match &self.outcome {
            Ok(outcome) => Ok(outcome.to_term()),
            Err(error) => Err(error.clone()),
        }
    }

    /// Like [`EvalResult::result`], without copying the error or a term.
    pub fn into_result(self) -> Result<Term<NamedDeBruijn>, Error> {
        self.outcome.map(Outcome::into_term)
    }
}

/// What an evaluation returned: a value, discharged into a term only when
/// asked for (as Plutus' `DischargeResult` is), or a term.
#[derive(Debug)]
enum Outcome {
    Value(Value),
    Term(Term<NamedDeBruijn>),
}

impl Outcome {
    fn to_term(&self) -> Term<NamedDeBruijn> {
        match self {
            Outcome::Value(value) => discharge::value_as_term(value),
            Outcome::Term(term) => term.clone(),
        }
    }

    fn into_term(self) -> Term<NamedDeBruijn> {
        match self {
            Outcome::Value(value) => discharge::value_as_term(&value),
            Outcome::Term(term) => term,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Language::*;
    use test_case::test_case;

    #[test_case(Ok(Term::integer(42.into())), PlutusV1, false => false)]
    #[test_case(Ok(Term::integer(42.into())), PlutusV2, false => false)]
    #[test_case(Ok(Term::integer(42.into())), PlutusV3, false => true)]
    #[test_case(Ok(Term::bool(true)),         PlutusV1, false => false)]
    #[test_case(Ok(Term::bool(true)),         PlutusV2, false => false)]
    #[test_case(Ok(Term::bool(true)),         PlutusV3, false => true)]
    #[test_case(Ok(Term::bool(true)),         PlutusV3, true  => false)]
    #[test_case(Ok(Term::bool(false)),        PlutusV1, false => false)]
    #[test_case(Ok(Term::bool(false)),        PlutusV2, false => false)]
    #[test_case(Ok(Term::bool(false)),        PlutusV3, false => true)]
    #[test_case(Ok(Term::bool(false)),        PlutusV3, true  => true)]
    #[test_case(Ok(Term::unit()),             PlutusV1, false => false)]
    #[test_case(Ok(Term::unit()),             PlutusV2, false => false)]
    #[test_case(Ok(Term::unit()),             PlutusV3, false => false)]
    #[test_case(Ok(Term::Error),              PlutusV1, false => true)]
    #[test_case(Ok(Term::Error),              PlutusV2, false => true)]
    #[test_case(Ok(Term::Error),              PlutusV3, false => true)]
    #[test_case(Ok(Term::var("x").lambda("x").try_into().unwrap()), PlutusV1, false => false)]
    #[test_case(Ok(Term::var("x").lambda("x").try_into().unwrap()), PlutusV2, false => false)]
    #[test_case(Ok(Term::var("x").lambda("x").try_into().unwrap()), PlutusV3, false => true)]
    #[test_case(Err(Error::OutOfExError(ExBudget::default())), Language::PlutusV1, false => true)]
    #[test_case(Err(Error::OutOfExError(ExBudget::default())), Language::PlutusV2, false => true)]
    #[test_case(Err(Error::OutOfExError(ExBudget::default())), Language::PlutusV3, false => true;
    )]
    fn eval_result_failed(
        term: Result<Term<NamedDeBruijn>, Error>,
        lang: Language,
        allow_bool_in_v3: bool,
    ) -> bool {
        EvalResult::new(term, ExBudget::default(), ExBudget::default(), vec![], None)
            .failed(allow_bool_in_v3, &lang)
    }
}
