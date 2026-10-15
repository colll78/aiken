use std::{collections::HashMap, convert::Infallible, rc::Rc};

use thiserror::Error;

use crate::{
    ast::{DeBruijn, FakeNamedDeBruijn, Name, NamedDeBruijn, Term, Unique},
    rebuild::{Rebuild, rebuild},
};

#[derive(Debug, Clone, PartialEq, Copy, Eq, Hash)]
struct Level(usize);

#[derive(Error, Debug)]
pub enum Error {
    #[error("Free Unique {} with name {}", .0.unique, .0.text)]
    FreeUnique(Name),
    #[error("Free Index `{0}`")]
    FreeIndex(DeBruijn),
}

pub struct Converter {
    current_level: Level,
    /// The levels each unique is bound at, innermost last.
    binders: HashMap<Unique, Vec<Level>>,
    /// The unique bound at each level up to the current one, if any.
    uniques: Vec<Option<Unique>>,
    current_unique: Unique,
}

impl Converter {
    pub fn new() -> Self {
        Converter {
            current_level: Level(0),
            binders: HashMap::new(),
            uniques: vec![None],
            current_unique: Unique::new(0),
        }
    }

    pub fn name_to_named_debruijn(
        &mut self,
        term: &Term<Name>,
    ) -> Result<Term<NamedDeBruijn>, Error> {
        rebuild(term, &mut NameToNamedDeBruijn(self))
    }

    pub fn name_to_debruijn(&mut self, term: &Term<Name>) -> Result<Term<DeBruijn>, Error> {
        rebuild(term, &mut NameToDeBruijn(self))
    }

    pub fn named_debruijn_to_name(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Result<Term<Name>, Error> {
        rebuild(term, &mut NamedDeBruijnToName(self))
    }

    pub fn debruijn_to_name(&mut self, term: &Term<DeBruijn>) -> Result<Term<Name>, Error> {
        rebuild(term, &mut DeBruijnToName(self))
    }

    pub fn named_debruijn_to_debruijn(&mut self, term: &Term<NamedDeBruijn>) -> Term<DeBruijn> {
        relabel(term, |name: &NamedDeBruijn| name.index)
    }

    pub fn debruijn_to_named_debruijn(&mut self, term: &Term<DeBruijn>) -> Term<NamedDeBruijn> {
        relabel(term, |index: &DeBruijn| NamedDeBruijn {
            text: "i".to_string(),
            index: *index,
        })
    }

    pub fn fake_named_debruijn_to_named_debruijn(
        &mut self,
        term: &Term<FakeNamedDeBruijn>,
    ) -> Term<NamedDeBruijn> {
        relabel(term, |name: &FakeNamedDeBruijn| name.clone().into())
    }

    pub fn named_debruijn_to_fake_named_debruijn(
        &mut self,
        term: &Term<NamedDeBruijn>,
    ) -> Term<FakeNamedDeBruijn> {
        relabel(term, |name: &NamedDeBruijn| name.clone().into())
    }

    fn get_index(&mut self, name: &Name) -> Result<DeBruijn, Error> {
        match self
            .binders
            .get(&name.unique)
            .and_then(|levels| levels.last())
        {
            Some(found_level) => Ok((self.current_level.0 - found_level.0).into()),
            None => Err(Error::FreeUnique(name.clone())),
        }
    }

    fn get_unique(&mut self, index: &DeBruijn) -> Result<Unique, Error> {
        let level = Level(
            self.current_level
                .0
                .checked_sub(index.inner())
                .ok_or(Error::FreeIndex(*index))?,
        );

        self.uniques[level.0].ok_or(Error::FreeIndex(*index))
    }

    fn declare_unique(&mut self, unique: Unique) {
        self.binders
            .entry(unique)
            .or_default()
            .push(self.current_level);
    }

    fn remove_unique(&mut self, unique: Unique) {
        if let Some(levels) = self.binders.get_mut(&unique) {
            levels.pop();
        }
    }

    fn declare_binder(&mut self) {
        self.uniques[self.current_level.0] = Some(self.current_unique);

        self.current_unique.increment();
    }

    fn start_scope(&mut self) {
        self.current_level = Level(self.current_level.0 + 1);

        self.uniques.push(None);
    }

    fn end_scope(&mut self) {
        self.current_level = Level(self.current_level.0 - 1);

        self.uniques.pop();
    }
}

/// Renames every binder and variable of a term on its own.
fn relabel<A, B>(term: &Term<A>, rename: impl FnMut(&A) -> B) -> Term<B> {
    struct Relabel<F>(F);

    impl<A, B, F: FnMut(&A) -> B> Rebuild<A, B> for Relabel<F> {
        type Error = Infallible;

        fn var(&mut self, name: &Rc<A>) -> Result<Rc<B>, Infallible> {
            Ok((self.0)(name).into())
        }

        fn enter_lambda(&mut self, parameter_name: &Rc<A>) -> Result<Rc<B>, Infallible> {
            Ok((self.0)(parameter_name).into())
        }
    }

    match rebuild(term, &mut Relabel(rename)) {
        Ok(term) => term,
        Err(never) => match never {},
    }
}

struct NameToNamedDeBruijn<'a>(&'a mut Converter);

impl Rebuild<Name, NamedDeBruijn> for NameToNamedDeBruijn<'_> {
    type Error = Error;

    fn var(&mut self, name: &Rc<Name>) -> Result<Rc<NamedDeBruijn>, Error> {
        Ok(NamedDeBruijn {
            text: name.text.to_string(),
            index: self.0.get_index(name)?,
        }
        .into())
    }

    fn enter_lambda(&mut self, parameter_name: &Rc<Name>) -> Result<Rc<NamedDeBruijn>, Error> {
        self.0.declare_unique(parameter_name.unique);

        let index = self.0.get_index(parameter_name)?;

        self.0.start_scope();

        Ok(NamedDeBruijn {
            text: parameter_name.text.to_string(),
            index,
        }
        .into())
    }

    fn exit_lambda(&mut self, parameter_name: &Rc<Name>) {
        self.0.end_scope();
        self.0.remove_unique(parameter_name.unique);
    }
}

struct NameToDeBruijn<'a>(&'a mut Converter);

impl Rebuild<Name, DeBruijn> for NameToDeBruijn<'_> {
    type Error = Error;

    fn var(&mut self, name: &Rc<Name>) -> Result<Rc<DeBruijn>, Error> {
        Ok(self.0.get_index(name)?.into())
    }

    fn enter_lambda(&mut self, parameter_name: &Rc<Name>) -> Result<Rc<DeBruijn>, Error> {
        self.0.declare_unique(parameter_name.unique);

        let index = self.0.get_index(parameter_name)?;

        self.0.start_scope();

        Ok(index.into())
    }

    fn exit_lambda(&mut self, parameter_name: &Rc<Name>) {
        self.0.end_scope();
        self.0.remove_unique(parameter_name.unique);
    }
}

struct NamedDeBruijnToName<'a>(&'a mut Converter);

impl Rebuild<NamedDeBruijn, Name> for NamedDeBruijnToName<'_> {
    type Error = Error;

    fn var(&mut self, name: &Rc<NamedDeBruijn>) -> Result<Rc<Name>, Error> {
        Ok(Name {
            text: name.text.to_string(),
            unique: self.0.get_unique(&name.index)?,
        }
        .into())
    }

    fn enter_lambda(&mut self, parameter_name: &Rc<NamedDeBruijn>) -> Result<Rc<Name>, Error> {
        self.0.declare_binder();

        let unique = self.0.get_unique(&parameter_name.index)?;

        self.0.start_scope();

        Ok(Name {
            text: parameter_name.text.to_string(),
            unique,
        }
        .into())
    }

    fn exit_lambda(&mut self, _: &Rc<NamedDeBruijn>) {
        self.0.end_scope();
    }
}

struct DeBruijnToName<'a>(&'a mut Converter);

impl Rebuild<DeBruijn, Name> for DeBruijnToName<'_> {
    type Error = Error;

    fn var(&mut self, index: &Rc<DeBruijn>) -> Result<Rc<Name>, Error> {
        let unique = self.0.get_unique(index)?;

        Ok(Name {
            text: format!("i_{unique}"),
            unique,
        }
        .into())
    }

    fn enter_lambda(&mut self, parameter_name: &Rc<DeBruijn>) -> Result<Rc<Name>, Error> {
        self.0.declare_binder();

        let unique = self.0.get_unique(parameter_name)?;

        self.0.start_scope();

        Ok(Name {
            text: format!("i_{unique}"),
            unique,
        }
        .into())
    }

    fn exit_lambda(&mut self, _: &Rc<DeBruijn>) {
        self.0.end_scope();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(unique: isize) -> Rc<Name> {
        Name {
            text: format!("x_{unique}"),
            unique: Unique::new(unique),
        }
        .into()
    }

    fn lambda(unique: isize, body: Term<Name>) -> Term<Name> {
        Term::Lambda {
            parameter_name: name(unique),
            body: body.into(),
        }
    }

    fn debruijn_lambda(body: Term<DeBruijn>) -> Term<DeBruijn> {
        Term::Lambda {
            parameter_name: DeBruijn::new(0).into(),
            body: body.into(),
        }
    }

    fn debruijn_var(index: usize) -> Term<DeBruijn> {
        Term::Var(DeBruijn::new(index).into())
    }

    #[test]
    fn names_resolve_to_their_innermost_binder() {
        // (lam x [(lam x x) x])
        let term = lambda(0, lambda(0, Term::Var(name(0))).apply(Term::Var(name(0))));

        let converted = Converter::new().name_to_debruijn(&term).unwrap();

        assert_eq!(
            converted,
            debruijn_lambda(debruijn_lambda(debruijn_var(1)).apply(debruijn_var(1)))
        );

        let named = Converter::new().debruijn_to_name(&converted).unwrap();

        assert_eq!(
            Converter::new().name_to_debruijn(&named).unwrap(),
            converted
        );
    }

    #[test]
    fn free_variables_are_rejected() {
        // (lam x [(lam y y) y])
        let term = lambda(0, lambda(1, Term::Var(name(1))).apply(Term::Var(name(1))));

        assert!(matches!(
            Converter::new().name_to_debruijn(&term),
            Err(Error::FreeUnique(name)) if name.unique == Unique::new(1)
        ));

        // (lam _ [(lam _ 1) 2])
        let term = debruijn_lambda(debruijn_lambda(debruijn_var(1)).apply(debruijn_var(2)));

        assert!(matches!(
            Converter::new().debruijn_to_name(&term),
            Err(Error::FreeIndex(index)) if index.inner() == 2
        ));
    }

    #[test]
    fn converts_variables_bound_far_out_in_linear_time() {
        #[cfg(not(miri))]
        const DEPTH: usize = 20_000;
        #[cfg(miri)]
        const DEPTH: usize = 50;

        // Every one of the variables refers to the outermost of the binders,
        // which a lookup through every scope in between would take quadratic
        // time to find.
        let named = lambda(
            0,
            (1..=DEPTH as isize).rev().fold(
                Term::Constr {
                    tag: 0,
                    fields: vec![Term::Var(name(0)); DEPTH],
                },
                |body, unique| lambda(unique, body),
            ),
        );

        let expected = debruijn_lambda((0..DEPTH).fold(
            Term::Constr {
                tag: 0,
                fields: vec![debruijn_var(DEPTH + 1); DEPTH],
            },
            |body, _| debruijn_lambda(body),
        ));

        let converted = Converter::new().name_to_debruijn(&named).unwrap();
        assert_eq!(converted, expected);

        let renamed = Converter::new().debruijn_to_name(&converted).unwrap();
        assert_eq!(
            Converter::new().name_to_debruijn(&renamed).unwrap(),
            expected
        );
    }
}
