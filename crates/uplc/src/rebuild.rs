//! Walks over terms that keep their pending work on the heap, so that terms
//! nested as deep as a script can make them (tens of thousands of levels, four
//! bits each in flat) are handled without overflowing the stack.

use crate::ast::{Constant, Term};
use std::rc::Rc;

/// How [`rebuild`] turns a `Term<A>` into a `Term<B>`.
pub(crate) trait Rebuild<A, B> {
    type Error;

    fn var(&mut self, name: &Rc<A>) -> Result<Rc<B>, Self::Error>;

    /// Called before the body of a lambda is rebuilt.
    fn enter_lambda(&mut self, parameter_name: &Rc<A>) -> Result<Rc<B>, Self::Error>;

    /// Called once the body of a lambda has been rebuilt.
    fn exit_lambda(&mut self, _parameter_name: &Rc<A>) {}

    fn constant(&mut self, constant: &Rc<Constant>) -> Rc<Constant> {
        constant.clone()
    }

    /// The rebuilt form of a subterm held behind an `Rc`, when it can be had
    /// without rebuilding the subterm (as by sharing it), or `None`.
    fn shared(&mut self, _term: &Rc<Term<A>>) -> Option<Rc<Term<B>>> {
        None
    }
}

/// A subterm still to rebuild, or the rest of a term once its subterms are.
enum Task<'a, A> {
    Visit(&'a Term<A>),
    VisitShared(&'a Rc<Term<A>>),
    Finish(&'a Term<A>),
}

/// A rebuilt subterm, owned or shared.
enum Built<B> {
    Owned(Term<B>),
    Shared(Rc<Term<B>>),
}

impl<B> Built<B> {
    fn into_rc(self) -> Rc<Term<B>> {
        match self {
            Built::Owned(term) => Rc::new(term),
            Built::Shared(term) => term,
        }
    }

    fn into_term(self) -> Term<B> {
        match self {
            Built::Owned(term) => term,
            Built::Shared(term) => Rc::unwrap_or_clone(term),
        }
    }
}

/// How deep [`rebuild`] recurses before it carries on with the pending work on
/// the heap. Recursing is quicker, and real scripts are rarely this deep.
const MAX_RECURSION_DEPTH: usize = 512;

/// Rebuilds `term` bottom up, calling the hooks of `rebuilder` in the order a
/// recursive walk would: subterms left to right, and each lambda entered
/// before its body and exited after it.
pub(crate) fn rebuild<A, B, R>(term: &Term<A>, rebuilder: &mut R) -> Result<Term<B>, R::Error>
where
    R: Rebuild<A, B>,
{
    rebuild_recursively(term, rebuilder, 0)
}

fn rebuild_recursively<A, B, R>(
    term: &Term<A>,
    rebuilder: &mut R,
    depth: usize,
) -> Result<Term<B>, R::Error>
where
    R: Rebuild<A, B>,
{
    if depth >= MAX_RECURSION_DEPTH {
        return rebuild_iteratively(term, rebuilder);
    }

    let depth = depth + 1;

    Ok(match term {
        Term::Var(name) => Term::Var(rebuilder.var(name)?),
        Term::Constant(constant) => Term::Constant(rebuilder.constant(constant)),
        Term::Builtin(fun) => Term::Builtin(*fun),
        Term::Error => Term::Error,
        Term::Delay(body) => Term::Delay(rebuild_shared(body, rebuilder, depth)?),
        Term::Force(body) => Term::Force(rebuild_shared(body, rebuilder, depth)?),
        Term::Lambda {
            parameter_name,
            body,
        } => {
            let rebuilt_parameter_name = rebuilder.enter_lambda(parameter_name)?;
            let body = rebuild_shared(body, rebuilder, depth)?;
            rebuilder.exit_lambda(parameter_name);

            Term::Lambda {
                parameter_name: rebuilt_parameter_name,
                body,
            }
        }
        Term::Apply { function, argument } => Term::Apply {
            function: rebuild_shared(function, rebuilder, depth)?,
            argument: rebuild_shared(argument, rebuilder, depth)?,
        },
        Term::Constr { tag, fields } => Term::Constr {
            tag: *tag,
            fields: fields
                .iter()
                .map(|field| rebuild_recursively(field, rebuilder, depth))
                .collect::<Result<_, _>>()?,
        },
        Term::Case { constr, branches } => Term::Case {
            constr: rebuild_shared(constr, rebuilder, depth)?,
            branches: branches
                .iter()
                .map(|branch| rebuild_recursively(branch, rebuilder, depth))
                .collect::<Result<_, _>>()?,
        },
    })
}

fn rebuild_shared<A, B, R>(
    term: &Rc<Term<A>>,
    rebuilder: &mut R,
    depth: usize,
) -> Result<Rc<Term<B>>, R::Error>
where
    R: Rebuild<A, B>,
{
    match rebuilder.shared(term) {
        Some(term) => Ok(term),
        None => rebuild_recursively(term, rebuilder, depth).map(Rc::new),
    }
}

fn rebuild_iteratively<A, B, R>(term: &Term<A>, rebuilder: &mut R) -> Result<Term<B>, R::Error>
where
    R: Rebuild<A, B>,
{
    let mut tasks = vec![Task::Visit(term)];
    let mut built: Vec<Built<B>> = Vec::new();
    let mut parameter_names: Vec<Rc<B>> = Vec::new();

    while let Some(task) = tasks.pop() {
        let term = match task {
            Task::Visit(term) => term,
            Task::VisitShared(term) => match rebuilder.shared(term) {
                Some(term) => {
                    built.push(Built::Shared(term));
                    continue;
                }
                None => term.as_ref(),
            },
            Task::Finish(term) => {
                let rebuilt = match term {
                    Term::Delay(_) => Term::Delay(pop(&mut built).into_rc()),
                    Term::Force(_) => Term::Force(pop(&mut built).into_rc()),
                    Term::Lambda { parameter_name, .. } => {
                        rebuilder.exit_lambda(parameter_name);

                        Term::Lambda {
                            parameter_name: parameter_names
                                .pop()
                                .expect("every lambda entered has a parameter name"),
                            body: pop(&mut built).into_rc(),
                        }
                    }
                    Term::Apply { .. } => {
                        let argument = pop(&mut built).into_rc();
                        let function = pop(&mut built).into_rc();

                        Term::Apply { function, argument }
                    }
                    Term::Constr { tag, fields } => Term::Constr {
                        tag: *tag,
                        fields: pop_many(&mut built, fields.len()),
                    },
                    Term::Case { branches, .. } => {
                        let branches = pop_many(&mut built, branches.len());

                        Term::Case {
                            constr: pop(&mut built).into_rc(),
                            branches,
                        }
                    }
                    Term::Var(_) | Term::Constant(_) | Term::Builtin(_) | Term::Error => {
                        unreachable!("only terms with subterms are finished")
                    }
                };

                built.push(Built::Owned(rebuilt));
                continue;
            }
        };

        let leaf = match term {
            Term::Var(name) => Term::Var(rebuilder.var(name)?),
            Term::Constant(constant) => Term::Constant(rebuilder.constant(constant)),
            Term::Builtin(fun) => Term::Builtin(*fun),
            Term::Error => Term::Error,
            Term::Delay(body) | Term::Force(body) => {
                tasks.push(Task::Finish(term));
                tasks.push(Task::VisitShared(body));
                continue;
            }
            Term::Lambda {
                parameter_name,
                body,
            } => {
                parameter_names.push(rebuilder.enter_lambda(parameter_name)?);
                tasks.push(Task::Finish(term));
                tasks.push(Task::VisitShared(body));
                continue;
            }
            Term::Apply { function, argument } => {
                tasks.push(Task::Finish(term));
                tasks.push(Task::VisitShared(argument));
                tasks.push(Task::VisitShared(function));
                continue;
            }
            Term::Constr { fields, .. } => {
                tasks.push(Task::Finish(term));
                tasks.extend(fields.iter().rev().map(Task::Visit));
                continue;
            }
            Term::Case { constr, branches } => {
                tasks.push(Task::Finish(term));
                tasks.extend(branches.iter().rev().map(Task::Visit));
                tasks.push(Task::VisitShared(constr));
                continue;
            }
        };

        built.push(Built::Owned(leaf));
    }

    Ok(pop(&mut built).into_term())
}

fn pop<B>(built: &mut Vec<Built<B>>) -> Built<B> {
    built
        .pop()
        .expect("every subterm is built before its parent")
}

fn pop_many<B>(built: &mut Vec<Built<B>>, count: usize) -> Vec<Term<B>> {
    built
        .drain(built.len() - count..)
        .map(Built::into_term)
        .collect()
}
