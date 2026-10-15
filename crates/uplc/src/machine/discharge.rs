use std::rc::Rc;

use crate::{
    ast::{NamedDeBruijn, Term},
    builtins::DefaultFunction,
};

use super::value::{Env, Value};

/// Converts a value into a term, replacing every variable bound in an
/// environment by the discharged value it is bound to (Plutus'
/// `dischargeCekValue`).
///
/// Values can nest as deep as a script or its evaluation makes them, so this
/// keeps its pending work on the heap instead of recursing.
pub(super) fn value_as_term(value: &Value) -> Term<NamedDeBruijn> {
    let mut tasks = vec![Task::Value(value)];
    let mut built: Vec<Term<NamedDeBruijn>> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Value(value) => match value {
                Value::Con(constant) => built.push(Term::Constant(constant.clone())),
                Value::Delay(body, env) => {
                    tasks.push(Task::Finish(Finish::Delay));
                    tasks.push(Task::Term(body, env, 0));
                }
                Value::Lambda {
                    parameter_name,
                    body,
                    env,
                } => {
                    // The index on a binder is meaningless; 0 by convention.
                    tasks.push(Task::Finish(Finish::Lambda(Rc::new(NamedDeBruijn {
                        text: parameter_name.text.clone(),
                        index: 0.into(),
                    }))));
                    tasks.push(Task::Term(body, env, 1));
                }
                Value::Builtin { fun, runtime } => {
                    tasks.push(Task::Finish(Finish::Builtin {
                        fun: *fun,
                        forces: runtime.forces,
                        arguments: runtime.args().len(),
                    }));
                    tasks.extend(runtime.args().iter().rev().map(Task::Value));
                }
                Value::Constr { tag, fields } => {
                    tasks.push(Task::Finish(Finish::Constr {
                        tag: *tag,
                        fields: fields.len(),
                    }));
                    tasks.extend(fields.iter().rev().map(Task::Value));
                }
            },
            // `shift` counts the lambdas entered since leaving `env`.
            Task::Term(term, env, shift) => match term {
                Term::Var(name) => {
                    let index: usize = name.index.into();

                    match index
                        .checked_sub(shift)
                        .filter(|&index| index > 0)
                        .and_then(|index| env.get(index))
                    {
                        Some(value) => tasks.push(Task::Value(value)),
                        // Bound inside the term, or free.
                        None => built.push(Term::Var(name.clone())),
                    }
                }
                Term::Delay(body) => {
                    tasks.push(Task::Finish(Finish::Delay));
                    tasks.push(Task::Term(body, env, shift));
                }
                Term::Force(body) => {
                    tasks.push(Task::Finish(Finish::Force));
                    tasks.push(Task::Term(body, env, shift));
                }
                Term::Lambda {
                    parameter_name,
                    body,
                } => {
                    tasks.push(Task::Finish(Finish::Lambda(parameter_name.clone())));
                    tasks.push(Task::Term(body, env, shift + 1));
                }
                Term::Apply { function, argument } => {
                    tasks.push(Task::Finish(Finish::Apply));
                    tasks.push(Task::Term(argument, env, shift));
                    tasks.push(Task::Term(function, env, shift));
                }
                Term::Constr { tag, fields } => {
                    tasks.push(Task::Finish(Finish::Constr {
                        tag: *tag,
                        fields: fields.len(),
                    }));
                    tasks.extend(
                        fields
                            .iter()
                            .rev()
                            .map(|field| Task::Term(field, env, shift)),
                    );
                }
                Term::Case { constr, branches } => {
                    tasks.push(Task::Finish(Finish::Case {
                        branches: branches.len(),
                    }));
                    tasks.extend(
                        branches
                            .iter()
                            .rev()
                            .map(|branch| Task::Term(branch, env, shift)),
                    );
                    tasks.push(Task::Term(constr, env, shift));
                }
                Term::Constant(constant) => built.push(Term::Constant(constant.clone())),
                Term::Builtin(fun) => built.push(Term::Builtin(*fun)),
                Term::Error => built.push(Term::Error),
            },
            Task::Finish(finish) => {
                let term = match finish {
                    Finish::Delay => Term::Delay(pop(&mut built).into()),
                    Finish::Force => Term::Force(pop(&mut built).into()),
                    Finish::Lambda(parameter_name) => Term::Lambda {
                        parameter_name,
                        body: pop(&mut built).into(),
                    },
                    Finish::Apply => {
                        let argument = pop(&mut built).into();
                        let function = pop(&mut built).into();

                        Term::Apply { function, argument }
                    }
                    Finish::Constr { tag, fields } => Term::Constr {
                        tag,
                        fields: built.split_off(built.len() - fields),
                    },
                    Finish::Case { branches } => {
                        let branches = built.split_off(built.len() - branches);

                        Term::Case {
                            constr: pop(&mut built).into(),
                            branches,
                        }
                    }
                    Finish::Builtin {
                        fun,
                        forces,
                        arguments,
                    } => {
                        let arguments = built.split_off(built.len() - arguments);

                        let mut term = Term::Builtin(fun);

                        for _ in 0..forces {
                            term = term.force();
                        }

                        for argument in arguments {
                            term = term.apply(argument);
                        }

                        term
                    }
                };

                built.push(term);
            }
        }
    }

    pop(&mut built)
}

enum Task<'a> {
    Value(&'a Value),
    Term(&'a Term<NamedDeBruijn>, &'a Env, usize),
    Finish(Finish),
}

/// Builds a term from the terms its subterms were discharged into.
enum Finish {
    Delay,
    Force,
    Lambda(Rc<NamedDeBruijn>),
    Apply,
    Constr {
        tag: u64,
        fields: usize,
    },
    Case {
        branches: usize,
    },
    Builtin {
        fun: DefaultFunction,
        forces: u32,
        arguments: usize,
    },
}

fn pop(built: &mut Vec<Term<NamedDeBruijn>>) -> Term<NamedDeBruijn> {
    built
        .pop()
        .expect("every subterm is discharged before its parent")
}
