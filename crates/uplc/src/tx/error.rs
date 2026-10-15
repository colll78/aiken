use crate::{
    TransactionInput,
    machine::{self, Trace, cost_model::ExBudget},
};
use pallas_primitives::conway::Language;

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub enum Error {
    #[error("{0}")]
    Address(#[from] pallas_addresses::Error),
    #[error("only shelley reward addresses can be a part of withdrawals")]
    BadWithdrawalAddress,
    #[error("{0}")]
    FlatDecode(#[from] pallas_codec::flat::de::Error),
    #[error("{0}")]
    FragmentDecode(#[from] pallas_primitives::Error),
    #[error("{}{}", .0, format_traces(.2))]
    Machine(machine::Error, ExBudget, Vec<Trace>),
    #[error(
        "the evaluation finished but the result value is not valid: PlutusV3 scripts must return unit{}",
        format_traces(.1)
    )]
    InvalidReturnValue(ExBudget, Vec<Trace>),

    #[error("{0} unexpected bytes follow a PlutusV3 script")]
    ScriptTrailingBytes(usize),
    #[error(
        "UPLC version {}.{}.{} is not available to {language:?} at protocol version {protocol_major_version}",
        .version.0, .version.1, .version.2
    )]
    UplcVersionNotAvailable {
        language: Language,
        version: (usize, usize, usize),
        protocol_major_version: u16,
    },
    #[error("native script can't be executed in phase-two")]
    NativeScriptPhaseTwo,
    #[error("can't eval without redeemers")]
    NoRedeemers,
    #[error(
        "missing and/or unexpected validator(s) and/or redeemer(s)\n{:>13} {}\n{:>13} {}",
        "Missing",
        if .missing.is_empty() { "ø".to_string() } else { .missing.join(&format!("\n{:>14}", "")) },
        "Unexpected",
        if .extra.is_empty() { "ø".to_string() } else { .extra.join(&format!("\n{:>14}", "")) },
    )]
    RequiredRedeemersMismatch {
        missing: Vec<String>,
        extra: Vec<String>,
    },
    #[error("extraneous redeemer")]
    ExtraneousRedeemer,
    #[error("resolved Input not found")]
    ResolvedInputNotFound(TransactionInput),
    #[error("redeemer points to a non-script withdrawal")]
    NonScriptWithdrawal,
    #[error("stake credential points to a non-script withdrawal")]
    NonScriptStakeCredential,
    #[error("the designated procedure defines no guardrail script")]
    NoGuardrailScriptForProcedure,
    #[error("cost model not found for language\n{:>13} {:?}", "Language", .0)]
    CostModelNotFound(Language),
    #[error("unsupported era, please use Conway\n{:>13} {0}", "Decoder error")]
    WrongEra(#[from] pallas_codec::minicbor::decode::Error),
    #[error("byron address not allowed when PlutusV2 scripts are present")]
    ByronAddressNotAllowed,
    #[error("inline datum not allowed when PlutusV1 scripts are present")]
    InlineDatumNotAllowed,
    #[error("script and input reference not allowed in PlutusV1")]
    ScriptAndInputRefNotAllowed,
    #[error("address doesn't contain a payment credential")]
    NoPaymentCredential,
    #[error("missing required datum\n{:>13} {}", "Datum", hash)]
    MissingRequiredDatum { hash: String },
    #[error("missing required script\n{:>13} {}", "Script", hash)]
    MissingRequiredScript { hash: String },
    #[error("missing required inline datum or datum hash in script input")]
    MissingRequiredInlineDatumOrHash,
    #[error("redeemer points to an unsupported certificate type")]
    UnsupportedCertificateType,
    #[error("failed script execution\n{:>13} {}", format!("{}[{}]", tag, index), err)]
    RedeemerError {
        tag: String,
        index: u32,
        err: Box<Error>,
    },
    #[error("missing script for redeemer")]
    MissingScriptForRedeemer,
    #[error("failed to apply parameters to Plutus script")]
    ApplyParamsError,
    #[error("validity start or end too far in the past")]
    SlotTooFarInThePast { oldest_allowed: u64 },
}

fn format_traces(traces: &[Trace]) -> String {
    traces
        .iter()
        .map(|trace| trace.to_string())
        .map(|trace| {
            format!(
                "\n{:>13} {}",
                "Trace",
                if trace.contains('\n') {
                    trace
                        .lines()
                        .enumerate()
                        .map(|(ix, row)| {
                            if ix == 0 {
                                row.to_string()
                            } else {
                                format!("{:>13} {}", "", row)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    trace.to_string()
                }
            )
        })
        .collect::<Vec<_>>()
        .join("")
}
