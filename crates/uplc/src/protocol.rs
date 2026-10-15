//! The major protocol versions at which the ledger changed how it decodes and
//! evaluates Plutus scripts, and what each ledger language may use at a given
//! protocol version (plutus-ledger-api's `PlutusLedgerApi.Common.Versions`).

use pallas_primitives::conway::Language;

pub const ALONZO_PROTOCOL_VERSION: u16 = 5;
pub const VASIL_PROTOCOL_VERSION: u16 = 7;
pub const VALENTINE_PROTOCOL_VERSION: u16 = 8;
pub const CHANG_PROTOCOL_VERSION: u16 = 9;
pub const PLOMIN_PROTOCOL_VERSION: u16 = 10;
pub const VAN_ROSSEM_PROTOCOL_VERSION: u16 = 11;
pub const DIJKSTRA_PROTOCOL_VERSION: u16 = 12;

/// The most fields a `constr` term may have in a script the ledger decodes
/// (`maxConstrFields`).
pub const MAX_CONSTR_FIELDS: usize = 1024;

/// The most type tags a constant's type may take to encode in a script the
/// ledger decodes (`maxHeaderSize`).
pub const MAX_CONSTANT_TYPE_SIZE: usize = 32;

/// The protocol version that introduced a ledger language
/// (`ledgerLanguageIntroducedIn`).
pub fn language_introduced_in(language: &Language) -> u16 {
    match language {
        Language::PlutusV1 => ALONZO_PROTOCOL_VERSION,
        Language::PlutusV2 => VASIL_PROTOCOL_VERSION,
        Language::PlutusV3 => CHANG_PROTOCOL_VERSION,
    }
}

/// Whether the ledger runs scripts of a ledger language whose program has
/// UPLC version `version` at a protocol version (`plcVersionsAvailableIn`).
/// UPLC 1.1.0 brought `constr` and `case`, and 1.2.0 `case` on Data.
pub fn is_uplc_version_available(
    language: &Language,
    protocol_major_version: u16,
    version: (usize, usize, usize),
) -> bool {
    let introduced_in = match (language, version) {
        (_, (1, 0, 0)) => language_introduced_in(language),
        (Language::PlutusV3, (1, 1, 0)) => CHANG_PROTOCOL_VERSION,
        (_, (1, 1, 0)) => VAN_ROSSEM_PROTOCOL_VERSION,
        (_, (1, 2, 0)) => DIJKSTRA_PROTOCOL_VERSION,
        _ => return false,
    };

    protocol_major_version >= introduced_in
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `plcVersionsIntroducedIn` at plutus 1b03c165.
    #[test]
    fn uplc_versions_become_available_as_plutus_ledger_api_says() {
        let introduced_in = [
            (Language::PlutusV1, (1, 0, 0), ALONZO_PROTOCOL_VERSION),
            (Language::PlutusV1, (1, 1, 0), VAN_ROSSEM_PROTOCOL_VERSION),
            (Language::PlutusV1, (1, 2, 0), DIJKSTRA_PROTOCOL_VERSION),
            (Language::PlutusV2, (1, 0, 0), VASIL_PROTOCOL_VERSION),
            (Language::PlutusV2, (1, 1, 0), VAN_ROSSEM_PROTOCOL_VERSION),
            (Language::PlutusV2, (1, 2, 0), DIJKSTRA_PROTOCOL_VERSION),
            (Language::PlutusV3, (1, 0, 0), CHANG_PROTOCOL_VERSION),
            (Language::PlutusV3, (1, 1, 0), CHANG_PROTOCOL_VERSION),
            (Language::PlutusV3, (1, 2, 0), DIJKSTRA_PROTOCOL_VERSION),
        ];

        for (language, version, protocol_major_version) in introduced_in {
            assert!(
                !is_uplc_version_available(&language, protocol_major_version - 1, version),
                "{language:?} {version:?} before protocol version {protocol_major_version}"
            );
            assert!(
                is_uplc_version_available(&language, protocol_major_version, version),
                "{language:?} {version:?} at protocol version {protocol_major_version}"
            );
        }

        for version in [(0, 0, 0), (1, 0, 1), (1, 3, 0), (2, 0, 0)] {
            assert!(!is_uplc_version_available(
                &Language::PlutusV3,
                u16::MAX,
                version
            ));
        }
    }
}
