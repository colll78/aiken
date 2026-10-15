use pallas_primitives::conway::Language;
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};
use uplc::{
    ast::{DeBruijn, Name, NamedDeBruijn, Program},
    machine::{Error as MachineError, cost_model::ExBudget, runtime::VAN_ROSSEM_PROTOCOL_VERSION},
    parser,
};
use walkdir::WalkDir;

const PARSE_ERROR: &str = "parse error";
const EVALUATION_FAILURE: &str = "evaluation failure";

// Upstream renamed the parse failure marker; older fixtures still use PARSE_ERROR.
const PARSE_OR_DECODE_ERROR: &str = "parse/decode error";

// Case on Data is a PV12 (Dijkstra) feature; these fixtures fail under PV11 by design.
const PV12_ONLY: &[&str] = &["term/constant-case/data"];

// Upstream computes fixtures in counting mode, where budgets saturate at `maxBound`. Every
// restricting evaluator, the ledger included, necessarily runs out of budget on such a program,
// so that is what these fixtures are held to here.
const OUT_OF_BUDGET: &str = "out of budget";

// As close to the reference counting mode as a restricting machine gets.
const UNLIMITED_BUDGET: ExBudget = ExBudget {
    mem: i64::MAX,
    cpu: i64::MAX,
};

// test_data/conformance/v3 is a byte-for-byte copy of
// plutus-conformance/test-cases/uplc/evaluation from IntersectMBO/plutus revision
// 1b03c165b854548cc8de77e7bb01a8d96b9c74f5 (minus the repository's helper script and
// multiScalarMul input dump). This PlutusV3 parameter array is the first 350 entries (everything
// up to batch 6, i.e. the Van Rossem set) of what that revision's
// plutus-core/cost-model/data/{builtinCostModelE,cekMachineCostsE}.json give in `ParamName` order.
// Both `.uplc` and `.flat` fixtures are checked: `cargo test -p uplc --test conformance`.
const V3_PV11_COSTS: &[i64] = &[
    100788, 420, 1, 1, 1000, 173, 0, 1, 1000, 59957, 4, 1, 11183, 32, 201305, 8356, 4, 16000, 100,
    16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 100, 100, 16000, 100, 94375, 32,
    132994, 32, 61462, 4, 72010, 178, 0, 1, 22151, 32, 91189, 769, 4, 2, 85848, 123203, 7305, -900,
    1716, 960, 57, 85848, 0, 1, 1, 1000, 42921, 4, 2, 30623, 28755, 75, 1, 898148, 27279, 1, 51775,
    558, 1, 39184, 1000, 60594, 1, 141895, 32, 83150, 32, 15299, 32, 76049, 1, 13169, 4, 22100, 10,
    28999, 74, 1, 28999, 74, 1, 43285, 552, 1, 44749, 541, 1, 33852, 32, 68246, 32, 72362, 32,
    7243, 32, 7391, 32, 11546, 32, 85848, 123203, 7305, -900, 1716, 960, 57, 85848, 0, 1, 90434,
    519, 0, 1, 74433, 32, 85848, 123203, 7305, -900, 1716, 960, 57, 85848, 0, 1, 1, 85848, 123203,
    7305, -900, 1716, 960, 57, 85848, 0, 1, 955506, 213312, 0, 2, 270652, 22588, 4, 1457325, 64566,
    4, 20467, 1, 4, 0, 141992, 32, 100788, 420, 1, 1, 81663, 32, 59498, 32, 20142, 32, 24588, 32,
    20744, 32, 25933, 32, 24623, 32, 43053543, 10, 53384111, 14333, 10, 43574283, 26308, 10, 16000,
    100, 16000, 100, 962335, 18, 2780678, 6, 442008, 1, 52538055, 3756, 18, 267929, 18, 76433006,
    8868, 18, 52948122, 18, 1995836, 36, 3227919, 12, 901022, 1, 166917843, 4307, 36, 284546, 36,
    158221314, 26549, 36, 74698472, 36, 333849714, 1, 254006273, 72, 2174038, 72, 2261318, 64571,
    4, 207616, 8310, 4, 1293828, 28716, 63, 0, 1, 1006041, 43623, 251, 0, 1, 100181, 726, 719, 0,
    1, 100181, 726, 719, 0, 1, 100181, 726, 719, 0, 1, 107878, 680, 0, 1, 95336, 1, 281145, 18848,
    0, 1, 180194, 159, 1, 1, 158519, 8942, 0, 1, 159378, 8813, 0, 1, 107490, 3298, 1, 106057, 655,
    1, 1964219, 24520, 3, 607153, 231697, 53144, 0, 1, 116711, 1957, 4, 231883, 10, 1000, 24838, 7,
    1, 232010, 32, 321837444, 25087669, 18, 617887431, 67302824, 36, 390841, 30257, 47, 27, 219951,
    9444, 1, 1000, 284948, 143815, 4, 26, 29, 213283, 618401, 1998, 28258, 1, 1000, 38159, 2, 22,
    1000, 128595, 1, 14, 18, 1000, 383231, 14, 29,
];

fn expected_to_program(expected_file: &PathBuf) -> Result<Program<Name>, String> {
    let code = fs::read_to_string(expected_file).expect("Failed to read .uplc.expected file");

    if code.contains(PARSE_ERROR) || code.contains(PARSE_OR_DECODE_ERROR) {
        Err(PARSE_ERROR.to_string())
    } else if code.contains(EVALUATION_FAILURE) {
        Err(EVALUATION_FAILURE.to_string())
    } else {
        parser::program(&code).map_err(|_| code)
    }
}

fn expected_budget(expected_file: &Path) -> Result<ExBudget, String> {
    let code = fs::read_to_string(expected_file)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", expected_file.display()));
    let code = code.strip_suffix('\n').unwrap_or(&code);

    match code {
        PARSE_ERROR | PARSE_OR_DECODE_ERROR => Err(PARSE_ERROR.to_string()),
        EVALUATION_FAILURE => Err(EVALUATION_FAILURE.to_string()),
        _ => {
            let fields = code
                .strip_prefix("({cpu: ")
                .and_then(|fields| fields.strip_suffix("})"))
                .unwrap_or_else(|| {
                    panic!(
                        "invalid budget fixture syntax in {}: {code:?}",
                        expected_file.display()
                    )
                });
            let (cpu, mem) = fields.split_once("\n| mem: ").unwrap_or_else(|| {
                panic!(
                    "invalid budget fixture syntax in {}: {code:?}",
                    expected_file.display()
                )
            });
            let cpu = cpu.parse().unwrap_or_else(|err| {
                panic!("invalid CPU budget in {}: {err}", expected_file.display())
            });
            let mem = mem.parse().unwrap_or_else(|err| {
                panic!(
                    "invalid memory budget in {}: {err}",
                    expected_file.display()
                )
            });

            Ok(ExBudget { cpu, mem })
        }
    }
}

#[derive(Debug)]
struct ConformanceResult {
    result: Result<Program<NamedDeBruijn>, String>,
    budget: Result<ExBudget, String>,
}

fn actual_evaluation_result(
    file: &Path,
    language: &Language,
    protocol_major_version: u16,
    costs: &[i64],
) -> ConformanceResult {
    let code = fs::read_to_string(file).expect("Failed to read .uplc file");

    let program = match parser::program(&code) {
        Ok(program) => program,
        Err(_) => return ConformanceResult::failure(PARSE_ERROR),
    };

    // Upstream reports scope errors (free variables) as parse/decode errors.
    match program.try_into() {
        Ok(program) => evaluate(program, language, protocol_major_version, costs),
        Err(_) => ConformanceResult::failure(PARSE_ERROR),
    }
}

fn evaluate(
    program: Program<NamedDeBruijn>,
    language: &Language,
    protocol_major_version: u16,
    costs: &[i64],
) -> ConformanceResult {
    let version = program.version;
    let eval = program.eval_as_with_protocol(
        language,
        protocol_major_version,
        costs,
        Some(&UNLIMITED_BUDGET),
    );
    let cost = eval.cost();

    match eval.result() {
        Ok(term) => ConformanceResult {
            result: Ok(Program { version, term }),
            budget: Ok(cost),
        },
        // The machine scope-checks before evaluating, as the ledger does; upstream
        // reports that as a parse/decode error.
        Err(MachineError::OpenTermEvaluated(_)) => ConformanceResult::failure(PARSE_ERROR),
        Err(MachineError::OutOfExError(_)) => ConformanceResult::failure(OUT_OF_BUDGET),
        Err(_) => ConformanceResult::failure(EVALUATION_FAILURE),
    }
}

impl ConformanceResult {
    fn failure(reason: &str) -> Self {
        ConformanceResult {
            result: Err(reason.to_string()),
            budget: Err(reason.to_string()),
        }
    }
}

fn expected_flat_program(expected_file: &Path) -> Result<Program<DeBruijn>, String> {
    let bytes = fs::read(expected_file)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", expected_file.display()));

    match std::str::from_utf8(&bytes).map(str::trim_end) {
        Ok(PARSE_OR_DECODE_ERROR) => Err(PARSE_ERROR.to_string()),
        Ok(EVALUATION_FAILURE) => Err(EVALUATION_FAILURE.to_string()),
        _ => Ok(
            Program::<DeBruijn>::from_flat(&bytes).unwrap_or_else(|err| {
                panic!("invalid flat fixture {}: {err}", expected_file.display())
            }),
        ),
    }
}

fn actual_flat_evaluation_result(
    file: &Path,
    language: &Language,
    protocol_major_version: u16,
    costs: &[i64],
) -> ConformanceResult {
    let bytes = fs::read(file).expect("Failed to read .flat file");

    match Program::<DeBruijn>::from_flat(&bytes) {
        Ok(program) => evaluate(program.into(), language, protocol_major_version, costs),
        Err(_) => ConformanceResult::failure(PARSE_ERROR),
    }
}

fn budget_file(path: &Path) -> Option<PathBuf> {
    [
        path.with_extension("budget.expected"),
        path.with_extension("uplc.budget.expected"),
    ]
    .into_iter()
    .find(|file| file.exists())
}

fn saturates(budget_file: Option<&PathBuf>) -> bool {
    budget_file
        .and_then(|file| expected_budget(file).ok())
        .is_some_and(|budget| budget.cpu == i64::MAX || budget.mem == i64::MAX)
}

fn is_pv12_only(root: &str, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);

    PV12_ONLY.iter().any(|prefix| relative.starts_with(prefix))
}

fn plutus_conformance_tests(
    root: &str,
    language: Language,
    protocol_major_version: u16,
    costs: &[i64],
    compare_budgets: bool,
) {
    let mut failures = Vec::new();
    let mut programs = 0;

    let mut check = |what: &str, path: &Path, matches: bool, detail: String| {
        if !matches {
            failures.push(format!("{what} mismatch for {}\n{detail}", path.display()));
        }
    };

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();

        if is_pv12_only(root, path) {
            continue;
        }

        let budget_file = budget_file(path);
        let saturates = saturates(budget_file.as_ref());

        if path.extension().and_then(OsStr::to_str) == Some("flat") {
            programs += 1;

            let expected = if saturates {
                Err(OUT_OF_BUDGET.to_string())
            } else {
                expected_flat_program(&path.with_extension("flat.expected"))
            };
            let actual =
                actual_flat_evaluation_result(path, &language, protocol_major_version, costs)
                    .result
                    .map(Program::<DeBruijn>::from);

            check(
                "flat result",
                path,
                expected == actual,
                format!("expected: {expected:?}\n  actual: {actual:?}"),
            );
        }

        if path.extension().and_then(OsStr::to_str) == Some("uplc") {
            programs += 1;

            let expected_file = path.with_extension("uplc.expected");

            let actual = actual_evaluation_result(path, &language, protocol_major_version, costs);
            let expected = if saturates {
                Err(OUT_OF_BUDGET.to_string())
            } else {
                expected_to_program(&expected_file)
                    .map(|program| Program::<NamedDeBruijn>::try_from(program).unwrap())
            };

            check(
                "result",
                path,
                expected == actual.result,
                format!("expected: {expected:?}\n  actual: {:?}", actual.result),
            );

            if compare_budgets {
                let budget_file = budget_file
                    .unwrap_or_else(|| panic!("missing budget fixture for {}", path.display()));
                let expected_cost = if saturates {
                    Err(OUT_OF_BUDGET.to_string())
                } else {
                    expected_budget(&budget_file)
                };

                check(
                    "budget",
                    path,
                    expected_cost == actual.budget,
                    format!("expected: {expected_cost:?}\n  actual: {:?}", actual.budget),
                );
            }
        }
    }

    // Guards against running from a directory where the fixtures are not found,
    // which would otherwise pass without checking anything.
    assert!(programs > 0, "no conformance programs found under {root}");

    assert!(
        failures.is_empty(),
        "{} conformance failure(s) under {root}:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn plutus_conformance_tests_v2() {
    plutus_conformance_tests(
        "test_data/conformance/v2",
        Language::PlutusV2,
        11,
        &[
            100788, 420, 1, 1, 1000, 173, 0, 1, 1000, 59957, 4, 1, 11183, 32, 201305, 8356, 4,
            16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 100, 100,
            16000, 100, 94375, 32, 132994, 32, 61462, 4, 72010, 178, 0, 1, 22151, 32, 91189, 769,
            4, 2, 85848, 228465, 122, 0, 1, 1, 1000, 42921, 4, 2, 30623, 28755, 75, 1, 898148,
            27279, 1, 51775, 558, 1, 39184, 1000, 60594, 1, 141895, 32, 83150, 32, 15299, 32,
            76049, 1, 13169, 4, 22100, 10, 28999, 74, 1, 28999, 74, 1, 43285, 552, 1, 44749, 541,
            1, 33852, 32, 68246, 32, 72362, 32, 7243, 32, 7391, 32, 11546, 32, 85848, 228465, 122,
            0, 1, 1, 90434, 519, 0, 1, 74433, 32, 85848, 228465, 122, 0, 1, 1, 85848, 228465, 122,
            0, 1, 1, 955506, 213312, 0, 2, 270652, 22588, 4, 1457325, 64566, 4, 20467, 1, 4, 0,
            141992, 32, 100788, 420, 1, 1, 81663, 32, 59498, 32, 20142, 32, 24588, 32, 20744, 32,
            25933, 32, 24623, 32, 43053543, 10, 53384111, 14333, 10, 43574283, 26308, 10, 1293828,
            28716, 63, 0, 1, 1006041, 43623, 251, 0, 1, 16000, 100, 16000, 100, 962335, 18,
            2780678, 6, 442008, 1, 52538055, 3756, 18, 267929, 18, 76433006, 8868, 18, 52948122,
            18, 1995836, 36, 3227919, 12, 901022, 1, 166917843, 4307, 36, 284546, 36, 158221314,
            26549, 36, 74698472, 36, 333849714, 1, 254006273, 72, 2174038, 72, 2261318, 64571, 4,
            207616, 8310, 4, 100181, 726, 719, 0, 1, 100181, 726, 719, 0, 1, 100181, 726, 719, 0,
            1, 107878, 680, 0, 1, 95336, 1, 281145, 18848, 0, 1, 180194, 159, 1, 1, 158519, 8942,
            0, 1, 159378, 8813, 0, 1, 107490, 3298, 1, 106057, 655, 1, 1964219, 24520, 3, 607153,
            231697, 53144, 0, 1, 116711, 1957, 4, 231883, 10, 1000, 24838, 7, 1, 232010, 32,
            321837444, 25087669, 18, 617887431, 67302824, 36, 390841, 30257, 47, 27, 219951, 9444,
            1, 1000, 284948, 143815, 4, 26, 29, 213283, 618401, 1998, 28258, 1, 1000, 38159, 2, 22,
            1000, 128595, 1, 14, 18, 1000, 383231, 14, 29,
        ],
        // This root combines budget fixtures from incompatible upstream cost-model revisions
        // (legacy BLS fixtures and newer CEK/builtin fixtures). Result conformance remains valid;
        // budget conformance is enabled for the homogeneous V3 PV11 root below.
        false,
    )
}

#[test]
fn plutus_conformance_tests_v3() {
    plutus_conformance_tests(
        "test_data/conformance/v3",
        Language::PlutusV3,
        VAN_ROSSEM_PROTOCOL_VERSION,
        V3_PV11_COSTS,
        true,
    )
}
