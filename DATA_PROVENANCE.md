# Data provenance

## Inventory

| Source | Contents | Status |
| :--- | :--- | :--- |
| `data/case-study/raw` | 12 semicolon-delimited streams | One recording set; summaries numerically match group row P02 |
| `data/individual/raw` | 12 different streams | Participant mapping undocumented; not duplicate files |
| `data/*/processed` | Cardiac, eye and historical derived CSVs | Sampled channels; contracts below |
| `data/*/psychometric` | Item responses and question timestamps | 88 items per current session; original wording retained |
| `data/group_results` | 3 tables, P01 to P10, 3 sessions | Released values; most raw source mappings unavailable |

**51 CSVs, 24 raw TXT files, 33 notebooks.** Suffixes 01,02,03 identify sessions; unsuffixed streams supply one baseline per modality. Analyses reuse it; three independent baselines are not documented.

## Schemas and units

| Fields | Meaning and limits |
| :--- | :--- |
| `reltime` | Seconds from that recording's origin; streams have independent origins |
| `datetime` | Calendar time; sensor exports are naive, question times explicitly UTC |
| `iSensor` | Channel identifier; current sample metrics keep channels separate |
| `ibi` | Reported interval in ms; repeated sampled values are not established unique NN beats |
| `heart_rate`, `confidence` | Reported bpm and source confidence |
| `pupil`, `pupilQ`, `gazeQ` | Reported pupil mm and quality 0 to 1; current metrics require finite positive readings and stated quality bounds |
| `gazeDir.*`, `gaze_*` | Original/renamed gaze components; legacy 0.01 rule is vector distance per sample, not angular velocity |
| `fixation`, `fixation_id`, `duration` | Historical sample annotations; current event tables give one row per run, seconds, gap/quality handling and censoring |
| `sdnn`, `rmssd` | Historical names; current interval-sample calculations use 30-sample windows and 29 internal adjacent differences |
| `Time(s)` | Response duration in seconds for current question exports |
| `Question Start Time`, `Question Answer Time` | Interval boundaries; no nearest-start or cross-clock guess |

Current questionnaires contain HADS 14, STAI-S 20, STAI-T 20, BFI 10 and FQ 24 items. `Answer` coding, reversals and historical form identity need a separate scoring contract; notebooks do not assign diagnostic categories.

`individual/psychometric/Psychometric_Test_Results_00.csv` is a malformed 88-row legacy export: `Time(s)` holds a start timestamp, `Question Start Time` a later timestamp, `Score` a duration and `Answer Time` is blank. It is retained, inspected explicitly and rejected by the canonical question loader.

`*_modified.csv`, `QQ.csv`, `QQ2.csv` and `QQHRV.csv` are historical derivatives. Their dates and anxiety/decrease flags remain source evidence, without current diagnostic endorsement.

The three `*_modified.csv` files and the legacy `*_00.csv` omit a leading header. Calendar-shift copies retain the inferred leading field as `source_index`; source bytes and historical column labels remain unchanged.

## Reconciliation

```sh
python -m pipeline.build_group_summaries --check
python -m pipeline.build_group_summaries --output-root /tmp/mms-review
```

The legacy recipe pools raw sample SDs without filtering. It is numerically consistent with P02; this does not independently prove identity. With `rtol=0`, absolute tolerances are **1e-9** for interval/pupil SD and **3e-5 seconds** for duration SD, covering known rounding drift. Unexpected drift fails with exit 1. Check mode writes nothing.

Review outputs include separate per-channel filtered sample metrics and source/settings hashes. They do not replace group values or establish beat provenance. Invalid readings affect legacy variance; clinical meaning cannot be recovered from a variance spike alone.

## Known equality patterns

| Pattern | Released value | Status |
| :--- | ---: | :--- |
| P01 interval SD, S2=S3 | 65.39 | Suspected duplicate; report lists 1012.69 for S3 |
| P07=P08 duration SD, S1 | 4.409281… | Suspected duplicate; underlying raw records unavailable |

Equality is not proof of copying. Tests constrain exact IDs, values and multiplicities; new duplicates fail. Original cells remain unchanged.

## Clocks and outputs

- Preserve naive/aware times. Question joins require acquisition-zone metadata; `MMS_SENSOR_TIMEZONE` supplies an explicit mapping, never an inferred one.
- Producers use `MMS_OUTPUT_ROOT` outside source data; absent means no export. `MMS_DATA_ROOT` selects a directory containing `case-study`, `individual` and `group_results`.
- Wheels contain code, not participant CSVs. Loaders accept `data_root=`; missing data gives a clear error.

[Report status](docs/REPORT_STATUS.md) · [Figure provenance](docs/FIGURES.md) · [Consent and residual identifiers](DATA_ETHICS.md)
