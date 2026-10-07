# Data and consent

## Recorded consent

The author states that all 10 adults gave written informed consent, including later open sharing for research and education and withdrawal without penalty. This authored account is retained; the repository does not independently establish every consent or review record.

The thesis describes an earlier team-only protocol. The poster proposes a CER submission, without evidence of approval granted. No approval identifier is recorded here. GDPR and French data-protection law are the author's stated framework, rather than proof of institutional approval or independently verified compliance.

Recognizable session photographs have explicit consent described in the thesis, PDF page 46, printed 49, Annexes E/F. Photo consent is separate from tabular pseudonymization and questionnaire copyright.

## Actual release scope

| Material | Residual information |
| :--- | :--- |
| CSVs and raw TXT | Responses, physiological data, calendar timestamps and channel/device labels |
| QQ and modified CSVs | Historical dates, derived flags and responses |
| Notebooks, PDFs and screenshots | Dates, device labels and context |
| Session photographs | Consented recognizable people |

Participant codes reduce naming in tables; they do not establish anonymity or make this entire release identifier-free. Two distinct raw folders are present; the individual folder's participant mapping is undocumented. Eye/cardiac records are available; GSR and facial outputs are absent.

## Timestamp copies

```sh
python -m scripts.deidentify_timestamps
python -m scripts.deidentify_timestamps --output-root /tmp/mms-time-copies
```

The tool validates all 51 CSVs before writing. It copies calendar-bearing CSVs only, shifting each file/clock domain to a relative calendar anchor. Raw TXT, notebooks, reports, media and non-time CSVs remain unchanged. Source files are never overwritten; malformed nonempty dates fail. Logs omit original dates and offsets.

File-local shifts preserve compatible within-file differences, not unestablished cross-stream synchronization. Copies retain sensitive responses and measurements; they are not an anonymous release.

## Use and contact

- Retain attribution and comply with [data terms](DATA_LICENSE.md), consent limits and [instrument rights](NOTICE.md). Do not reidentify participants or make individual decisions from these records.
- Report identifying evidence privately through [Security](https://github.com/urmeo/Multimodal-Multisensor/security/advisories/new). Public reproducibility issues should use synthetic examples.
- Data-subject access, correction, erasure and objection requests go to recorded controller Urme Bose, [@urmeo](https://github.com/urmeo), through a private channel.
