# Data & Ethics Statement

This repository contains human-subjects data: psychometric responses (HADS,
STAI-S, BFI-10, Fear Questionnaire) and physiological recordings (heart rate,
HRV, IBI, electrodermal activity, eye tracking) collected from 10 adult
participants across three sessions. Because it includes health-related
measurements, it is handled as special-category personal data.

## Legal basis and data protection
This study was conducted, and the data are shared, in compliance with the EU
General Data Protection Regulation (Regulation (EU) 2016/679, "GDPR") and the
French Data Protection Act (Loi n° 78-17 du 6 janvier 1978, "Informatique et
Libertés", as amended). No separate institutional review board (IRB) number
applies; governance rests on the data-protection framework above together with
the participants' explicit consent.

- Lawful basis: the participants' explicit, informed consent (GDPR Art. 6(1)(a)
  and, for special-category health data, Art. 9(2)(a)).
- Principles applied: data minimisation, purpose limitation, and release of only
  pseudonymised records (GDPR Art. 5).

## Informed consent
All participants gave written informed consent before participation, including
explicit consent for the pseudonymised data to be shared openly for research and
educational purposes. Participants were free to withdraw at any time without
penalty.

## De-identification
- The released data contains **no direct identifiers** (no names, contact
  details, dates of birth, or device identifiers).
- Participants are referred to only by non-reversible pseudonymous codes
  (e.g. `01`, `02`).
- Recording timestamps are retained for time-series analysis. To minimise the
  residual re-identification risk of absolute appointment dates/times, run
  [`scripts/deidentify_timestamps.py`](scripts/deidentify_timestamps.py)
  (`--apply`) to shift each session to a relative origin (`2000-01-01`),
  preserving within-session alignment while removing the wall-clock date/time.
  The tool is dry-run by default.
- If you believe any released field could re-identify a participant, please open
  an issue and it will be removed.

## Data-subject rights
Participants retain their GDPR rights of access, rectification, erasure, and
objection. Requests can be made via the contact below.

## Permitted use
This data is released under the repository's license for **research and
educational purposes** only. Do not attempt to re-identify participants or use
the data to make decisions about any individual.

## Contact
Data controller / questions about this statement: Urme Bose — urme.bose1@gmail.com.
