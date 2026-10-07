# Report status

Original PDFs are preserved byte for byte. Locations below use PDF pages, with printed pages where different. Current code/tables take precedence for reproducible calculations, without rewriting historical observations.

| Report | Status |
| :--- | :--- |
| [Technical report](reports/Multimodal%20Multisensor%20Technical%20Report.pdf), 12 pages | Historical methods/results |
| [Thesis](reports/Thesis%20Report.pdf), 53 pages | Protocol, analyses and consent appendices |
| [Poster](reports/CYPSY_Poster.pdf), 1 page | Proposed protocol and expected results |

## Corrections and source gaps

| Location | Correction or unresolved issue |
| :--- | :--- |
| Technical p3 | Eye-open polarity above 1 conflicts with old code's below/equal1 closure. Units/polarity need confirmation. Current blink output is a stated heuristic, requiring observed reopening and gap handling. Gaze distance was not angular velocity. |
| Technical p6 | P01 S3 interval SD is 1012.69 versus 65.39 in CSV. Original raw source unavailable; neither value is guessed or silently replaced. |
| Technical p8 to 11 | k=3 agreement, PCA-first clustering, HADS validation and stable-trait claims exceed code evidence. Case-study clusters timestamp samples from one person, without clinical or participant-category validation. |
| Thesis p25, printed 28 | Some p values exceed 1; 0.00 is rounded. Original statistical output is needed to recover lost exponents/precision; replacements are not invented. |
| Thesis p37 to 38, printed 40 to 41 | Available eye/cardiac analyses differ from planned GSR/facial integration. Diagnostic/stable-trait conclusions exceed current 10-person uncertainty. |
| Thesis p4 to 5, printed 7 to 8 | Team-only protocol differs from the author's later sharing-consent account. No review identifier is independently established. |
| Thesis p46, printed 49 | Annexes E/F explicitly record photo consent. |
| Poster | Target N=10, expected results, planned CER submission. Tobii/Empatica/PHQ/ACEs and ages 18 to 30 differ from later methods. No approval or measured results established. |

## Bibliography

The Green/Harlow eye-tracking attribution to *Journal of Anxiety Disorders 73:102233* is incorrect. The identifier belongs to McKay, Yang, Elhai and Asmundson's 2020 COVID/disgust study, DOI [10.1016/j.janxdis.2020.102233](https://pmc.ncbi.nlm.nih.gov/articles/PMC7194061/); it does not support the attributed eye-tracking claim. Historical Johnson/Bennett titles remain unverified.

## Current evidence

- **ICC(1,1):** group tables give 0.22, 0.45, 0.61 with wide 95% intervals; no anxiety diagnosis or stable-trait validation.
- **Sample units:** repeated sampled cardiac intervals, without unique NN-beat timing; durations are within-session SDs in seconds.
- **Source gaps:** equality patterns, missing clock metadata and undocumented participant mapping limit independent reconstruction.

[Data contract](../DATA_PROVENANCE.md) · [Figures](FIGURES.md)
