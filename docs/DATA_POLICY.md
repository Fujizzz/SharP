# Data and third-party resource policy

The public repository does not redistribute PHEME, LIAR, Twitter15/16, prompt-seed, or third-party lexicon files. The original private reconstruction grouped these resources under `data/`; the public release retains only source code, schemas, statistics, preparation instructions, and the research paper.

## PHEME

- Project release page: <https://www.pheme.eu/2016/06/13/pheme-rumour-dataset-support-certainty-and-evidentiality/>
- Dataset record referenced by the literature: <https://doi.org/10.6084/m9.figshare.4010619.v1>
- The data contains Twitter-derived text. The upstream dataset terms and applicable platform terms should be checked before republishing processed tweet text.

## LIAR

- Paper: <https://aclanthology.org/P17-2067/>
- The paper describes LIAR as publicly available for research, but this review did not find an explicit license file in the local dataset copy. Do not infer a software-style license from public availability.

## Twitter15/16

- The local raw copy identifies the original release as `rumdetect2017.zip`.
- Source tweets are Twitter-derived content. The safest public-repository approach is to publish preprocessing code, ids/manifests where permitted, and download instructions rather than a repackaged corpus of tweet text.

## NRC and other lexical resources

The handcrafted feature module uses NRC affect/emotion data plus morality, imageability, and hyperbolic lexicons. These resources may have terms independent of this code and require separate review before public distribution.

## Release decision

This repository uses the code-only public release mode: code, paper, schemas, statistics, and download/preparation instructions are public; dataset text, prepared tensors, prompt seeds, and third-party lexicon files are excluded from the entire public Git history.

This note is a release-safety record, not legal advice.
