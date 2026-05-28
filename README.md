# Relational Rank Geometry in Transformers

This repository contains the paper source and replication materials for:

**Relational Rank Geometry in Transformers: Detecting and Steering Hidden-State Relation Frames**  
Mazen Kobrosly, Independent Researcher

The paper studies whether multi-token relation states in transformer hidden activations have a rank-indexed orientation signature, and whether the corresponding relation-frame geometry can be used as an intervention target.

## Repository layout

- `paper/relational_rank_geometry_in_transformers.pdf`  
  Compiled paper PDF.

- `arxiv_source/`  
  Clean LaTeX source tree for arXiv. This is the extracted source package that compiles from `main.tex`.

- `release_assets/relational_rank_geometry_arxiv_source.zip`  
  Clean arXiv source zip. This is the file intended for arXiv source upload.

- `release_assets/RRG_replication_materials_20260524.tar.gz`  
  Public replication materials bundle with prompt banks, run configs, runner scripts, row-level CSVs, path-quality tables, bootstrap summaries, multi-template outputs, site/order controls, manifests, and SHA256 hashes.

- `checksums/`  
  SHA256 checksums and manifests for release assets and replication materials.

- `requirements.txt`  
  Minimal Python package list for inspecting and rerunning the public runner scripts in a suitable GPU/model environment.

- `LICENSE-CODE-MIT` and `LICENSE-CONTENT-CC-BY-4.0`  
  Explicit dual-license files for code/scripts and paper/artifact content.

## Replication materials

The replication bundle contains the local artifacts used to audit the paper's reported results. Model checkpoints are not redistributed. The paper lists the public checkpoint IDs and the local loading paths used for the reported runs.

A superseded RTXPRO 405B pilot run is excluded; the reported site/order rows use the validated H100/H100-SXM 70B and 405B artifacts.

The GitHub bundle is a cleaned public copy of the replication materials: Python bytecode caches, cache-only manifests, superseded pilot artifacts, and operating-system metadata are omitted, and the top-level repository and bundle names use neutral public labels. It includes a `headline_table_sources/` map for the controlled-arity headline table and held-out audit. Some historical local paths and run-directory names remain inside archived manifests, logs, and CSV provenance fields to preserve exact audit traceability. The included runner scripts and lightweight helper modules are sufficient for CLI inspection; full reruns require local model checkpoint access and a compatible GPU environment. The archival DOI record remains on Zenodo.

## Checksums

The release asset checksums are recorded in:

```text
checksums/release_assets_SHA256SUMS.txt
```

The expanded replication-materials manifest and checksums are recorded in:

```text
checksums/replication_FILE_MANIFEST.txt
checksums/replication_SHA256SUMS.txt
```

## License

Code and scripts are MIT licensed; see `LICENSE` and `LICENSE-CODE-MIT`. Paper text, figures, prompt/data artifacts, and documentation are CC BY 4.0 unless otherwise noted; see `LICENSE-CONTENT-CC-BY-4.0`. Archives may contain materials under both licenses according to file type. The `CITATION.cff` license field refers to the citable paper/artifact record; code licensing is governed by the MIT license files. Model checkpoints are not redistributed.

## Citation

See `CITATION.cff`.

Zenodo all-versions DOI: <https://doi.org/10.5281/zenodo.20373871>

The arXiv identifier will be added after the arXiv submission is live.
