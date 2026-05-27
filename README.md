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
  Public replication materials bundle with prompt banks, run configs, scripts, row-level CSVs, path-quality tables, bootstrap summaries, multi-template outputs, site/order controls, manifests, and SHA256 hashes.

- `checksums/`  
  SHA256 checksums and manifests for release assets and replication materials.

## Replication materials

The replication bundle contains the local artifacts used to audit the paper's reported results. Model checkpoints are not redistributed. The paper lists the public checkpoint IDs and the local loading paths used for the reported runs.

The valid 70B and 405B site/order control artifacts are included. The invalid RTXPRO 405B attempt is intentionally excluded from the public replication bundle.

The GitHub bundle is a cleaned public copy of the replication materials: Python bytecode caches and cache-only manifests are omitted, and internal working filenames have been replaced with neutral public names. The archival DOI record remains on Zenodo.

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

Code and scripts in this repository are released under the MIT license. The paper, figures, and release artifacts are released under CC BY 4.0 unless otherwise noted. Model checkpoints are not redistributed.

## Citation

See `CITATION.cff`.

Zenodo DOI: <https://doi.org/10.5281/zenodo.20373872>

The arXiv identifier will be added after the arXiv submission is live.
