# Public release checklist

- [x] Public project name and manuscript title are consistent.
- [x] A CPU smoke test uses generated test data and relative paths.
- [x] Direct runtime dependencies are documented and versioned.
- [x] Final benchmark, ablation, and case-study config families are indexed.
- [x] Data formats and required local filenames are documented.
- [x] Citation metadata include all manuscript authors.
- [x] Generated data, checkpoints, caches, and experiment outputs are ignored.
- [x] The author-selected MIT License is included.
- [x] Internal sweep orchestration and experiment-tracking integrations are
      omitted; retained configurations use local CSV logging.
- [x] Retained paper configurations contain no laboratory-specific absolute
      data paths.
- [x] Public configuration filenames and run names omit internal machine and
      run-date suffixes.
- [x] Lightweight CI validates public assets on supported Python versions.
- [x] Contribution guidance covers tests, data, credentials, and licensing.
- [x] A one-command MNIST example downloads data, trains DMT-Dens, and exports
      a two-dimensional PNG without using labels for representation learning.
- [ ] Authors verify all dataset redistribution permissions and publish stable
      accession/download links.
- [ ] Replace manuscript placeholders with the accepted article DOI when known.
- [ ] Create a tagged release and archive it with a DOI provider such as Zenodo.
- [ ] Run the clean-environment smoke test on Linux and record GPU/CUDA details.
