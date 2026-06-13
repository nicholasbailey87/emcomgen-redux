# results

Local `output_root` target. Results are arranged by experiment as
`<experiment>/<config>_seedN/` (e.g. `transformer/transformer_seed0/metrics.csv`).

On Hyperion the real `output_root` is `~/archive/results` (see `config.json`); this
directory is the local fallback. Contents are gitignored except this README.
