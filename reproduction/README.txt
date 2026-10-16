Query Preservation under Version Selection
reproduction

Requirements
  CPython 3.14 on a POSIX system with numpy 2.5.3.  The package is
  self-contained: every input is included and no network access is used.

Command
  python3.14 -B reproduce.py

  The command copies code/ and data/ into runs/<UTC timestamp>/ and runs five
  stages there: the primary query pipeline, its diagnostics, a policy-array
  digest, the transformed-audio controls, and the cross-domain transfer
  experiment.  The shipped tree is never modified.  Each stage has a
  120-second CPU limit; use --cpu-limit 1800 if the machine is slow.

Comparison
  Each result file under runs/.../results/ is compared with expected.json,
  which holds the corresponding reference result.  Execution records,
  environment fields, absolute paths and wall-clock timings are dropped before
  comparison; integers and text must then agree exactly and floats exactly too,
  since this package reproduces without drift.  A CSV is compared as a
  value-sorted row multiset, because row order is not part of the contract.
  PASS and CHECK.json indicate a full match.

Expected counts
  100,000 retained submission rows with 4 descriptor columns; 2,117 raster
  cells over 2 archive versions and 16 years; 22 compared result files.

Data
  data/official_sample/ holds the retained AcousticBrainz feature and metadata
  values (CC0).  data/audio/ and data/raster_panel.npz hold the transfer
  experiment inputs.  data/control_measurements.json holds the transformed-audio
  control measurements.  Sources and terms are in SOURCE.txt and LICENSE.txt.

Files
  reproduce.py      the driver
  expected.json     expected result content, keyed by run-relative path
  code/             the analysis programs
  data/             the inputs
  RESULTS.tsv       paper item to program, line, output and input
  ENVIRONMENT.json  the recorded release environment
