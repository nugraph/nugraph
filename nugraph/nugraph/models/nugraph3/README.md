# NuGraph3: how the pieces fit, and the evidence for them

This file explains the NuGraph3 changes that arrive as a stack of
pull requests, one idea per PR. Each PR adds its own section below, so this file
always describes what the code on that branch contains. Section numbers match
the PR positions in the stack.

Unless a section says otherwise, results are from the MicroBooNE open data
(`.gnn.h5` graphs), evaluated on validation events (usually 2,000).

## The model in one page

NuGraph3 works on one heterogeneous graph per event:

```
 hits (one node per wire hit, per plane)
   |  planar message passing (plane_net)
   v
 nexus nodes = spacepoints (3D points that join hits across planes)
   |  hit -> spacepoint, optional spacepoint <-> spacepoint (3D graph)
   v
 interaction node (one per event)
   |  and back: interaction -> spacepoints -> hits
   v
 decoders: semantic (hit class), filter (hit noise), instance (object
           condensation), vertex, event, and in this stack the nexus decoder
```

What this stack adds is a 3D stage between hits and particles:

1. **Spacepoints with truth** (08): a 3D graph over spacepoints, with
   labels saying which spacepoints are real (all hits from one particle) and
   which edges join the same particle.
2. **Nexus decoder** (09-14): scores every spacepoint (real or ghost) and every
   3D edge (same particle or not), helped by vertex votes and local directions
   and a better graph (Gabriel + stubs).
3. **Fragments**: the kept spacepoints joined by kept edges form *fragments*,
   pieces of single particles. Fragments are judged by purity at matched
   fragmentation: a wrong merge cannot be undone later, a split can.
4. **Particles** (not in this stack): a later stage merges fragments into
   particles (showers, delta rays onto muons) and links decays. That work
   builds on the fragment scores defined here.

Two further PRs feed the 3D stage back into the 2D one (semantic feedback,
15) and refine the event vertex from the spacepoints (21). The last three PRs
(23-25) encode each plane by its geometry and let NuGraph3 build its own
spacepoint candidates from hit crossings. Together these make the model
less tied to one detector. PRs 16-20 are training infrastructure.

## Measures used below

- **Grouping ARI**: adjusted Rand index of the connected components of kept
  spacepoint edges against true particles (threshold stated).
  **Average linkage (AL)**: the same with average-linkage clustering of edge
  scores, which is less sensitive to single wrong edges.
- **Containment @0.9**: for each true particle, the share of its labelled
  hits that lie in fragments it dominates with purity of at least 0.9,
  averaged over particles (`studies/hit_efficiency.py`). Purity of a fragment
  is the share of its labelled hits from its majority particle.
- **Ghosts removed / real kept**: ghost-filter efficiency at threshold 0.5.
- **Seed noise**: repeated runs with `--seed` differ by about 0.005-0.007 ARI,
  so smaller ARI differences are reported as "within noise".

## The stack

Each PR is based on the previous one (the row above it in this table), and
01 is based on `main`. *Needs* lists the PRs whose code a PR uses. The last
column says whether a PR needs the previous PR: "yes" if it does, "no" if it
sits there only to keep the stack linear. A "no" PR could move up to just after
the last PR it needs, perhaps with small textual conflicts.

| # | Branch | Change | Needs | Needs the previous PR? |
|---|---|---|---|---|
| 01 | `stack/01-regression-tests` | Phase 0 regression tests | main | base |
| 02 | `stack/02-oc-loss-batch` | Object condensation loss independent of batch size | 01 | yes |
| 03 | `stack/03-truth-edges-row` | Truth hit-particle edges from the hit row | 01 | no |
| 04 | `stack/04-input-norm-buffers` | InputNorm statistics in buffers, freeze flag saved | 01 | no |
| 05 | `stack/05-test-script` | scripts/test.py runs end to end | 01 | no |
| 06 | `stack/06-seeded-splits` | Seeded train/validation/test split | 01 | no |
| 07 | `stack/07-labeller` | Labeller label(), invisible hits, labelling errors | 01 | no |
| 08 | `stack/08-spacepoint-truth` | Spacepoint graph with truth labels | 01 | no |
| 09 | `stack/09-nexus-decoder` | Nexus decoder: spacepoint ghost and edge scores | 08 | yes |
| 10 | `stack/10-vertex-edge-features` | Vertex-relative edge features | 09 | yes |
| 11 | `stack/11-vertex-votes-directions` | Spacepoint vertex votes and directions | 10 | yes |
| 12 | `stack/12-gabriel-stubs-graph` | Gabriel+stubs spacepoint graph on the GPU | 09, 11 | yes |
| 13 | `stack/13-residual-blocks` | Residual message-passing blocks | 09 | no |
| 14 | `stack/14-edge-classifier-options` | Edge-classifier inputs and size | 11, 12 | no |
| 15 | `stack/15-semantic-feedback` | Semantic feedback from spacepoints to hits | 09 | no |
| 16 | `stack/16-precision-checkpoints` | --precision fix, 16-mixed, per-epoch checkpoints | 01 | no |
| 17 | `stack/17-multi-gpu` | Multi-GPU training outside SLURM | main | no |
| 18 | `stack/18-training-speed` | Fewer host syncs and a compiled stub kernel | 12, 14, 15 | no |
| 19 | `stack/19-oom-guard` | Out-of-memory guard | 12, 16 | no |
| 20 | `stack/20-sampler-length-clip` | BalanceSampler length fix and --gradient-clip | 01 | no |
| 21 | `stack/21-vertex-refine` | Nexus vertex refinement | 11 | no |
| 22 | `stack/22-filter-class-weights` | Per-class ghost-filter weights | 14 | no |
| 23 | `stack/23-plane-geometry` | Plane geometry encoding | main | no |
| 24 | `stack/24-sp-build` | Own spacepoint candidates from hit crossings | 23 | yes |
| 25 | `stack/25-sp-build-best` | Best-K spacepoint candidates per hit | 24 | yes |

The recipe that the evidence below refers to trains with
`--semantic --filter --nexus --sp-graph gabriel+stubs --sp-gpu
--nexus-semantic --nexus-balance-filter --nexus-log-length --nexus-stub-count
--nexus-edge-layers 2 --nexus-edge-width 64 --nexus-feedback --nexus-feats 64
--nexus-vertex-refine --nexus-vertex vote --nexus-direction
--nexus-filter-weight hadron=3 --sp-build scripts/geometry/microboone.json
--sp-build-best 2 --precision 16-mixed --save-every-epoch`. Every flag is off
by default, except residual blocks in the training CLI (13).

## 01 Phase 0 regression tests

**What.** Tests only, no library change. Small synthetic events
(`nugraph/tests/synthetic.py`) in the gen-2 HitGraphProducer layout, so the
tests need no input file, plus model, loss, decoder, training-script and
data-utility tests (`nugraph/tests/test_regression_nugraph3.py`,
`pynuml/tests/test_regression_pynuml.py`).

**How it fits.** Every known bug from the roadmap's Phase 0 list is a strict
`xfail` that names its cause. The fix PRs below (02-07, 16, 20) remove their
marker; a fix that is not marked fails CI as an XPASS. The synthetic events are
reused by most later tests.

**Guards that pass today (11).** An event's predictions do not depend on the
other events in its batch; semantic, filter, event, vertex, spacepoint and OC
background losses are unchanged when an event is duplicated in the batch;
CPU bf16 autocast forward and backward; `materialize` finds three separated
clusters; label `index()` round trip; truth edges are right when hit rows are
ordered by `hit_id`.

**Known bugs (16 xfail).** `--precision` never reaches the Trainer;
`16-mixed` rejected; OC potential grows with duplicated events (2.28, 5.19,
11.0 for 1, 2, 4 copies); OC loss with a hitless particle; particle loss halves
when the batch doubles; diffuse hits clustered (hard-coded class index); empty
particle batch vector; truth edges keyed on `hit_id`; `label()` broken in three
labellers; InputNorm freeze flag and Parameters; BalanceSampler with
`balance_frac=0` and its `__len__`; FeatureExtension with an isolated node.

**Run.** `pytest -rx nugraph/tests pynuml/tests`, and
`pytest --runxfail --tb=short ...` to watch each bug fail.

## 02 Object condensation loss independent of batch size

**What.** Two OC fixes (Phase 0 bugs 2 and 3).
- The potential term built a hit x condensation-point matrix over the whole
  batch, so hits were repelled by particles in *other* events. It is now
  masked to same-event pairs, using the hit `batch` vector.
- `RecallLoss(reduction="none")` divided each hit's loss by the number of
  valid hits, so the particle loss shrank as the batch grew. It now returns
  unnormalised per-hit losses; the normalisation (and its DDP average) moves to
  `RecallLoss.n_valid()`. The semantic loss value is unchanged.

**Effect** (synthetic event, 1 / 2 / 4 copies in one batch): potential 2.28 /
5.19 / 11.0 before, 2.28 / 2.28 / 2.28 after; particle loss 1.6e-2 / 8.0e-3 /
4.0e-3 before, 1.25 each after; semantic loss 2.50 each, before and after.

**Training evidence.** None yet. The single-event potential is unchanged, but
in training batches the cross-event repulsion disappears, which changes its
balance against the other losses. With `--particle-loss` (off by default) that
term is now ~100x larger, on the cross-entropy scale. No controlled training
comparison exists: the nexus runs below do not train the hit-level instance
head. A before/after comparison of instance ARI is still to be run.

**Notes.** The dense matrix is still built and then masked, so memory use is
unchanged; computing it event by event can follow once profiled.
`test_potential_repels_within_event_only` (suggested by Andrii) checks both
directions: within-event repulsion stays, cross-event repulsion goes.

## 03 Truth hit-particle edges from the hit row

**What.** pynuml's `HitGraphProducer` used each hit's `hit_id` as its node
index in the hit -> particle truth edges. That holds only while the hit table
is dense and ordered by `hit_id`. The edges now use the row index, as the
nexus edges already do (Phase 0 bug 6).

**Evidence.** The 40 MicroBooNE test-file events give identical truth edges
before and after (`hit_id` equals the row in every event), so existing files
are unaffected. The fix matters for other detectors' files and any hit
filtering. Tests shuffle the hit rows and drop every fifth hit; the old code
gives wrong or out-of-range particles.
