# Seeded multi-corpus encoder transfer: verified negative result

Kaggle `truongbaokien/5g-ni-reproducible-multi-corpus-encoder-transfer`, version 1,
completed using source `d80a05cb318f103cfdf3a96c16f2b916669b5062`.
Archive SHA256: `4457a5c9acc19b956a3c2465521adc964fb9840a8f58a005b0673efb5ba02aac`.
The importer verified all 57 artifact files. Neither production artifacts nor
observations from index 504 onward were used for this research.

## Confirmation fold (reused development)

MAE uses physical DLPRB usage units; WAPE is shown as a percentage.
The comparator is the unchanged historical MLP/hurdle blend.

| Horizon | Multi-corpus MAE | WAPE | MAE gain vs comparator |
|---|---:|---:|---:|
| 1h | 0.237791 | 40.3426% | -0.2538% |
| 6h | 0.251097 | 43.3043% | +0.9577% |
| 24h | 0.260244 | 42.6011% | -0.3720% |

Multi-corpus transfer loses to the comparator on five of six fold/horizon
comparisons. It fails the predeclared no-regression/material-gain gate and does
not approach WAPE below 10%. No model promotion is justified.
Compute-matched target self-pretraining also fails the gate. The separate cold
seed-17 architecture job has now completed and passed the same verifier.
All target cold tensors match bitwise across these jobs. Multi-corpus transfer
reduces MAE by 0.18–0.64% in five of six comparisons with the cold architecture,
but increases confirmation-fold 1h MAE by 0.2815%. This is a small mixed gain,
not the material improvement requested. See the matched-cold comparison CSV.

## Verification

Both cases passed full metric reconstruction, fit-prefix preprocessing,
checkpoint stopping/termination checks, and encoder-only transfer isolation.
Every source run completed 5,120 optimizer updates. All target runs completed
early stopping rather than reaching their epoch cap. Across both cases, 2,304
history-only beam predictions were independently replayed on CPU; maximum
CPU/GPU difference was 0.00000190735, below the unchanged 0.0001 limit.

Cold initialization is bitwise reproducible from the declared seed. Windows
uniform initialization originally differed by at most 1.49e-8 from Linux because
of multiply/add rounding. Replaying the hosted FMA arithmetic reproduces all
source and target cold tensors exactly; no initialization tolerance was added.
Transfer comparisons within archived checkpoints remain bitwise exact.

## Limitations and next evidence

This is repeatedly reused development, not independent generalization evidence.
Nine source domains represent four external collections; Vigo subsets are not
independent operators. Publisher scaling scope for Vigo remains unknown.
More external data has not yet demonstrated a material accuracy gain.
Telecom Italia is being prepared separately and was absent from this frozen run.
Do not repeat fitting merely to change presentation or obtain a better seed.

Machine-readable comparisons, termination records and verification evidence are
in the adjacent `cell_multisource_v2_*` reports.
