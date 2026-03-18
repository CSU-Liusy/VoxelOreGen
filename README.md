# VoxelOreGen: Voxel Orebody Prior Generation and 3D GAN Augmentation Framework (Fully Audited README)

## Abstract
This document is written based on actual directory inspection rather than assumptions. Its goal is to accurately describe the methodology, engineering structure, runtime entry points, and current artifact snapshot of VoxelOreGen. The project provides three core capabilities:
- Prior geological rule-based generation (legacy / staged / workflow / physics / hybrid).
- Unified main-entry orchestration (oregen -> gan-train -> gan-generate).
- 3D conditional WGAN-GP learning, sampling, and mesh export for orebody voxels.

## Current Snapshot (for this README)
- Scan time: 2026-03-16 15:00:57
- Total files: 1878
- Source code files: 10 Python files
- Output meshes: 1238 PLY, 619 OBJ
- Intermediate data: 0 NPZ, 0 NPY
- Bytecode cache: 10 PYC
- Key artifact counts: ore_body_*.ply = 619; ore_surface_*.obj = 619; ore_surface_*.ply = 619

## Methods and Workflow (Research Perspective)
### Overall Methodological Framework
- This project adopts a dual-engine architecture: “prior mineralization mechanisms + data-driven generation.” It first creates interpretable orebodies using geological/physical rules, then uses a 3D GAN to learn morphological distributions and augment samples.
- Why this design: pure rule-based methods are interpretable but limited in diversity; pure deep-learning methods are diverse but weak in geological constraints. Combining both preserves “geological plausibility” and “statistical diversity.”

### 1) Ore Prior Layer
#### Legacy path: empirical-rule incubation
- In `seed_generator`, geometric seeds are created first (main veins, branch veins, satellite veins), then `incubation_engine` schedules 20 rules for iterative incubation.
- Why this is done: it explicitly encodes the idea of “a mineralized geometric embryo first, then modified by fluids and structures,” reducing invalid samples caused by random initialization.
- Difference from other paths: legacy is more “rule-driven,” with lower computational cost and higher speed, suitable for producing large baseline sample sets.
- Main benefits: quickly generates large batches of initial orebody drafts, with controllable style bias through rule weights.

#### Staged path: causal process layering
- Executes from stage0 to stage5 in sequence. The core chain is: background initialization and structural weakening -> source activation and complexation -> Darcy transport and fracturing -> boiling/mixing precipitation -> zonation and alteration -> supergene enrichment and ductile overprint.
- Why this is done: it decomposes the source-transport-deposition-post-modification causal chain into auditable stages, making it easier to track each stage’s contribution to final morphology.
- Difference from other paths: staged emphasizes geological chronology and mechanistic interpretability more than legacy, suitable for research analysis and stage-wise comparative experiments.
- Main benefits: outputs stage logs and stage snapshots, supporting mechanism visualization and ablation analysis in papers.

#### Workflow path: engineering-oriented seven-stage pipeline
- The seven stages include: grid/lithology initialization, core source localization, anisotropic ellipsoidal decay, permeability-modulated noise, fault offset superposition, probabilistic micro-events (fracturing/boiling/skarn), and normalization + isosurface preparation.
- Why this is done: to build a stable, repeatable, batch-runnable engineering pipeline while preserving geological meaning.
- Difference from other paths: workflow is export-friendly (directly targets PLY/OBJ), has centralized parameters, and strong default robustness, making it suitable as the production main path.
- Main benefits: low failure rate in batch generation, with good balance between morphological diversity and geometric continuity.

#### Physics path: explicit physical-field evolution
- Uses temperature diffusion, seepage-driven migration, temperature-threshold precipitation, reactive expansion, and optional shear deformation to evolve the orebody by time steps.
- Why this is done: key physical processes are made explicit as interpretable state variables, enabling analysis of how parameter changes affect orebody outcomes.
- Difference from other paths: physics is the most process-simulation-oriented route, computationally heavier but strongest in interpretability.
- Main benefits: exports time-step snapshots (`step_*.npz` + manifest), suitable for temporal QC, process visualization, and parameter sensitivity studies.

#### Hybrid path: mechanism fusion and steady-shape refinement
- In `main.py`, mineralization potentials from staged and legacy are normalized and fused, followed by smoothing, boundary attenuation, and threshold sparsification.
- Why this is done: a single path often becomes either overly regular or overly noisy; fusion compensates for weaknesses.
- Difference from other paths: hybrid does not introduce new physical equations; instead, it emphasizes statistical robustness and engineering usability of output morphology.
- Main benefits: improves orebody continuity, reduces boundary artifacts, and is more GAN-training-friendly downstream.

### 2) Generative Learning Layer
#### prepare-data: data unification and distribution alignment
- Recursively scans samples in outputs, automatically identifies voxel key names, reshapes to `[N, 32, 32, 32]`, and normalizes to `[-1, 1]`.
- Why this is done: GANs are sensitive to input distributions; scale alignment significantly improves training stability.
- Additional benefit: generates a manifest preserving sample-source mapping for auditability and reproducibility.

#### train: 3D conditional WGAN-GP training
- The generator uses 3D transposed convolutions to upsample progressively to `32^3`; the critic broadcasts condition vectors as voxel condition maps and jointly discriminates conditions with voxels.
- Why WGAN-GP (`n_critic + gradient penalty`) instead of vanilla GAN: it is more stable in 3D voxel scenarios and suffers less mode collapse.
- Supports resume training; checkpoints record network parameters, optimizer states, hyperparameters, and scaling info.
- Main benefits: maintains training stability while supporting both conditional control and unconditional modeling tasks.

#### generate: controlled sampling and multi-format export
- Supports three condition sources: single vector, condition file, and random condition; outputs include normalized tensor and de-normalized grade tensor.
- Optionally exports marching-cubes meshes in OBJ/PLY, forming dual artifacts: “voxel data + visualization meshes.”
- Why this is done: voxels support numerical analysis, while meshes support geometric presentation and interoperability with 3D software.
- Main benefits: one generated result can simultaneously serve modeling, evaluation, and visualization needs.

### 3) Orchestration Layer
- `main.py` uses a subcommand structure (`oregen` / `gan-train` / `gan-generate`) to unify parameter entry, path conventions, and output directories.
- Supports in-code `use_code_mode` + `code_mode_args` for fixed script-based experiment reproduction, reducing command-line errors.
- By default, `oregen` exports `ore_body_###.ply` and `ore_surface_###.{ply,obj}`, and also synchronously exports `ore_tensor_###.npz` (readable directly by GAN data-building).
- Outputs are consistently written to `outputs/oregen` and `outputs/gan`, enabling automatic chaining of data-building, training, and generation stages.
- Main benefits: reduces experiment-management complexity and improves reproducibility and handoff readiness.

## Per-File Source Code Breakdown (10/10)
### main.py
- Module role: central scheduler for the whole project, responsible for CLI subcommands, algorithm dispatch, export strategy, and batch execution.
- Key capabilities:
	1. Unified parameter entry covering `oregen` / `gan-train` / `gan-generate`.
	2. Unified export chain (voxel PLY, surface PLY, surface OBJ, stage logs).
	3. Unified experiment-reproduction switch (`use_code_mode`).
- Why this design: avoids parameter drift and path inconsistency caused by multiple entry scripts.
- Direct benefits: workflows become standardized and reproducible, and new algorithm branches can be added easily.

### gan_wgangp.py
- Module role: standalone implementation and utility set for 3D conditional WGAN-GP.
- Key components:
	1. `VoxelTensorDataset`: data validation, shape normalization, optional auto-normalization.
	2. `Generator3D` / `Critic3D`: `32^3` voxel generation and conditional discrimination.
	3. `gradient_penalty`: core stabilization component for WGAN-GP training.
	4. `prepare_data` / `train` / `generate`: forms a closed-loop workflow.
- Why this design: keeps “data engineering + model training + result export” in one semantic domain, reducing cross-script coupling.
- Direct benefits: controllable training, directly consumable outputs, and easy hyperparameter comparisons.

### ore_state.py
- Module role: shared data-structure layer across algorithms.
- Key content: unified channels and indexing tools for `potential`, `temperature`, `pressure`, `permeability`, `fluid_flux`, `reactivity`, etc.
- Why this design: different algorithms need shared voxel semantics; otherwise, algorithm switching and result fusion are difficult.
- Direct benefits: lower adaptation cost between modules and better interoperability among staged/workflow/physics.

### seed_generator.py
- Module role: geometric prior constructor for the legacy path.
- Key mechanisms: directional sampling, path propagation, local thickening (spherical stamp), and multi-branch vein expansion.
- Why this design: orebodies typically exhibit a “trunk + branches + local enrichment” structure; geometric priors significantly reduce invalid search space.
- Direct benefits: outputs better match geological intuition, and incubation rules converge more easily to plausible morphologies.

### incubation_engine.py
- Module role: rule execution scheduler and logger.
- Key mechanisms: reads rule titles, computes effective weights (influence factor and conformity), applies rules in sequence, and records stage deltas.
- Why this design: decouples “rule definition” from “rule scheduling,” making rule-set replacement and weight-strategy tuning easier.
- Direct benefits: higher maintainability of the rule system and easier rule-level ablation experiments.

### incubation_rules.py
- Module role: geological mechanism implementation library of 20 rules.
- Coverage: system controls, structural controls, fluid supply and transport, boiling precipitation, contact metasomatism, metamorphic remelting, and microstructure evolution.
- Why this design: discretizes mineralization knowledge into composable rule units for tailoring by deposit type.
- Direct benefits: reusable and extensible knowledge, with contribution traceability at rule granularity.

### staged_metallogenesis.py
- Module role: staged mineralization main model (strongest research interpretability).
- Stage responsibilities:
	1. `stage0`: background fields, structural weakening, boundary constraints, and style presets.
	2. `stage1`: source activation and metal complexation.
	3. `stage2`: Darcy transport and fracture-channel formation.
	4. `stage3`: boiling/mixing-triggered precipitation and metasomatism.
	5. `stage4`: zonation and alteration halos.
	6. `stage5`: supergene enrichment and ductile deformation.
- Why this design: follows geological narrative order and splits complex processes into verifiable stages.
- Direct benefits: helps explain the causal chain of “which mechanism causes which morphology” in papers.

### workflow_generator.py
- Module role: engineering-oriented high-throughput generation and mesh export.
- Key mechanisms: anisotropic decay fields, permeability-modulated noise, fault-displacement overlay, probabilistic micro-event superposition.
- Mesh capability: built-in isosurface extraction and Laplacian smoothing for direct high-quality OBJ/PLY output.
- Why this design: targets batch production and visualization delivery, emphasizing stability and export quality.
- Direct benefits: maintains morphology quality and export consistency as generation scale grows.

### physics_pipeline.py
- Module role: explicit physical-process simulator.
- Key mechanisms: boundary locking, central seed injection, directional permeability fields, temperature diffusion, seepage migration, threshold precipitation, reactive expansion, optional ductile shear deformation.
- Observability: supports time-step state snapshot export with manifest for downstream statistical analysis.
- Why this design: emphasizes interpretable mapping from “parameter -> process -> outcome.”
- Direct benefits: convenient for parameter sensitivity analysis, process visualization, and physical plausibility checks.

### snapshot_qc.py
- Module role: QC and visualization tool for physics time-series results.
- Key capabilities: slice-frame export, GIF animation, time-series statistics (CSV/JSON), and curve plotting.
- Why this design: physics results are temporal data; looking only at the final state loses critical process information.
- Direct benefits: quickly detects unstable periods, anomalous precipitation events, and parameter mismatch issues.

## Quick Reproducible Experiment
```bash
# 1) Generate prior orebodies
python main.py oregen --num-files 100 --algorithm workflow

# 2) Automatically build a training set from outputs/oregen and train GAN
python main.py gan-train --gan-epochs 300 --gan-batch-size 16

# 3) Generate new orebodies from checkpoint
python main.py gan-generate --gan-checkpoint outputs/gan/runs/checkpoints/wgangp_epoch_0300.pt --gan-num-samples 64
```

## Authenticity Notes
- The file inventory in this README comes from real-time scanning and does not rely on manual guesses.
- For batch-indexed files (e.g., `ore_body_###.ply`), per-file descriptions are provided in index+purpose form with one-to-one correspondence.
- Since output directories may keep growing, please re-scan and update this file after running generation commands again.

## Appendix A: Per-File List (Current Snapshot, with descriptions)
| No. | File Path | Category | Description |
|---:|---|---|---|
| 1 | __pycache__/incubation_engine.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `incubation_engine.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 2 | __pycache__/incubation_rules.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `incubation_rules.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 3 | __pycache__/main.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `main.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 4 | __pycache__/ore_state.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `ore_state.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 5 | __pycache__/physics_pipeline.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `physics_pipeline.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 6 | __pycache__/seed_generator.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `seed_generator.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 7 | __pycache__/snapshot_qc.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `snapshot_qc.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 8 | __pycache__/staged_metallogenesis.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `staged_metallogenesis.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 9 | __pycache__/voxel_workflow.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `voxel_workflow.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 10 | __pycache__/workflow_generator.cpython-39.pyc | Cached bytecode | Python cached bytecode (source module: `workflow_generator.py`); auto-generated at runtime, can be deleted and rebuilt by interpreter. |
| 11 | gan_wgangp.py | Source code | Full pipeline script for 3D conditional WGAN-GP; includes data building and normalization (`prepare-data`), training (`train`), and sampling + mesh export (`generate`). |
| 12 | incubation_engine.py | Source code | Rule incubation scheduler; reads rule titles, computes effective weights, invokes 20 incubation rules, and records stage logs. |
| 13 | incubation_rules.py | Source code | Implementation set of 20 mineralization rules; covers system control, fluid migration, precipitation triggers, structural control, and micro dynamics. |
| 14 | main.py | Source code | Unified main entry; subcommands include `oregen` / `gan-train` / `gan-generate`; handles argument parsing, algorithm dispatch, PLY/OBJ export, log aggregation, and batch loops. |
| 15 | ore_state.py | Source code | Core voxel-state container; defines multi-physics channels, indexing/neighborhood access, boundary clamping; shared data-interface layer for all algorithms. |
| 1873 | physics_pipeline.py | Source code | Physics-driven mineralization pipeline; based on temperature diffusion + seepage migration + threshold precipitation + reactive expansion + ductile shear; supports time-step snapshots and marching-cubes export. |
| 1874 | README.md | Other file | File purpose not separately defined in rules; recommended to interpret by name and same-directory context. |
| 1875 | seed_generator.py | Source code | Legacy prior seed generator; builds main ore veins, branch veins, and satellite veins, outputting an initial mineralization potential field for incubation. |
| 1876 | snapshot_qc.py | Source code | QC tool for physics snapshots; converts `step_*.npz` into slice frames/GIFs and exports time-series CSV/JSON and statistical curves. |
| 1877 | staged_metallogenesis.py | Source code | Multi-stage geological mineralization pipeline; implements stage0~stage5 for background initialization, source activation, transport/deposition, zonation/alteration, and supergene mineralization. |
| 1878 | workflow_generator.py | Source code | Seven-stage workflow generator; includes anisotropic decay, fault displacement, probabilistic micro-events, isosurface extraction, and smoothed mesh export. |
