# RLfOLD NoCrash 0.9.15 Seed-0 Pilot

This evidence bundle publishes the selected demonstration-assisted Pixel SAC
checkpoint from the first CarlaRLLab vision pilot.

## Selected Policy

- Algorithm: SAC with 5,000 BC pretraining updates and `0.5 * L_BC` retained
  in the online actor objective.
- Online budget: 20,000 CARLA steps; selected checkpoint: step 8,000.
- Training: Town01, fixed 20 vehicles / 50 walkers, seed 0.
- Input: one front RGB camera, two `84x84` frames, route waypoints, speed, and
  previous steering.
- Checkpoint SHA-256:
  `9ff30f291781814d33f6ee56005eb78d878de9340065c78ca41a4f3124349c2a`.
- Training source commit: `5ffa947808071ed87194a6480cec6c4c3dd66171`.

## Test Results

The frozen checkpoint scored 46% success on 50 Town02 Empty episodes, 24% on
50 Regular episodes, and 4% on 50 Dense episodes. All per-episode JSON reports
are tracked in the repository evidence bundle.

## Limitations

This is a 20k-step, single-seed pilot trained with a reduced fixed traffic
curriculum. It is not the project's three-seed baseline and does not claim
protocol-equivalent reproduction of the original RLfOLD paper. Later training
checkpoints regressed, and the growing late-stage critic losses are visible in
the published curves.

The 245 MB demonstration dataset is not in Git history. Its reproducible
collection command, metadata, shape, and SHA-256 are recorded in the evidence
bundle.
