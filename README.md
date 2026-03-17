# ascii-dynamcic-art-learning

Continuous online learning with Burn on WebGPU, rendered as high-frame-rate ANSI art in the terminal.

## What it does

- Simulates a population of stringworms moving on a high-resolution hidden canvas.
- Runs continuous online learning on GPU with Burn (WebGPU backend).
- Downsamples and anti-aliases the fast internal state to terminal resolution for smooth ASCII output.
- Colors each cell by both intensity and learning age, so recently learned regions glow differently than older regions.
- Maintains a fixed visual frame target (default 20 FPS) while allowing much faster learn/update cycles.

## Why WebGPU here

This app uses Burn with the WGPU backend, which is compatible with native GPU APIs and the WebGPU ecosystem.
The same learning design can be ported to a browser-facing renderer later while keeping the GPU tensor pipeline model.

## Run

From this project directory:

```bash
cargo run --release
```

Exit with `q` or `Esc`.

## Runtime tuning

```bash
cargo run --release -- --fps=20 --learn_steps=24 --batch=256
```

Parameters:

- `--fps`: terminal refresh rate target.
- `--learn_steps`: online learning iterations per frame.
- `--batch`: training batch size for each learning step.

## Notes

- Higher `learn_steps` and `batch` increase learning throughput but also increase GPU work.
- The renderer intentionally performs anti-aliased downsampling so fast internal dynamics remain visually stable.
- Color encoding uses time-since-learning to annotate temporal learning history.
