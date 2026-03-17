use anyhow::Result;
use burn::tensor::{Distribution, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, size as term_size, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use crossterm::{execute, queue};
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use lisp_snakes_core::*;

type B = Wgpu<f32, i32, u32>;

const CHECKPOINT_PATH: &str = "lisp_snake_checkpoint.bin";
const CHECKPOINT_INTERVAL: u64 = 500;
const BATCH_SIZE_DEFAULT: usize = 128;

// ── Online GPU learner ────────────────────────────────────────────────────────

struct OnlineLearner {
    device: WgpuDevice,
    w: Tensor<B, 2>,
    last_loss: f32,
}

impl OnlineLearner {
    fn new(device: WgpuDevice) -> Self {
        let w = Tensor::<B, 2>::random(
            [FEATURE_DIM, OUTPUT_DIM], Distribution::Normal(0.0, 0.12), &device,
        );
        Self { device, w, last_loss: 0.0 }
    }

    fn train_step(&mut self, x_cpu: &[f32], y_cpu: &[f32], batch: usize, lr: f32) {
        let x = Tensor::<B, 1>::from_floats(x_cpu, &self.device).reshape([batch, FEATURE_DIM]);
        let y = Tensor::<B, 1>::from_floats(y_cpu, &self.device).reshape([batch, OUTPUT_DIM]);
        let pred = x.clone().matmul(self.w.clone());
        let err = pred.sub(y);
        let grad = x.transpose().matmul(err.clone()).div_scalar(batch as f32);
        self.w = self.w.clone().sub(grad.mul_scalar(lr));
        self.last_loss = err.powf_scalar(2.0).mean().into_scalar();
    }

    fn weights_cpu(&self) -> Vec<f32> {
        self.w.to_data().to_vec::<f32>().expect("weights f32")
    }

    fn load_weights(&mut self, weights: &[f32]) {
        self.w = Tensor::<B, 1>::from_floats(weights, &self.device)
            .reshape([FEATURE_DIM, OUTPUT_DIM]);
    }
}

// ── Shared state for background learner ───────────────────────────────────────

struct SharedLearnState {
    weights: Vec<f32>,
    loss: f32,
    train_steps: u64,
    enabled: bool,
    snapshot_x: Vec<f32>,
    snapshot_y: Vec<f32>,
    snapshot_ready: bool,
}

// ── checkpoint save / load ────────────────────────────────────────────────────

fn save_checkpoint(frame: u64, weights: &[f32]) -> Result<()> {
    use std::io::Write as IoWrite;
    let mut f = std::fs::File::create(CHECKPOINT_PATH)?;
    f.write_all(b"LSNK0001")?;
    f.write_all(&frame.to_le_bytes())?;
    f.write_all(&(weights.len() as u32).to_le_bytes())?;
    for w in weights { f.write_all(&w.to_le_bytes())?; }
    Ok(())
}

fn load_checkpoint() -> Result<(u64, Vec<f32>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(CHECKPOINT_PATH)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"LSNK0001", "bad checkpoint magic");
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let frame = u64::from_le_bytes(buf8);
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    anyhow::ensure!(n == FEATURE_DIM * OUTPUT_DIM, "weight dimension mismatch");
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n { f.read_exact(&mut buf4)?; weights.push(f32::from_le_bytes(buf4)); }
    Ok((frame, weights))
}

// ── terminal guard ────────────────────────────────────────────────────────────

struct TerminalGuard;
impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        execute!(stdout(), EnterAlternateScreen, Hide, MoveTo(0, 0))?;
        Ok(Self)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(stdout(), Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let key = format!("--{name}=");
    std::env::args()
        .find_map(|a| a.strip_prefix(&key).and_then(|v| v.parse::<T>().ok()))
        .unwrap_or(default)
}

// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let batch_size = parse_arg("batch", BATCH_SIZE_DEFAULT).clamp(16, 1024);
    let mut fps = parse_arg("fps", 8_u64).max(2);
    let mut frame_time = Duration::from_micros(1_000_000 / fps);
    let mut paused = false;
    let mut step_one = false;
    let mut show_help = true;
    let mut learning_enabled = true;
    let mut output_mode: u8 = 0;
    let mut list_view = false;

    let _guard = TerminalGuard::new()?;
    let (w, h) = term_size()?;
    let mut tw = w as usize;
    let mut th = h as usize;
    let mut sim = Sim::new(tw, th);

    let mut frame: u64 = 0;
    let initial_weights = if let Ok((saved_frame, weights)) = load_checkpoint() {
        if weights.len() == FEATURE_DIM * OUTPUT_DIM { frame = saved_frame; weights }
        else { vec![0.0; FEATURE_DIM * OUTPUT_DIM] }
    } else { vec![0.0; FEATURE_DIM * OUTPUT_DIM] };

    let shared = Arc::new(Mutex::new(SharedLearnState {
        weights: initial_weights, loss: 0.0, train_steps: 0, enabled: true,
        snapshot_x: Vec::new(), snapshot_y: Vec::new(), snapshot_ready: false,
    }));

    let learn_shared = Arc::clone(&shared);
    let _learn_thread = thread::spawn(move || {
        let device = WgpuDevice::default();
        let mut learner = OnlineLearner::new(device);
        {
            let st = learn_shared.lock().unwrap();
            if st.weights.len() == FEATURE_DIM * OUTPUT_DIM { learner.load_weights(&st.weights); }
        }
        loop {
            let (enabled, snap_x, snap_y) = {
                let mut st = learn_shared.lock().unwrap();
                if !st.enabled { drop(st); thread::sleep(Duration::from_millis(50)); continue; }
                if !st.snapshot_ready || st.snapshot_x.is_empty() {
                    drop(st); thread::sleep(Duration::from_millis(5)); continue;
                }
                st.snapshot_ready = false;
                (true, st.snapshot_x.clone(), st.snapshot_y.clone())
            };
            if !enabled { continue; }
            let actual_batch = snap_x.len() / FEATURE_DIM;
            if actual_batch == 0 { continue; }
            for _ in 0..20 {
                let step_count = { learn_shared.lock().unwrap().train_steps };
                let lr = 0.015 * (1.0 / (1.0 + step_count as f32 * 0.00003));
                learner.train_step(&snap_x, &snap_y, actual_batch, lr);
                let mut st = learn_shared.lock().unwrap();
                st.weights = learner.weights_cpu();
                st.loss = learner.last_loss;
                st.train_steps += 1;
                if st.snapshot_ready { break; }
            }
        }
    });

    let mut out = String::with_capacity((tw + 32) * th);
    let mut last_checkpoint_frame = frame;
    let mut cached_weights = vec![0.0f32; FEATURE_DIM * OUTPUT_DIM];
    let mut cached_loss: f32 = 0.0;
    let mut cached_train_steps: u64 = 0;

    loop {
        let frame_start = Instant::now();

        while poll(Duration::from_millis(0))? {
            match read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        let weights = shared.lock().unwrap().weights.clone();
                        let _ = save_checkpoint(frame, &weights);
                        return Ok(());
                    }
                    KeyCode::Char(' ') => paused = !paused,
                    KeyCode::Char('n') => { if paused { step_one = true; } }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        fps = (fps + 2).min(60);
                        frame_time = Duration::from_micros(1_000_000 / fps);
                    }
                    KeyCode::Char('-') => {
                        fps = fps.saturating_sub(2).max(2);
                        frame_time = Duration::from_micros(1_000_000 / fps);
                    }
                    KeyCode::Char('s') => {
                        let play_h = th.saturating_sub(2).max(1);
                        let color = sim.next_color; sim.next_color += 1;
                        sim.snakes.push(Sim::make_snake(tw, play_h, color, frame, &mut sim.rng));
                    }
                    KeyCode::Char('k') => {
                        if let Some(s) = sim.snakes.iter_mut().find(|s| s.alive) { s.alive = false; }
                    }
                    KeyCode::Char('c') => { sim.trail.fill(0.0); sim.learn_age.fill(1024); }
                    KeyCode::Char('r') => {
                        sim = Sim::new(tw, th); frame = 0; last_checkpoint_frame = 0;
                        let mut st = shared.lock().unwrap();
                        st.weights = vec![0.0; FEATURE_DIM * OUTPUT_DIM];
                        st.loss = 0.0; st.train_steps = 0;
                    }
                    KeyCode::Char('l') => {
                        learning_enabled = !learning_enabled;
                        shared.lock().unwrap().enabled = learning_enabled;
                    }
                    KeyCode::Char('d') => list_view = !list_view,
                    KeyCode::Char('v') => output_mode = (output_mode + 1) % NUM_OUTPUT_MODES,
                    KeyCode::Char('p') => {
                        let weights = shared.lock().unwrap().weights.clone();
                        let _ = save_checkpoint(frame, &weights);
                    }
                    KeyCode::Char('?') | KeyCode::Char('h') => show_help = !show_help,
                    _ => {}
                },
                Event::Resize(nw, nh) => {
                    tw = nw as usize; th = nh as usize;
                    sim.resize(tw, th);
                    let _ = execute!(stdout(), MoveTo(0, 0));
                }
                _ => {}
            }
        }

        {
            let st = shared.lock().unwrap();
            if st.weights.len() == FEATURE_DIM * OUTPUT_DIM {
                cached_weights.copy_from_slice(&st.weights);
            }
            cached_loss = st.loss;
            cached_train_steps = st.train_steps;
        }

        let should_step = !paused || step_one;
        step_one = false;

        if should_step {
            if learning_enabled {
                let (x, y, _marks) = sim.sample_batch(batch_size);
                let mut st = shared.lock().unwrap();
                st.snapshot_x = x; st.snapshot_y = y; st.snapshot_ready = true;
            }
            sim.step(&cached_weights, frame);
            frame += 1;
            if frame - last_checkpoint_frame >= CHECKPOINT_INTERVAL {
                let _ = save_checkpoint(frame, &cached_weights);
                last_checkpoint_frame = frame;
            }
        }

        let alive = sim.alive_count();
        let total = sim.snakes.len();
        let pause_tag = if paused { " \u{23f8}" } else { "" };
        let learn_tag = if learning_enabled { format!(" t:{cached_train_steps}") }
            else { " \u{00b7}learn:off".to_string() };
        let out_tag = match output_mode {
            0 => "src", 1 => "out", 2 => "comments", 3 => "result", 4 => "stats", _ => "dna",
        };
        let view_tag = if list_view { "list" } else { "2D" };
        let hud = format!(
            " {alive}/{total} \u{00b7} f:{frame} \u{00b7} loss:{:.4} \u{00b7} meet:{} \u{00b7} {view_tag}:{out_tag}{pause_tag}{learn_tag} \u{00b7} [?]",
            cached_loss, sim.total_meetings,
        );

        if list_view { sim.render_list_view(&mut out, &hud, tw, th, output_mode); }
        else { sim.render(&mut out, &hud, frame, output_mode); }

        if show_help {
            lisp_snakes_core::sim::render_help_overlay(
                &mut out, tw, th, paused, fps, output_mode, list_view, learning_enabled,
                alive, total, frame, sim.total_meetings, cached_loss, &cached_weights,
            );
        }

        let mut term = stdout();
        queue!(term, MoveTo(0, 0))?;
        term.write_all(out.as_bytes())?;
        term.flush()?;

        let spent = frame_start.elapsed();
        if spent < frame_time { thread::sleep(frame_time - spent); }
    }
}
