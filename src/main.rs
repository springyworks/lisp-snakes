mod lisp;

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
use rand::Rng;
use std::f32::consts::TAU;
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

type B = Wgpu<f32, i32, u32>;

const FEATURE_DIM: usize = 8;
const OUTPUT_DIM: usize = 3; // dx, dy, mood
const INITIAL_SNAKES: usize = 3;
const MAX_SNAKES: usize = 12;
const MEET_DIST: i16 = 2;
const SPAWN_COOLDOWN: u32 = 120;
const MAX_BODY_LEN: usize = 30;
const LISP_BUDGET_PER_TICK: usize = 100;
const CHECKPOINT_PATH: &str = "lisp_snake_checkpoint.bin";
const CHECKPOINT_INTERVAL: u64 = 500;
const BOLD_SPAWN_FRAMES: u32 = 24;

// ── 4 cardinal directions ─────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    fn dx(self) -> i16 {
        match self {
            Dir::Left => -1,
            Dir::Right => 1,
            _ => 0,
        }
    }
    fn dy(self) -> i16 {
        match self {
            Dir::Up => -1,
            Dir::Down => 1,
            _ => 0,
        }
    }
    fn opposite(self) -> Dir {
        match self {
            Dir::Up => Dir::Down,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
            Dir::Right => Dir::Left,
        }
    }
    fn head_char(self) -> char {
        match self {
            Dir::Right => '›',
            Dir::Left => '‹',
            Dir::Down => 'ˬ',
            Dir::Up => 'ˆ',
        }
    }
    fn from_index(i: usize) -> Dir {
        match i % 4 {
            0 => Dir::Up,
            1 => Dir::Down,
            2 => Dir::Left,
            _ => Dir::Right,
        }
    }
}

// ── Lisp Snake ────────────────────────────────────────────────────────────────

struct LispSnake {
    alive: bool,
    x: i16,
    y: i16,
    dir: Dir,
    energy: f32,
    lisp_code: String,
    lisp_env: lisp::Env,
    lisp_result: f64,
    mood: f32,
    print_buf: Vec<String>,
    body: Vec<(i16, i16)>,
    color_id: usize,
    cooldown: u32,
    steps_to_turn: u16,
    age: u32,
    meetings: u32,
    spawn_frame: u64,
}

// ── Shared state for background learner ───────────────────────────────────────

struct SharedLearnState {
    weights: Vec<f32>,
    loss: f32,
    train_steps: u64,
    enabled: bool,
    /// Snapshot data for async training (features, targets, marks)
    snapshot_x: Vec<f32>,
    snapshot_y: Vec<f32>,
    snapshot_marks: Vec<usize>,
    snapshot_ready: bool,
}

// ── Simulation ────────────────────────────────────────────────────────────────

struct Sim {
    term_w: usize,
    term_h: usize,
    snakes: Vec<LispSnake>,
    trail: Vec<f32>,
    learn_age: Vec<u16>,
    rng: rand::rngs::ThreadRng,
    next_color: usize,
    total_meetings: u64,
}

// ── Online GPU learner ────────────────────────────────────────────────────────

struct OnlineLearner {
    device: WgpuDevice,
    w: Tensor<B, 2>,
    last_loss: f32,
    last_lr: f32,
}

impl OnlineLearner {
    fn new(device: WgpuDevice) -> Self {
        let w = Tensor::<B, 2>::random(
            [FEATURE_DIM, OUTPUT_DIM],
            Distribution::Normal(0.0, 0.12),
            &device,
        );
        Self {
            device,
            w,
            last_loss: 0.0,
            last_lr: 0.0,
        }
    }

    fn train_step(&mut self, x_cpu: &[f32], y_cpu: &[f32], batch: usize, lr: f32) {
        let x =
            Tensor::<B, 1>::from_floats(x_cpu, &self.device).reshape([batch, FEATURE_DIM]);
        let y =
            Tensor::<B, 1>::from_floats(y_cpu, &self.device).reshape([batch, OUTPUT_DIM]);

        let pred = x.clone().matmul(self.w.clone());
        let err = pred.sub(y);
        let grad = x.transpose().matmul(err.clone()).div_scalar(batch as f32);
        self.w = self.w.clone().sub(grad.mul_scalar(lr));

        self.last_loss = err.powf_scalar(2.0).mean().into_scalar();
        self.last_lr = lr;
    }

    fn weights_cpu(&self) -> Vec<f32> {
        self.w.to_data().to_vec::<f32>().expect("weights f32")
    }

    fn load_weights(&mut self, weights: &[f32]) {
        self.w = Tensor::<B, 1>::from_floats(weights, &self.device)
            .reshape([FEATURE_DIM, OUTPUT_DIM]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sim – Lisp snakes on a terminal grid
// ═══════════════════════════════════════════════════════════════════════════════

impl Sim {
    fn new(tw: usize, th: usize) -> Self {
        let mut rng = rand::rng();
        let play_h = th.saturating_sub(2);
        let mut snakes = Vec::with_capacity(INITIAL_SNAKES);
        for i in 0..INITIAL_SNAKES {
            snakes.push(Self::make_snake(tw, play_h, i, 0, &mut rng));
        }
        Self {
            term_w: tw,
            term_h: th,
            snakes,
            trail: vec![0.0; tw * th],
            learn_age: vec![1024; tw * th],
            rng,
            next_color: INITIAL_SNAKES,
            total_meetings: 0,
        }
    }

    fn make_snake(
        tw: usize,
        play_h: usize,
        color: usize,
        cur_frame: u64,
        rng: &mut rand::rngs::ThreadRng,
    ) -> LispSnake {
        let code = lisp::random_snippet(rng.random_range(0..1000_usize)).to_string();
        let x = rng.random_range(2..tw.saturating_sub(2).max(3) as i16);
        let y = rng.random_range(2..play_h.saturating_sub(2).max(3) as i16);
        let dir = Dir::from_index(rng.random_range(0..4_usize));
        LispSnake {
            alive: true,
            x,
            y,
            dir,
            energy: rng.random_range(0.7..1.0),
            body: vec![(x, y)],
            lisp_code: code,
            lisp_env: lisp::Env::new(),
            lisp_result: 0.0,
            mood: 0.5,
            print_buf: Vec::new(),
            color_id: color,
            cooldown: 0,
            steps_to_turn: rng.random_range(3..12),
            age: 0,
            meetings: 0,
            spawn_frame: cur_frame,
        }
    }

    fn resize(&mut self, tw: usize, th: usize) {
        self.term_w = tw;
        self.term_h = th;
        self.trail = vec![0.0; tw * th];
        self.learn_age = vec![1024; tw * th];
    }

    fn wrap_x(&self, v: i16) -> i16 {
        let w = self.term_w as i16;
        ((v % w) + w) % w
    }

    fn wrap_y(&self, v: i16, play_h: usize) -> i16 {
        let h = play_h as i16;
        ((v % h) + h) % h
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.term_w + x
    }

    fn pick_direction_and_mood(&mut self, wi: usize, weights: &[f32]) -> (Dir, f32) {
        let snake = &self.snakes[wi];
        let play_h = self.term_h.saturating_sub(2).max(1);

        let fx = snake.x as f32 / self.term_w.max(1) as f32;
        let fy = snake.y as f32 / play_h.max(1) as f32;
        let features = [
            (fx * TAU).sin(),
            (fy * TAU).cos(),
            snake.dir.dx() as f32,
            snake.dir.dy() as f32,
            self.trail_at(snake.x as usize, snake.y as usize),
            self.social_signal(wi),
            (snake.lisp_result as f32).clamp(-1.0, 1.0),
            snake.energy,
        ];

        let mut pred_x: f32 = 0.0;
        let mut pred_y: f32 = 0.0;
        let mut pred_mood: f32 = 0.0;
        for f in 0..FEATURE_DIM {
            pred_x += features[f] * weights[f * OUTPUT_DIM];
            pred_y += features[f] * weights[f * OUTPUT_DIM + 1];
            pred_mood += features[f] * weights[f * OUTPUT_DIM + 2];
        }
        // mood sigmoid: maps to 0..1
        let mood = 1.0 / (1.0 + (-pred_mood).exp());

        pred_x += self.rng.random_range(-0.3..0.3);
        pred_y += self.rng.random_range(-0.3..0.3);

        let cur = snake.dir;
        let candidate = if pred_x.abs() > pred_y.abs() {
            if pred_x > 0.0 {
                Dir::Right
            } else {
                Dir::Left
            }
        } else {
            if pred_y > 0.0 {
                Dir::Down
            } else {
                Dir::Up
            }
        };

        let dir = if candidate == cur.opposite() {
            match cur {
                Dir::Up | Dir::Down => {
                    if pred_x > 0.0 {
                        Dir::Right
                    } else {
                        Dir::Left
                    }
                }
                Dir::Left | Dir::Right => {
                    if pred_y > 0.0 {
                        Dir::Down
                    } else {
                        Dir::Up
                    }
                }
            }
        } else {
            candidate
        };
        (dir, mood)
    }

    fn trail_at(&self, x: usize, y: usize) -> f32 {
        if x < self.term_w && y < self.term_h {
            self.trail[self.idx(x, y)]
        } else {
            0.0
        }
    }

    fn social_signal(&self, wi: usize) -> f32 {
        let snake = &self.snakes[wi];
        let mut sig: f32 = 0.0;
        for (j, other) in self.snakes.iter().enumerate() {
            if j == wi || !other.alive {
                continue;
            }
            let dx = (other.x - snake.x) as f32;
            let dy = (other.y - snake.y) as f32;
            let d2 = dx * dx + dy * dy + 1.0;
            sig += (1.0 / d2).min(0.5);
        }
        sig
    }

    fn interaction_vector(&self, i: usize) -> (f32, f32, f32) {
        let wi = &self.snakes[i];
        let mut ax: f32 = 0.0;
        let mut ay: f32 = 0.0;
        let mut social: f32 = 0.0;
        for (j, wj) in self.snakes.iter().enumerate() {
            if i == j || !wj.alive {
                continue;
            }
            let dx = (wj.x - wi.x) as f32;
            let dy = (wj.y - wi.y) as f32;
            let d2 = dx * dx + dy * dy + 0.1;
            if d2 < 20.0 {
                ax -= dx / d2;
                ay -= dy / d2;
            } else if d2 < 400.0 {
                ax += dx / d2;
                ay += dy / d2;
            }
            social += (1.0 / d2).min(0.5);
        }
        (ax, ay, social)
    }

    fn sample_batch(&mut self, batch: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        let mut x = vec![0.0; batch * FEATURE_DIM];
        let mut y = vec![0.0; batch * OUTPUT_DIM];
        let mut marks = Vec::with_capacity(batch);
        let play_h = self.term_h.saturating_sub(2).max(1);

        for b in 0..batch {
            let mut idx = self.rng.random_range(0..self.snakes.len().max(1));
            for _ in 0..8 {
                if idx < self.snakes.len() && self.snakes[idx].alive {
                    break;
                }
                idx = self.rng.random_range(0..self.snakes.len().max(1));
            }
            if idx >= self.snakes.len() {
                continue;
            }
            let w = &self.snakes[idx];
            let (ax, ay, social) = self.interaction_vector(idx);
            let cell = self.idx(
                (w.x as usize).min(self.term_w.saturating_sub(1)),
                (w.y as usize).min(self.term_h.saturating_sub(1)),
            );
            let local = if cell < self.trail.len() {
                self.trail[cell]
            } else {
                0.0
            };

            let lisp_mod = w.lisp_result as f32;

            let row_x = b * FEATURE_DIM;
            x[row_x] = (w.x as f32 / self.term_w.max(1) as f32 * TAU).sin();
            x[row_x + 1] = (w.y as f32 / play_h.max(1) as f32 * TAU).cos();
            x[row_x + 2] = w.dir.dx() as f32;
            x[row_x + 3] = w.dir.dy() as f32;
            x[row_x + 4] = local;
            x[row_x + 5] = social;
            x[row_x + 6] = lisp_mod.clamp(-1.0, 1.0);
            x[row_x + 7] = w.energy;

            let row_y = b * OUTPUT_DIM;
            y[row_y] = (ax + 0.2 * (w.y as f32 * 0.07).sin() + 0.1 * lisp_mod).tanh();
            y[row_y + 1] = (ay + 0.2 * (w.x as f32 * 0.06).cos() + 0.1 * lisp_mod).tanh();
            // mood target: energy-weighted social signal modulated by lisp
            y[row_y + 2] = (w.energy * 0.6 + social * 0.3 + lisp_mod * 0.1).tanh();
            marks.push(cell);
        }
        (x, y, marks)
    }

    /// Advance all snakes one step – movement, Lisp ticks, meeting interactions
    fn step(&mut self, weights: &[f32], cur_frame: u64) {
        let play_h = self.term_h.saturating_sub(2).max(1);

        // Decay trail
        for t in &mut self.trail {
            *t *= 0.992;
        }
        for age in &mut self.learn_age {
            *age = age.saturating_add(1).min(1024);
        }

        let n = self.snakes.len();

        // ── Lisp tick: cooperative multitasking ───────────────────────────
        for i in 0..n {
            if !self.snakes[i].alive {
                continue;
            }
            self.snakes[i].age += 1;

            let fx = self.snakes[i].x as f64 / self.term_w.max(1) as f64;
            let fy = self.snakes[i].y as f64 / play_h.max(1) as f64;
            let code = self.snakes[i].lisp_code.clone();
            let energy = self.snakes[i].energy as f64;
            let age = self.snakes[i].age as f64;
            let mood = self.snakes[i].mood as f64;
            let result = lisp::tick(
                &code,
                &mut self.snakes[i].lisp_env,
                energy,
                fx,
                fy,
                age,
                mood,
                LISP_BUDGET_PER_TICK,
            );
            self.snakes[i].lisp_result = result.0;
            self.snakes[i].print_buf = result.1;
        }

        // ── Movement ─────────────────────────────────────────────────────
        for i in 0..n {
            if !self.snakes[i].alive {
                continue;
            }
            self.snakes[i].cooldown = self.snakes[i].cooldown.saturating_sub(1);

            if self.snakes[i].steps_to_turn == 0 {
                let (new_dir, mood) = self.pick_direction_and_mood(i, weights);
                self.snakes[i].dir = new_dir;
                self.snakes[i].mood = mood;
                self.snakes[i].steps_to_turn = self.rng.random_range(4..15);
            } else {
                self.snakes[i].steps_to_turn -= 1;
            }

            let dir = self.snakes[i].dir;
            let new_x = self.wrap_x(self.snakes[i].x + dir.dx());
            let new_y = self.wrap_y(self.snakes[i].y + dir.dy(), play_h);

            self.snakes[i].body.insert(0, (new_x, new_y));
            let max_len = self.snakes[i].lisp_code.len().min(MAX_BODY_LEN);
            while self.snakes[i].body.len() > max_len {
                self.snakes[i].body.pop();
            }

            self.snakes[i].x = new_x;
            self.snakes[i].y = new_y;

            let tx = (new_x as usize).min(self.term_w.saturating_sub(1));
            let ty = (new_y as usize).min(self.term_h.saturating_sub(1));
            let cell = self.idx(tx, ty);
            if cell < self.trail.len() {
                self.trail[cell] = (self.trail[cell] + 0.4).min(2.0);
            }

            self.snakes[i].energy -= 0.002;
            if self.snakes[i].energy < 0.05 {
                self.snakes[i].alive = false;
            }
        }

        // ── Meeting / interaction: when heads are within MEET_DIST ───────
        let alive_count = self.snakes.iter().filter(|s| s.alive).count();
        let mut spawn_queue: Vec<(i16, i16, String, usize)> = Vec::new();

        for i in 0..n {
            if !self.snakes[i].alive {
                continue;
            }
            for j in (i + 1)..n {
                if !self.snakes[j].alive {
                    continue;
                }
                let dx = (self.snakes[i].x - self.snakes[j].x).abs();
                let dy = (self.snakes[i].y - self.snakes[j].y).abs();
                if dx > MEET_DIST || dy > MEET_DIST {
                    continue;
                }
                let dist = (dx + dy) as f64;

                // Both snakes evaluate their Lisp code for interaction
                let result_i = lisp::eval_interaction(
                    &self.snakes[i].lisp_code,
                    self.snakes[i].energy as f64,
                    self.snakes[j].energy as f64,
                    dist,
                    self.snakes[i].age as f64,
                );
                let result_j = lisp::eval_interaction(
                    &self.snakes[j].lisp_code,
                    self.snakes[j].energy as f64,
                    self.snakes[i].energy as f64,
                    dist,
                    self.snakes[j].age as f64,
                );

                // Update persistent envs with meeting context
                let ej = self.snakes[j].energy as f64;
                let ei = self.snakes[i].energy as f64;
                self.snakes[i]
                    .lisp_env
                    .set("d", lisp::Val::Num(dist));
                self.snakes[i]
                    .lisp_env
                    .set("oe", lisp::Val::Num(ej));
                self.snakes[j]
                    .lisp_env
                    .set("d", lisp::Val::Num(dist));
                self.snakes[j]
                    .lisp_env
                    .set("oe", lisp::Val::Num(ei));

                // Energy transfer based on Lisp results
                let transfer = (result_i - result_j) as f32 * 0.02;
                self.snakes[i].energy = (self.snakes[i].energy + transfer).clamp(0.0, 1.5);
                self.snakes[j].energy = (self.snakes[j].energy - transfer).clamp(0.0, 1.5);

                self.snakes[i].meetings += 1;
                self.snakes[j].meetings += 1;
                self.total_meetings += 1;

                // Spawn child if both results positive and room available
                if result_i > 0.0
                    && result_j > 0.0
                    && alive_count + spawn_queue.len() < MAX_SNAKES
                    && self.snakes[i].cooldown == 0
                    && self.snakes[j].cooldown == 0
                {
                    let sx = (self.snakes[i].x + self.snakes[j].x) / 2;
                    let sy = (self.snakes[i].y + self.snakes[j].y) / 2;
                    let seed = self.snakes[i].age as usize
                        + self.snakes[j].age as usize
                        + self.total_meetings as usize;
                    let child_code = lisp::combine_snippets(
                        &self.snakes[i].lisp_code,
                        &self.snakes[j].lisp_code,
                        seed,
                    );
                    let color = self.next_color;
                    self.next_color += 1;
                    self.snakes[i].cooldown = SPAWN_COOLDOWN;
                    self.snakes[j].cooldown = SPAWN_COOLDOWN;
                    spawn_queue.push((sx, sy, child_code, color));
                }
            }
        }

        // Actually spawn children
        for (sx, sy, code, color) in spawn_queue {
            let dir = Dir::from_index(self.rng.random_range(0..4_usize));
            self.snakes.push(LispSnake {
                alive: true,
                x: sx,
                y: sy,
                dir,
                energy: 0.9,
                body: vec![(sx, sy)],
                lisp_code: code,
                lisp_env: lisp::Env::new(),
                lisp_result: 0.0,
                mood: 0.5,
                print_buf: Vec::new(),
                color_id: color,
                cooldown: SPAWN_COOLDOWN,
                steps_to_turn: self.rng.random_range(3..10),
                age: 0,
                meetings: 0,
                spawn_frame: cur_frame,
            });
        }

        // ── respawn if too few alive ──────────────────────────────────────
        let alive_count = self.snakes.iter().filter(|s| s.alive).count();
        if alive_count < 2 {
            let color = self.next_color;
            self.next_color += 1;
            self.snakes
                .push(Self::make_snake(self.term_w, play_h, color, cur_frame, &mut self.rng));
        }

        // Clean up dead snakes
        self.snakes.retain(|s| s.alive || !s.body.is_empty());
        for s in &mut self.snakes {
            if !s.alive && !s.body.is_empty() {
                s.body.pop();
            }
        }
    }

    fn mark_learning_cells(&mut self, marks: &[usize]) {
        for &cell in marks {
            if cell < self.learn_age.len() {
                self.learn_age[cell] = 0;
            }
        }
    }

    fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| s.alive).count()
    }

    fn render(&self, out: &mut String, hud: &str, cur_frame: u64, output_mode: u8) {
        let tw = self.term_w;
        let th = self.term_h;
        let play_h = th.saturating_sub(2);

        let cells = tw * play_h;
        // (char, color_id, is_head, is_bold)
        let mut overlay: Vec<Option<(char, usize, bool, bool)>> = vec![None; cells];

        for snake in &self.snakes {
            if snake.body.is_empty() {
                continue;
            }
            let is_new = snake.alive
                && cur_frame.saturating_sub(snake.spawn_frame) < BOLD_SPAWN_FRAMES as u64;

            // Output modes: 0=src 1=out 2=comments 3=result 4=stats 5=dna
            let display_text: String = match output_mode {
                1 if !snake.print_buf.is_empty() => {
                    snake.print_buf.iter().map(|s| {
                        let t: String = s.chars().take(MAX_BODY_LEN).collect();
                        t
                    }).collect::<Vec<_>>().join("\u{00b7}")
                }
                2 => {
                    let comments = lisp::extract_comments(&snake.lisp_code);
                    if comments.is_empty() {
                        snake.lisp_code.clone()
                    } else {
                        comments.join(" \u{00b7} ")
                    }
                }
                3 => {
                    format!("={:.2}", snake.lisp_result)
                }
                4 => {
                    format!("e:{:.1} a:{} m:{}",
                        snake.energy, snake.age, snake.meetings)
                }
                5 => {
                    // dna: full raw code including comments
                    snake.lisp_code.replace('\n', " ")
                }
                _ => snake.lisp_code.lines()
                    .filter(|l| !l.trim_start().starts_with(';'))
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string(),
            };
            let chars: Vec<char> = display_text.chars().take(MAX_BODY_LEN).collect();

            for (seg, &(bx, by)) in snake.body.iter().enumerate() {
                let ux = bx as usize;
                let uy = by as usize;
                if ux < tw && uy < play_h {
                    let oi = uy * tw + ux;
                    let ch = if seg == 0 {
                        snake.dir.head_char()
                    } else if seg < chars.len() {
                        chars[seg]
                    } else {
                        '\u{00b7}'
                    };
                    overlay[oi] = Some((ch, snake.color_id, seg == 0, is_new));
                }
            }
        }

        out.clear();
        out.push_str("\x1b[H");
        // HUD line — subtle dim white
        out.push_str("\x1b[2;37;40m");
        let hud_display: String = hud.chars().take(tw).collect();
        out.push_str(&hud_display);
        for _ in hud_display.len()..tw {
            out.push(' ');
        }
        out.push_str("\x1b[0m\n");

        // Separator — thin line
        out.push_str("\x1b[2;38;2;60;60;70m");
        for _ in 0..tw {
            out.push('\u{2500}');
        }
        out.push_str("\x1b[0m\n");

        // Play area
        for ty in 0..play_h {
            for tx in 0..tw {
                let oi = ty * tw + tx;
                if let Some((ch, color_id, is_head, is_bold)) = overlay[oi] {
                    let (r, g, b) = snake_color(color_id);
                    if is_head {
                        // Head: inverse subtle — fg dark on colored bg
                        if is_bold {
                            out.push_str("\x1b[1;");
                        } else {
                            out.push_str("\x1b[");
                        }
                        out.push_str(&format!(
                            "38;2;20;20;25;48;2;{r};{g};{b}m"
                        ));
                        out.push(ch);
                        out.push_str("\x1b[0m");
                    } else {
                        let fade = if !self
                            .snakes
                            .iter()
                            .any(|s| s.color_id == color_id && s.alive)
                        {
                            0.35
                        } else {
                            1.0
                        };
                        let fr = (r as f32 * fade) as u8;
                        let fg = (g as f32 * fade) as u8;
                        let fb = (b as f32 * fade) as u8;
                        if is_bold {
                            out.push_str("\x1b[1;");
                        } else {
                            out.push_str("\x1b[");
                        }
                        out.push_str(&format!("38;2;{fr};{fg};{fb}m"));
                        out.push(ch);
                    }
                } else {
                    let cell = self.idx(tx, ty);
                    let density =
                        if cell < self.trail.len() { self.trail[cell] } else { 0.0 };
                    let age = if cell < self.learn_age.len() {
                        self.learn_age[cell] as f32
                    } else {
                        1024.0
                    };

                    let norm = (density / 2.0).clamp(0.0, 1.0);
                    let aa = smoothstep(0.03, 0.85, norm);

                    // Subtle unicode trail chars
                    const TRAIL_CHARS: &[char] = &[' ', '\u{00b7}', '\u{2219}', '\u{2027}', '\u{00b0}', '\u{2022}', '\u{2218}', '\u{2236}'];
                    let gi = (aa * (TRAIL_CHARS.len() as f32 - 1.0)).round() as usize;
                    let ch = TRAIL_CHARS[gi.min(TRAIL_CHARS.len() - 1)];

                    if ch == ' ' {
                        out.push(' ');
                    } else {
                        let (r, g, b) = color_from_density_age(aa, age);
                        push_rgb_fg(out, r, g, b);
                        out.push(ch);
                    }
                }
            }
            out.push_str("\x1b[0m\r\n");
        }
    }
}

// ── color helpers ─────────────────────────────────────────────────────────────

fn push_rgb_fg(out: &mut String, r: u8, g: u8, b: u8) {
    out.push_str("\x1b[38;2;");
    out.push_str(&r.to_string());
    out.push(';');
    out.push_str(&g.to_string());
    out.push(';');
    out.push_str(&b.to_string());
    out.push('m');
}

fn snake_color(id: usize) -> (u8, u8, u8) {
    // Muted, desaturated palette — European minimalist
    const PALETTE: [(u8, u8, u8); 12] = [
        (180, 100, 90),   // terracotta
        (100, 160, 110),  // sage
        (100, 140, 170),  // slate blue
        (170, 150, 80),   // ochre
        (140, 110, 160),  // lavender
        (80, 160, 150),   // teal
        (170, 120, 80),   // sienna
        (130, 160, 90),   // olive
        (160, 100, 140),  // mauve
        (100, 150, 170),  // steel
        (160, 155, 100),  // wheat
        (120, 160, 130),  // celadon
    ];
    PALETTE[id % PALETTE.len()]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn color_from_density_age(density: f32, age: f32) -> (u8, u8, u8) {
    let t_recent = (1.0 - age / 320.0).clamp(0.0, 1.0);
    let t_old = (age / 1024.0).clamp(0.0, 1.0);
    // Cooler, more subdued trail colours
    let r = (25.0 + 60.0 * density + 30.0 * t_recent - 10.0 * t_old).clamp(0.0, 255.0) as u8;
    let g = (30.0 + 70.0 * density + 20.0 * t_recent + 10.0 * t_old).clamp(0.0, 255.0) as u8;
    let b = (35.0 + 50.0 * density + 15.0 * t_recent + 30.0 * t_old).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ── checkpoint save / load ────────────────────────────────────────────────────

fn save_checkpoint(frame: u64, weights: &[f32]) -> Result<()> {
    use std::io::Write as IoWrite;
    let mut f = std::fs::File::create(CHECKPOINT_PATH)?;
    f.write_all(b"LSNK0001")?;
    f.write_all(&frame.to_le_bytes())?;
    f.write_all(&(weights.len() as u32).to_le_bytes())?;
    for w in weights {
        f.write_all(&w.to_le_bytes())?;
    }
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
    for _ in 0..n {
        f.read_exact(&mut buf4)?;
        weights.push(f32::from_le_bytes(buf4));
    }
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

// ── help overlay ──────────────────────────────────────────────────────────────

fn render_help_overlay(
    out: &mut String,
    tw: usize,
    th: usize,
    paused: bool,
    fps: u64,
    output_mode: u8,
    list_view: bool,
    learning_enabled: bool,
    alive: usize,
    total: usize,
    frame: u64,
    meetings: u64,
    loss: f32,
) {
    // Box inner width = 44 chars (between │ borders)
    const W: usize = 44;
    let hr = format!("\u{250c}{}\u{2510}", "\u{2500}".repeat(W));
    let hr_mid = format!("\u{251c}{}\u{2524}", "\u{2500}".repeat(W));
    let hr_bot = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(W));

    let pad = |s: &str| -> String {
        let len = s.chars().count();
        if len >= W { s.chars().take(W).collect() }
        else { format!("{}{}", s, " ".repeat(W - len)) }
    };
    let row = |content: &str| -> String {
        format!("\u{2502}{}\u{2502}", pad(content))
    };

    let pause_st = if paused { "paused" } else { "running" };
    let out_st = match output_mode {
        0 => "src", 1 => "out", 2 => "comments",
        3 => "result", 4 => "stats", _ => "dna",
    };
    let view_st = if list_view { "list" } else { "2D" };
    let learn_st = if learning_enabled { "on" } else { "off" };

    let lines: Vec<String> = vec![
        hr.clone(),
        row("  lisp snakes \u{00b7} controls"),
        hr_mid.clone(),
        row(&format!("  space   pause / resume        [{pause_st}]")),
        row("  n       step one frame (paused)"),
        row(&format!("  +/-     speed                  [{fps} fps]")),
        row(&format!("  d       view toggle              [{view_st}]")),
        row(&format!("  v       output mode           [{out_st}]")),
        row("        src  out  comments"),
        row("        result  stats  dna"),
        row("  s       spawn new snake"),
        row("  k       kill oldest snake"),
        row("  c       clear trails"),
        row("  r       reset simulation"),
        row(&format!("  l       learning                  [{learn_st}]")),
        row("  p       save checkpoint"),
        row("  ?/h     this overlay"),
        row("  q/esc   quit"),
        hr_mid.clone(),
        row(&format!("  snakes {alive}/{total}  frame {frame}")),
        row(&format!("  meetings {meetings}  loss {loss:.4}")),
        row("  meeting (dist\u{2264}2) \u{2192} lisp eats lisp"),
        row("  ml learns continuously via burn."),
        hr_bot,
    ];

    let box_w = lines[0].chars().count();
    let box_h = lines.len();
    let ox = tw.saturating_sub(box_w) / 2;
    let oy = th.saturating_sub(box_h) / 2;

    for (i, line) in lines.iter().enumerate() {
        let y = oy + i;
        if y >= th {
            break;
        }
        out.push_str(&format!("\x1b[{};{}H", y + 1, ox + 1));
        out.push_str("\x1b[2;37;48;2;30;30;40m");
        out.push_str(line);
        out.push_str("\x1b[0m");
    }
}

// ── list view renderer ───────────────────────────────────────────────────────

fn render_list_view(sim: &Sim, out: &mut String, hud: &str, tw: usize, th: usize, output_mode: u8) {
    out.clear();
    out.push_str("\x1b[H");

    // HUD line
    out.push_str("\x1b[2;37;40m");
    let hud_display: String = hud.chars().take(tw).collect();
    out.push_str(&hud_display);
    for _ in hud_display.len()..tw {
        out.push(' ');
    }
    out.push_str("\x1b[0m\n");

    // Separator
    out.push_str("\x1b[2;38;2;60;60;70m");
    for _ in 0..tw {
        out.push('\u{2500}');
    }
    out.push_str("\x1b[0m\n");

    let avail_rows = th.saturating_sub(2);
    let mut row = 0;

    for snake in &sim.snakes {
        if row >= avail_rows {
            break;
        }

        let (r, g, b) = snake_color(snake.color_id);
        let status = if snake.alive { "\u{25cf}" } else { "\u{25cb}" };
        let mood_bar = mood_indicator(snake.mood);

        // Output-mode-dependent content
        let content: String = match output_mode {
            1 => {
                // out: println output
                if snake.print_buf.is_empty() {
                    "(no output)".to_string()
                } else {
                    snake.print_buf.join(" \u{00b7} ")
                }
            }
            2 => {
                // comments
                let comments = lisp::extract_comments(&snake.lisp_code);
                if comments.is_empty() { "(no comments)".to_string() }
                else { comments.join(" \u{00b7} ") }
            }
            3 => {
                // result
                format!("={:.4}", snake.lisp_result)
            }
            4 => {
                // stats
                format!("e:{:.2} mood:{:.2} age:{} meet:{} pos:({},{})",
                    snake.energy, snake.mood, snake.age, snake.meetings,
                    snake.x, snake.y)
            }
            5 => {
                // dna
                snake.lisp_code.replace('\n', " ")
            }
            _ => {
                // src (code without comments)
                snake.lisp_code.lines()
                    .filter(|l| !l.trim_start().starts_with(';'))
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            }
        };

        // Header line: status dot + snake id + mood
        let header = format!(" {status} #{:<2} {mood_bar} e:{:.2}", snake.color_id, snake.energy);
        let header_trunc: String = header.chars().take(tw).collect();

        // Dim if dead
        let (fr, fg, fb) = if snake.alive {
            (r, g, b)
        } else {
            ((r as f32 * 0.35) as u8, (g as f32 * 0.35) as u8, (b as f32 * 0.35) as u8)
        };

        out.push_str(&format!("\x1b[38;2;{fr};{fg};{fb}m"));
        out.push_str(&header_trunc);
        for _ in header_trunc.chars().count()..tw {
            out.push(' ');
        }
        out.push_str("\x1b[0m\r\n");
        row += 1;

        if row >= avail_rows {
            break;
        }

        // Content line(s) — wrap long content
        let content_prefix = "   ";
        let content_w = tw.saturating_sub(content_prefix.len());
        let content_chars: Vec<char> = content.chars().collect();
        let mut ci = 0;
        while ci < content_chars.len() && row < avail_rows {
            let end = (ci + content_w).min(content_chars.len());
            let chunk: String = content_chars[ci..end].iter().collect();
            out.push_str(&format!("\x1b[2;38;2;{fr};{fg};{fb}m"));
            out.push_str(content_prefix);
            out.push_str(&chunk);
            let line_len = content_prefix.len() + chunk.chars().count();
            for _ in line_len..tw {
                out.push(' ');
            }
            out.push_str("\x1b[0m\r\n");
            row += 1;
            ci = end;
        }
    }

    // Fill remaining rows with blanks
    while row < avail_rows {
        for _ in 0..tw {
            out.push(' ');
        }
        out.push_str("\r\n");
        row += 1;
    }
}

fn mood_indicator(mood: f32) -> &'static str {
    if mood > 0.8 { "\u{2593}\u{2593}\u{2593}" }      // ▓▓▓
    else if mood > 0.6 { "\u{2593}\u{2593}\u{2591}" }  // ▓▓░
    else if mood > 0.4 { "\u{2593}\u{2591}\u{2591}" }  // ▓░░
    else if mood > 0.2 { "\u{2591}\u{2591}\u{2591}" }  // ░░░
    else { "\u{2591}\u{2591} " }                        // ░░
}

// ═══════════════════════════════════════════════════════════════════════════════
//  main – Lisp snakes + GPU learning loop
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let initial_fps = parse_arg("fps", 8_u64).max(2);
    let batch_size = parse_arg("batch", 128_usize).clamp(16, 1024);

    let mut fps = initial_fps;
    let mut frame_time = Duration::from_micros(1_000_000 / fps);
    let mut paused = false;
    let mut step_one = false;
    let mut show_help = true;
    let mut learning_enabled = true;
    // output modes: 0=src 1=out 2=comments 3=result 4=stats 5=dna
    const NUM_OUTPUT_MODES: u8 = 6;
    let mut output_mode: u8 = 0;
    let mut list_view = false;

    let _guard = TerminalGuard::new()?;

    let (w, h) = term_size()?;
    let mut tw = w as usize;
    let mut th = h as usize;
    let mut sim = Sim::new(tw, th);

    // Try loading checkpoint – get initial weights
    let mut frame: u64 = 0;
    let initial_weights = if let Ok((saved_frame, weights)) = load_checkpoint() {
        if weights.len() == FEATURE_DIM * OUTPUT_DIM {
            frame = saved_frame;
            weights
        } else {
            vec![0.0; FEATURE_DIM * OUTPUT_DIM]
        }
    } else {
        vec![0.0; FEATURE_DIM * OUTPUT_DIM]
    };

    // ── shared learning state (main ←→ background thread) ─────────────
    let shared = Arc::new(Mutex::new(SharedLearnState {
        weights: initial_weights,
        loss: 0.0,
        train_steps: 0,
        enabled: true,
        snapshot_x: Vec::new(),
        snapshot_y: Vec::new(),
        snapshot_marks: Vec::new(),
        snapshot_ready: false,
    }));

    // ── background learning thread — runs at full GPU speed ───────────
    let learn_shared = Arc::clone(&shared);
    let _learn_thread = thread::spawn(move || {
        let device = WgpuDevice::default();
        let mut learner = OnlineLearner::new(device);
        // load initial weights from checkpoint
        {
            let st = learn_shared.lock().unwrap();
            if st.weights.len() == FEATURE_DIM * OUTPUT_DIM {
                learner.load_weights(&st.weights);
            }
        }

        loop {
            // Check if learning is enabled + grab snapshot
            let (enabled, snap_x, snap_y) = {
                let mut st = learn_shared.lock().unwrap();
                if !st.enabled {
                    drop(st);
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }
                if !st.snapshot_ready || st.snapshot_x.is_empty() {
                    drop(st);
                    thread::sleep(Duration::from_millis(5));
                    continue;
                }
                st.snapshot_ready = false;
                (true, st.snapshot_x.clone(), st.snapshot_y.clone())
            };

            if !enabled { continue; }

            let actual_batch = snap_x.len() / FEATURE_DIM;
            if actual_batch == 0 { continue; }

            // Run multiple train steps on the snapshot (GPU at full speed)
            for _ in 0..20 {
                let step_count = {
                    learn_shared.lock().unwrap().train_steps
                };
                let lr = 0.015 * (1.0 / (1.0 + step_count as f32 * 0.00003));
                learner.train_step(&snap_x, &snap_y, actual_batch, lr);

                // Publish weights + loss back
                let mut st = learn_shared.lock().unwrap();
                st.weights = learner.weights_cpu();
                st.loss = learner.last_loss;
                st.train_steps += 1;

                // If a new snapshot arrived, break and grab it
                if st.snapshot_ready { break; }
            }
        }
    });

    let mut out = String::with_capacity((tw + 32) * th);
    let start = Instant::now();
    let mut last_checkpoint_frame = frame;
    let mut cached_weights = vec![0.0f32; FEATURE_DIM * OUTPUT_DIM];
    let mut cached_loss: f32 = 0.0;
    let mut cached_train_steps: u64 = 0;

    loop {
        let frame_start = Instant::now();

        // ── handle input ──────────────────────────────────────────────────
        while poll(Duration::from_millis(0))? {
            match read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        // Save checkpoint on exit
                        let weights = shared.lock().unwrap().weights.clone();
                        let _ = save_checkpoint(frame, &weights);
                        return Ok(());
                    }
                    KeyCode::Char(' ') => paused = !paused,
                    KeyCode::Char('n') => {
                        if paused {
                            step_one = true;
                        }
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        fps = (fps + 2).min(60);
                        frame_time = Duration::from_micros(1_000_000 / fps);
                    }
                    KeyCode::Char('-') => {
                        fps = (fps.saturating_sub(2)).max(2);
                        frame_time = Duration::from_micros(1_000_000 / fps);
                    }
                    KeyCode::Char('s') => {
                        let play_h = th.saturating_sub(2).max(1);
                        let color = sim.next_color;
                        sim.next_color += 1;
                        sim.snakes
                            .push(Sim::make_snake(tw, play_h, color, frame, &mut sim.rng));
                    }
                    KeyCode::Char('k') => {
                        if let Some(s) = sim.snakes.iter_mut().find(|s| s.alive) {
                            s.alive = false;
                        }
                    }
                    KeyCode::Char('c') => {
                        sim.trail.fill(0.0);
                        sim.learn_age.fill(1024);
                    }
                    KeyCode::Char('r') => {
                        sim = Sim::new(tw, th);
                        frame = 0;
                        last_checkpoint_frame = 0;
                        let mut st = shared.lock().unwrap();
                        st.weights = vec![0.0; FEATURE_DIM * OUTPUT_DIM];
                        st.loss = 0.0;
                        st.train_steps = 0;
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
                    tw = nw as usize;
                    th = nh as usize;
                    sim.resize(tw, th);
                    let _ = execute!(stdout(), MoveTo(0, 0));
                }
                _ => {}
            }
        }

        // ── read latest weights from background learner ───────────────
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
            // ── feed snapshot to background learner ────────────────────
            if learning_enabled {
                let (x, y, marks) = sim.sample_batch(batch_size);
                let mut st = shared.lock().unwrap();
                st.snapshot_x = x;
                st.snapshot_y = y;
                st.snapshot_marks = marks;
                st.snapshot_ready = true;
            }

            // ── step snakes on grid ───────────────────────────────────
            sim.step(&cached_weights, frame);

            frame += 1;

            // ── auto-save checkpoint ──────────────────────────────────
            if frame - last_checkpoint_frame >= CHECKPOINT_INTERVAL {
                let _ = save_checkpoint(frame, &cached_weights);
                last_checkpoint_frame = frame;
            }
        }

        // ── render ────────────────────────────────────────────────────
        let _elapsed = start.elapsed().as_secs_f32();
        let alive = sim.alive_count();
        let total = sim.snakes.len();
        let pause_tag = if paused { " \u{23f8}" } else { "" };
        let learn_tag = if learning_enabled {
            format!(" t:{cached_train_steps}")
        } else {
            " \u{00b7}learn:off".to_string()
        };
        let out_tag = match output_mode {
            0 => "src", 1 => "out", 2 => "comments",
            3 => "result", 4 => "stats", _ => "dna",
        };
        let view_tag = if list_view { "list" } else { "2D" };
        let hud = format!(
            " {alive}/{total} \u{00b7} f:{frame} \u{00b7} loss:{:.4} \u{00b7} meet:{} \u{00b7} {view_tag}:{out_tag}{pause_tag}{learn_tag} \u{00b7} [?]",
            cached_loss,
            sim.total_meetings,
        );

        if list_view {
            render_list_view(&sim, &mut out, &hud, tw, th, output_mode);
        } else {
            sim.render(&mut out, &hud, frame, output_mode);
        }

        if show_help {
            render_help_overlay(
                &mut out, tw, th,
                paused, fps, output_mode, list_view, learning_enabled,
                alive, total, frame, sim.total_meetings, cached_loss,
            );
        }

        let mut term = stdout();
        queue!(term, MoveTo(0, 0))?;
        term.write_all(out.as_bytes())?;
        term.flush()?;

        let spent = frame_start.elapsed();
        if spent < frame_time {
            thread::sleep(frame_time - spent);
        }
    }
}
