//! Platform-agnostic simulation: Dir, LispSnake, Sim, rendering to ANSI strings.
//! No threads, no filesystem, no GPU — those are provided by the platform crate.

use crate::lisp;
use rand::Rng;
use std::f32::consts::TAU;

pub const FEATURE_DIM: usize = 8;
pub const OUTPUT_DIM: usize = 3; // dx, dy, mood
pub const INITIAL_SNAKES: usize = 3;
pub const MAX_SNAKES: usize = 12;
pub const MEET_DIST: i16 = 2;
pub const SPAWN_COOLDOWN: u32 = 120;
pub const MAX_BODY_LEN: usize = 30;
pub const LISP_BUDGET_PER_TICK: usize = 100;
pub const BOLD_SPAWN_FRAMES: u32 = 24;
pub const NUM_OUTPUT_MODES: u8 = 6;

// ── 4 cardinal directions ─────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    pub fn dx(self) -> i16 {
        match self {
            Dir::Left => -1,
            Dir::Right => 1,
            _ => 0,
        }
    }
    pub fn dy(self) -> i16 {
        match self {
            Dir::Up => -1,
            Dir::Down => 1,
            _ => 0,
        }
    }
    pub fn opposite(self) -> Dir {
        match self {
            Dir::Up => Dir::Down,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
            Dir::Right => Dir::Left,
        }
    }
    pub fn head_char(self) -> char {
        match self {
            Dir::Right => '›',
            Dir::Left => '‹',
            Dir::Down => 'ˬ',
            Dir::Up => 'ˆ',
        }
    }
    pub fn from_index(i: usize) -> Dir {
        match i % 4 {
            0 => Dir::Up,
            1 => Dir::Down,
            2 => Dir::Left,
            _ => Dir::Right,
        }
    }
}

// ── Lisp Snake ────────────────────────────────────────────────────────────────

pub struct LispSnake {
    pub alive: bool,
    pub x: i16,
    pub y: i16,
    pub dir: Dir,
    pub energy: f32,
    pub lisp_code: String,
    pub lisp_env: lisp::Env,
    pub lisp_result: f64,
    pub mood: f32,
    pub print_buf: Vec<String>,
    pub body: Vec<(i16, i16)>,
    pub color_id: usize,
    pub cooldown: u32,
    pub steps_to_turn: u16,
    pub age: u32,
    pub meetings: u32,
    pub spawn_frame: u64,
}

// ── Simulation ────────────────────────────────────────────────────────────────

pub struct Sim {
    pub term_w: usize,
    pub term_h: usize,
    pub snakes: Vec<LispSnake>,
    pub trail: Vec<f32>,
    pub learn_age: Vec<u16>,
    pub rng: rand::rngs::ThreadRng,
    pub next_color: usize,
    pub total_meetings: u64,
}

impl Sim {
    pub fn new(tw: usize, th: usize) -> Self {
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

    pub fn make_snake(
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
        let mut env = lisp::Env::new();
        lisp::boot_kernel(&mut env);
        LispSnake {
            alive: true,
            x,
            y,
            dir,
            energy: rng.random_range(0.7..1.0),
            body: vec![(x, y)],
            lisp_code: code,
            lisp_env: env,
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

    pub fn resize(&mut self, tw: usize, th: usize) {
        self.term_w = tw;
        self.term_h = th;
        self.trail = vec![0.0; tw * th];
        self.learn_age = vec![1024; tw * th];
    }

    pub fn wrap_x(&self, v: i16) -> i16 {
        let w = self.term_w as i16;
        ((v % w) + w) % w
    }

    pub fn wrap_y(&self, v: i16, play_h: usize) -> i16 {
        let h = play_h as i16;
        ((v % h) + h) % h
    }

    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.term_w + x
    }

    pub fn pick_direction_and_mood(&mut self, wi: usize, weights: &[f32]) -> (Dir, f32) {
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
        let mood = 1.0 / (1.0 + (-pred_mood).exp());

        pred_x += self.rng.random_range(-0.3..0.3);
        pred_y += self.rng.random_range(-0.3..0.3);

        let cur = snake.dir;
        let candidate = if pred_x.abs() > pred_y.abs() {
            if pred_x > 0.0 { Dir::Right } else { Dir::Left }
        } else {
            if pred_y > 0.0 { Dir::Down } else { Dir::Up }
        };

        let dir = if candidate == cur.opposite() {
            match cur {
                Dir::Up | Dir::Down => {
                    if pred_x > 0.0 { Dir::Right } else { Dir::Left }
                }
                Dir::Left | Dir::Right => {
                    if pred_y > 0.0 { Dir::Down } else { Dir::Up }
                }
            }
        } else {
            candidate
        };
        (dir, mood)
    }

    pub fn trail_at(&self, x: usize, y: usize) -> f32 {
        if x < self.term_w && y < self.term_h {
            self.trail[self.idx(x, y)]
        } else {
            0.0
        }
    }

    pub fn social_signal(&self, wi: usize) -> f32 {
        let snake = &self.snakes[wi];
        let mut sig: f32 = 0.0;
        for (j, other) in self.snakes.iter().enumerate() {
            if j == wi || !other.alive { continue; }
            let dx = (other.x - snake.x) as f32;
            let dy = (other.y - snake.y) as f32;
            let d2 = dx * dx + dy * dy + 1.0;
            sig += (1.0 / d2).min(0.5);
        }
        sig
    }

    pub fn interaction_vector(&self, i: usize) -> (f32, f32, f32) {
        let wi = &self.snakes[i];
        let mut ax: f32 = 0.0;
        let mut ay: f32 = 0.0;
        let mut social: f32 = 0.0;
        for (j, wj) in self.snakes.iter().enumerate() {
            if i == j || !wj.alive { continue; }
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

    pub fn sample_batch(&mut self, batch: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        let mut x = vec![0.0; batch * FEATURE_DIM];
        let mut y = vec![0.0; batch * OUTPUT_DIM];
        let mut marks = Vec::with_capacity(batch);
        let play_h = self.term_h.saturating_sub(2).max(1);

        for b in 0..batch {
            let mut idx = self.rng.random_range(0..self.snakes.len().max(1));
            for _ in 0..8 {
                if idx < self.snakes.len() && self.snakes[idx].alive { break; }
                idx = self.rng.random_range(0..self.snakes.len().max(1));
            }
            if idx >= self.snakes.len() { continue; }
            let w = &self.snakes[idx];
            let (ax, ay, social) = self.interaction_vector(idx);
            let cell = self.idx(
                (w.x as usize).min(self.term_w.saturating_sub(1)),
                (w.y as usize).min(self.term_h.saturating_sub(1)),
            );
            let local = if cell < self.trail.len() { self.trail[cell] } else { 0.0 };

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
            y[row_y + 2] = (w.energy * 0.6 + social * 0.3 + lisp_mod * 0.1).tanh();
            marks.push(cell);
        }
        (x, y, marks)
    }

    pub fn step(&mut self, weights: &[f32], cur_frame: u64) {
        let play_h = self.term_h.saturating_sub(2).max(1);

        for t in &mut self.trail { *t *= 0.992; }
        for age in &mut self.learn_age { *age = age.saturating_add(1).min(1024); }

        let n = self.snakes.len();

        // Lisp tick
        for i in 0..n {
            if !self.snakes[i].alive { continue; }
            self.snakes[i].age += 1;
            let fx = self.snakes[i].x as f64 / self.term_w.max(1) as f64;
            let fy = self.snakes[i].y as f64 / play_h.max(1) as f64;
            let code = self.snakes[i].lisp_code.clone();
            let energy = self.snakes[i].energy as f64;
            let age = self.snakes[i].age as f64;
            let mood = self.snakes[i].mood as f64;
            let result = lisp::tick(
                &code, &mut self.snakes[i].lisp_env,
                energy, fx, fy, age, mood, LISP_BUDGET_PER_TICK,
            );
            self.snakes[i].lisp_result = result.0;
            self.snakes[i].print_buf = result.1;
        }

        // Movement
        for i in 0..n {
            if !self.snakes[i].alive { continue; }
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
            while self.snakes[i].body.len() > max_len { self.snakes[i].body.pop(); }

            self.snakes[i].x = new_x;
            self.snakes[i].y = new_y;

            let tx = (new_x as usize).min(self.term_w.saturating_sub(1));
            let ty = (new_y as usize).min(self.term_h.saturating_sub(1));
            let cell = self.idx(tx, ty);
            if cell < self.trail.len() {
                self.trail[cell] = (self.trail[cell] + 0.4).min(2.0);
            }

            self.snakes[i].energy -= 0.002;
            if self.snakes[i].energy < 0.05 { self.snakes[i].alive = false; }
        }

        // Meeting / interaction
        let alive_count = self.snakes.iter().filter(|s| s.alive).count();
        let mut spawn_queue: Vec<(i16, i16, String, usize)> = Vec::new();

        for i in 0..n {
            if !self.snakes[i].alive { continue; }
            for j in (i + 1)..n {
                if !self.snakes[j].alive { continue; }
                let dx = (self.snakes[i].x - self.snakes[j].x).abs();
                let dy = (self.snakes[i].y - self.snakes[j].y).abs();
                if dx > MEET_DIST || dy > MEET_DIST { continue; }
                let dist = (dx + dy) as f64;

                let result_i = lisp::eval_interaction(
                    &self.snakes[i].lisp_code, self.snakes[i].energy as f64,
                    self.snakes[j].energy as f64, dist, self.snakes[i].age as f64,
                );
                let result_j = lisp::eval_interaction(
                    &self.snakes[j].lisp_code, self.snakes[j].energy as f64,
                    self.snakes[i].energy as f64, dist, self.snakes[j].age as f64,
                );

                let ej = self.snakes[j].energy as f64;
                let ei = self.snakes[i].energy as f64;
                self.snakes[i].lisp_env.set("d", lisp::Val::Num(dist));
                self.snakes[i].lisp_env.set("oe", lisp::Val::Num(ej));
                self.snakes[j].lisp_env.set("d", lisp::Val::Num(dist));
                self.snakes[j].lisp_env.set("oe", lisp::Val::Num(ei));

                let transfer = (result_i - result_j) as f32 * 0.02;
                self.snakes[i].energy = (self.snakes[i].energy + transfer).clamp(0.0, 1.5);
                self.snakes[j].energy = (self.snakes[j].energy - transfer).clamp(0.0, 1.5);

                self.snakes[i].meetings += 1;
                self.snakes[j].meetings += 1;
                self.total_meetings += 1;

                if result_i > 0.0 && result_j > 0.0
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
                        &self.snakes[i].lisp_code, &self.snakes[j].lisp_code, seed,
                    );
                    let color = self.next_color;
                    self.next_color += 1;
                    self.snakes[i].cooldown = SPAWN_COOLDOWN;
                    self.snakes[j].cooldown = SPAWN_COOLDOWN;
                    spawn_queue.push((sx, sy, child_code, color));
                }
            }
        }

        for (sx, sy, code, color) in spawn_queue {
            let dir = Dir::from_index(self.rng.random_range(0..4_usize));
            let mut env = lisp::Env::new();
            lisp::boot_kernel(&mut env);
            self.snakes.push(LispSnake {
                alive: true, x: sx, y: sy, dir, energy: 0.9,
                body: vec![(sx, sy)], lisp_code: code,
                lisp_env: env, lisp_result: 0.0, mood: 0.5,
                print_buf: Vec::new(), color_id: color, cooldown: SPAWN_COOLDOWN,
                steps_to_turn: self.rng.random_range(3..10),
                age: 0, meetings: 0, spawn_frame: cur_frame,
            });
        }

        let alive_count = self.snakes.iter().filter(|s| s.alive).count();
        if alive_count < 2 {
            let color = self.next_color;
            self.next_color += 1;
            self.snakes.push(Self::make_snake(self.term_w, play_h, color, cur_frame, &mut self.rng));
        }

        self.snakes.retain(|s| s.alive || !s.body.is_empty());
        for s in &mut self.snakes {
            if !s.alive && !s.body.is_empty() { s.body.pop(); }
        }
    }

    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| s.alive).count()
    }

    pub fn render(&self, out: &mut String, hud: &str, cur_frame: u64, output_mode: u8) {
        let tw = self.term_w;
        let th = self.term_h;
        let play_h = th.saturating_sub(2);
        let cells = tw * play_h;
        let mut overlay: Vec<Option<(char, usize, bool, bool)>> = vec![None; cells];

        for snake in &self.snakes {
            if snake.body.is_empty() { continue; }
            let is_new = snake.alive
                && cur_frame.saturating_sub(snake.spawn_frame) < BOLD_SPAWN_FRAMES as u64;

            let display_text: String = match output_mode {
                1 if !snake.print_buf.is_empty() => {
                    snake.print_buf.iter().map(|s| {
                        let t: String = s.chars().take(MAX_BODY_LEN).collect(); t
                    }).collect::<Vec<_>>().join("\u{00b7}")
                }
                2 => {
                    let comments = lisp::extract_comments(&snake.lisp_code);
                    if comments.is_empty() { snake.lisp_code.clone() }
                    else { comments.join(" \u{00b7} ") }
                }
                3 => format!("={:.2}", snake.lisp_result),
                4 => format!("e:{:.1} a:{} m:{}", snake.energy, snake.age, snake.meetings),
                5 => snake.lisp_code.replace('\n', " "),
                _ => snake.lisp_code.lines()
                    .filter(|l| !l.trim_start().starts_with(';'))
                    .collect::<Vec<_>>().join(" ").trim().to_string(),
            };
            let chars: Vec<char> = display_text.chars().take(MAX_BODY_LEN).collect();

            for (seg, &(bx, by)) in snake.body.iter().enumerate() {
                let ux = bx as usize;
                let uy = by as usize;
                if ux < tw && uy < play_h {
                    let oi = uy * tw + ux;
                    let ch = if seg == 0 { snake.dir.head_char() }
                        else if seg < chars.len() { chars[seg] }
                        else { '\u{00b7}' };
                    overlay[oi] = Some((ch, snake.color_id, seg == 0, is_new));
                }
            }
        }

        out.clear();
        out.push_str("\x1b[H");
        out.push_str("\x1b[2;37;40m");
        let hud_display: String = hud.chars().take(tw).collect();
        out.push_str(&hud_display);
        for _ in hud_display.len()..tw { out.push(' '); }
        out.push_str("\x1b[0m\n");

        out.push_str("\x1b[2;38;2;60;60;70m");
        for _ in 0..tw { out.push('\u{2500}'); }
        out.push_str("\x1b[0m\n");

        for ty in 0..play_h {
            for tx in 0..tw {
                let oi = ty * tw + tx;
                if let Some((ch, color_id, is_head, is_bold)) = overlay[oi] {
                    let (r, g, b) = snake_color(color_id);
                    if is_head {
                        if is_bold { out.push_str("\x1b[1;"); }
                        else { out.push_str("\x1b["); }
                        out.push_str(&format!("38;2;20;20;25;48;2;{r};{g};{b}m"));
                        out.push(ch);
                        out.push_str("\x1b[0m");
                    } else {
                        let fade = if !self.snakes.iter().any(|s| s.color_id == color_id && s.alive)
                        { 0.35 } else { 1.0 };
                        let fr = (r as f32 * fade) as u8;
                        let fg = (g as f32 * fade) as u8;
                        let fb = (b as f32 * fade) as u8;
                        if is_bold { out.push_str("\x1b[1;"); }
                        else { out.push_str("\x1b["); }
                        out.push_str(&format!("38;2;{fr};{fg};{fb}m"));
                        out.push(ch);
                    }
                } else {
                    let cell = self.idx(tx, ty);
                    let density = if cell < self.trail.len() { self.trail[cell] } else { 0.0 };
                    let age = if cell < self.learn_age.len() { self.learn_age[cell] as f32 } else { 1024.0 };
                    let norm = (density / 2.0).clamp(0.0, 1.0);
                    let aa = smoothstep(0.03, 0.85, norm);
                    const TRAIL_CHARS: &[char] = &[' ', '\u{00b7}', '\u{2219}', '\u{2027}', '\u{00b0}', '\u{2022}', '\u{2218}', '\u{2236}'];
                    let gi = (aa * (TRAIL_CHARS.len() as f32 - 1.0)).round() as usize;
                    let ch = TRAIL_CHARS[gi.min(TRAIL_CHARS.len() - 1)];
                    if ch == ' ' { out.push(' '); }
                    else {
                        let (r, g, b) = color_from_density_age(aa, age);
                        push_rgb_fg(out, r, g, b);
                        out.push(ch);
                    }
                }
            }
            out.push_str("\x1b[0m\r\n");
        }
    }

    pub fn render_list_view(&self, out: &mut String, hud: &str, tw: usize, th: usize, output_mode: u8) {
        out.clear();
        out.push_str("\x1b[H");
        out.push_str("\x1b[2;37;40m");
        let hud_display: String = hud.chars().take(tw).collect();
        out.push_str(&hud_display);
        for _ in hud_display.len()..tw { out.push(' '); }
        out.push_str("\x1b[0m\n");
        out.push_str("\x1b[2;38;2;60;60;70m");
        for _ in 0..tw { out.push('\u{2500}'); }
        out.push_str("\x1b[0m\n");

        let avail_rows = th.saturating_sub(2);
        let mut row = 0;

        for snake in &self.snakes {
            if row >= avail_rows { break; }
            let (r, g, b) = snake_color(snake.color_id);
            let status = if snake.alive { "\u{25cf}" } else { "\u{25cb}" };
            let mood_bar = mood_indicator(snake.mood);
            let content: String = match output_mode {
                1 => if snake.print_buf.is_empty() { "(no output)".to_string() }
                     else { snake.print_buf.join(" \u{00b7} ") },
                2 => { let c = lisp::extract_comments(&snake.lisp_code);
                       if c.is_empty() { "(no comments)".to_string() } else { c.join(" \u{00b7} ") } },
                3 => format!("={:.4}", snake.lisp_result),
                4 => format!("e:{:.2} mood:{:.2} age:{} meet:{} pos:({},{})",
                    snake.energy, snake.mood, snake.age, snake.meetings, snake.x, snake.y),
                5 => snake.lisp_code.replace('\n', " "),
                _ => snake.lisp_code.lines().filter(|l| !l.trim_start().starts_with(';'))
                    .collect::<Vec<_>>().join(" ").trim().to_string(),
            };
            let header = format!(" {status} #{:<2} {mood_bar} e:{:.2}", snake.color_id, snake.energy);
            let header_trunc: String = header.chars().take(tw).collect();
            let (fr, fg, fb) = if snake.alive { (r, g, b) }
                else { ((r as f32 * 0.35) as u8, (g as f32 * 0.35) as u8, (b as f32 * 0.35) as u8) };

            out.push_str(&format!("\x1b[38;2;{fr};{fg};{fb}m"));
            out.push_str(&header_trunc);
            for _ in header_trunc.chars().count()..tw { out.push(' '); }
            out.push_str("\x1b[0m\r\n");
            row += 1;
            if row >= avail_rows { break; }

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
                for _ in line_len..tw { out.push(' '); }
                out.push_str("\x1b[0m\r\n");
                row += 1;
                ci = end;
            }
        }
        while row < avail_rows {
            for _ in 0..tw { out.push(' '); }
            out.push_str("\r\n");
            row += 1;
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

pub fn snake_color(id: usize) -> (u8, u8, u8) {
    const PALETTE: [(u8, u8, u8); 12] = [
        (180, 100, 90), (100, 160, 110), (100, 140, 170), (170, 150, 80),
        (140, 110, 160), (80, 160, 150), (170, 120, 80), (130, 160, 90),
        (160, 100, 140), (100, 150, 170), (160, 155, 100), (120, 160, 130),
    ];
    PALETTE[id % PALETTE.len()]
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn color_from_density_age(density: f32, age: f32) -> (u8, u8, u8) {
    let t_recent = (1.0 - age / 320.0).clamp(0.0, 1.0);
    let t_old = (age / 1024.0).clamp(0.0, 1.0);
    let r = (25.0 + 60.0 * density + 30.0 * t_recent - 10.0 * t_old).clamp(0.0, 255.0) as u8;
    let g = (30.0 + 70.0 * density + 20.0 * t_recent + 10.0 * t_old).clamp(0.0, 255.0) as u8;
    let b = (35.0 + 50.0 * density + 15.0 * t_recent + 30.0 * t_old).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

pub fn mood_indicator(mood: f32) -> &'static str {
    if mood > 0.8 { "\u{2593}\u{2593}\u{2593}" }
    else if mood > 0.6 { "\u{2593}\u{2593}\u{2591}" }
    else if mood > 0.4 { "\u{2593}\u{2591}\u{2591}" }
    else if mood > 0.2 { "\u{2591}\u{2591}\u{2591}" }
    else { "\u{2591}\u{2591} " }
}

pub fn render_help_overlay(
    out: &mut String, tw: usize, th: usize,
    paused: bool, fps: u64, output_mode: u8, list_view: bool,
    learning_enabled: bool, alive: usize, total: usize,
    frame: u64, meetings: u64, loss: f32, weights: &[f32],
) {
    const W: usize = 44;
    let hr = format!("\u{250c}{}\u{2510}", "\u{2500}".repeat(W));
    let hr_mid = format!("\u{251c}{}\u{2524}", "\u{2500}".repeat(W));
    let hr_bot = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(W));
    let pad = |s: &str| -> String {
        let len = s.chars().count();
        if len >= W { s.chars().take(W).collect() }
        else { format!("{}{}", s, " ".repeat(W - len)) }
    };
    let row = |content: &str| -> String { format!("\u{2502}{}\u{2502}", pad(content)) };

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
        hr_mid.clone(),
        row("  burn learned weights [8\u{00d7}3 linear]:"),
    ];

    let mut lines = lines;

    // Feature names and weight interpretation
    const FEAT: [&str; 8] = [
        "sin(x)", "cos(y)", "dir.dx", "dir.dy",
        "trail", "social", "lisp", "energy",
    ];
    const OUT: [&str; 3] = ["dx", "dy", "mood"];

    if weights.len() == FEATURE_DIM * OUTPUT_DIM {
        // Find strongest influences per output
        for o in 0..OUTPUT_DIM {
            let mut pairs: Vec<(f32, &str)> = (0..FEATURE_DIM)
                .map(|f| (weights[f * OUTPUT_DIM + o], FEAT[f]))
                .collect();
            pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
            // Show top 2 influences
            let top = &pairs[..2.min(pairs.len())];
            let desc: String = top.iter().map(|(w, name)| {
                let sign = if *w > 0.0 { "+" } else { "\u{2212}" };
                format!("{sign}{:.2} {name}", w.abs())
            }).collect::<Vec<_>>().join("  ");
            lines.push(row(&format!("  {}: {desc}", OUT[o])));
        }

        // Overall behavior summary
        let social_dx = weights[5 * OUTPUT_DIM];      // social → dx
        let social_dy = weights[5 * OUTPUT_DIM + 1];   // social → dy
        let energy_mood = weights[7 * OUTPUT_DIM + 2];  // energy → mood
        let trail_dx = weights[4 * OUTPUT_DIM];         // trail → dx
        let lisp_mood = weights[6 * OUTPUT_DIM + 2];    // lisp → mood

        let social_str = if social_dx.abs() + social_dy.abs() > 0.3 {
            if social_dx + social_dy > 0.0 { "seeks company" } else { "avoids crowds" }
        } else { "indifferent to others" };

        let energy_str = if energy_mood.abs() > 0.2 {
            if energy_mood > 0.0 { "energy \u{2192} happy" } else { "energy \u{2192} anxious" }
        } else { "energy \u{2260} mood" };

        let trail_str = if trail_dx.abs() > 0.2 {
            if trail_dx > 0.0 { "follows trails" } else { "avoids trails" }
        } else { "ignores trails" };

        let lisp_str = if lisp_mood.abs() > 0.15 {
            if lisp_mood > 0.0 { "lisp lifts mood" } else { "lisp darkens mood" }
        } else { "lisp \u{2260} mood" };

        lines.push(row(&format!("  \u{2192} {social_str}, {energy_str}")));
        lines.push(row(&format!("  \u{2192} {trail_str}, {lisp_str}")));
    } else {
        lines.push(row("  (no weights yet)"));
    }

    lines.push(hr_bot);

    let box_w = lines[0].chars().count();
    let box_h = lines.len();
    let ox = tw.saturating_sub(box_w) / 2;
    let oy = th.saturating_sub(box_h) / 2;

    for (i, line) in lines.iter().enumerate() {
        let y = oy + i;
        if y >= th { break; }
        out.push_str(&format!("\x1b[{};{}H", y + 1, ox + 1));
        out.push_str("\x1b[2;37;48;2;30;30;40m");
        out.push_str(line);
        out.push_str("\x1b[0m");
    }
}
