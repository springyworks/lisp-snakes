use lisp_snakes_core::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// ── xterm.js bindings ─────────────────────────────────────────────────────────

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window"])]
    type Terminal;

    #[wasm_bindgen(constructor, js_namespace = ["window"], js_class = "Terminal")]
    fn new(opts: &JsValue) -> Terminal;

    #[wasm_bindgen(method)]
    fn open(this: &Terminal, el: &web_sys::Element);

    #[wasm_bindgen(method)]
    fn write(this: &Terminal, data: &str);

    #[wasm_bindgen(method)]
    fn clear(this: &Terminal);

    #[wasm_bindgen(method)]
    fn reset(this: &Terminal);

    #[wasm_bindgen(method, js_name = "onKey")]
    fn on_key(this: &Terminal, cb: &Closure<dyn FnMut(JsValue)>);

    #[wasm_bindgen(method, getter)]
    fn cols(this: &Terminal) -> u32;

    #[wasm_bindgen(method, getter)]
    fn rows(this: &Terminal) -> u32;
}

// ── inline CPU learner (24 weights — no GPU needed) ───────────────────────────

struct CpuLearner {
    weights: Vec<f32>,
    train_steps: u64,
    last_loss: f32,
}

impl CpuLearner {
    fn new() -> Self {
        Self {
            weights: vec![0.0; FEATURE_DIM * OUTPUT_DIM],
            train_steps: 0,
            last_loss: 0.0,
        }
    }

    fn train_step(&mut self, x: &[f32], y: &[f32], batch: usize, lr: f32) {
        // Manual matmul gradient descent for tiny linear model
        // pred = X @ W,  grad = X^T @ (pred - Y) / batch
        let mut loss_sum = 0.0f32;
        let mut grad = vec![0.0f32; FEATURE_DIM * OUTPUT_DIM];
        for b in 0..batch {
            for o in 0..OUTPUT_DIM {
                let mut pred = 0.0f32;
                for f in 0..FEATURE_DIM {
                    pred += x[b * FEATURE_DIM + f] * self.weights[f * OUTPUT_DIM + o];
                }
                let target = y[b * OUTPUT_DIM + o];
                let err = pred - target;
                loss_sum += err * err;
                for f in 0..FEATURE_DIM {
                    grad[f * OUTPUT_DIM + o] += x[b * FEATURE_DIM + f] * err;
                }
            }
        }
        let inv_batch = 1.0 / batch as f32;
        for i in 0..grad.len() {
            self.weights[i] -= lr * grad[i] * inv_batch;
        }
        self.last_loss = loss_sum * inv_batch / OUTPUT_DIM as f32;
        self.train_steps += 1;
    }

    fn save_to_storage(&self) {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };
        let storage = match window.local_storage() {
            Ok(Some(s)) => s,
            _ => return,
        };
        let encoded: String = self.weights.iter().map(|w| format!("{w:.6}")).collect::<Vec<_>>().join(",");
        let _ = storage.set_item("lisp_snakes_weights", &encoded);
        let _ = storage.set_item("lisp_snakes_steps", &self.train_steps.to_string());
    }

    fn load_from_storage(&mut self) {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };
        let storage = match window.local_storage() {
            Ok(Some(s)) => s,
            _ => return,
        };
        if let Ok(Some(data)) = storage.get_item("lisp_snakes_weights") {
            let parsed: Vec<f32> = data.split(',').filter_map(|s| s.parse().ok()).collect();
            if parsed.len() == FEATURE_DIM * OUTPUT_DIM {
                self.weights = parsed;
            }
        }
        if let Ok(Some(steps)) = storage.get_item("lisp_snakes_steps") {
            self.train_steps = steps.parse().unwrap_or(0);
        }
    }
}

// ── App state ─────────────────────────────────────────────────────────────────

struct App {
    sim: Sim,
    learner: CpuLearner,
    frame: u64,
    paused: bool,
    show_help: bool,
    learning_enabled: bool,
    output_mode: u8,
    list_view: bool,
    tw: usize,
    th: usize,
    out: String,
}

// ── WASM entry point ──────────────────────────────────────────────────────────

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let container = document.get_element_by_id("terminal").unwrap();

    // Create xterm.js Terminal
    let opts = js_sys::Object::new();
    js_sys::Reflect::set(&opts, &"cols".into(), &JsValue::from(120)).unwrap();
    js_sys::Reflect::set(&opts, &"rows".into(), &JsValue::from(40)).unwrap();
    js_sys::Reflect::set(&opts, &"cursorBlink".into(), &JsValue::FALSE).unwrap();
    js_sys::Reflect::set(&opts, &"cursorStyle".into(), &"bar".into()).unwrap();
    js_sys::Reflect::set(&opts, &"disableStdin".into(), &JsValue::TRUE).unwrap();
    js_sys::Reflect::set(&opts, &"convertEol".into(), &JsValue::TRUE).unwrap();
    let theme = js_sys::Object::new();
    js_sys::Reflect::set(&theme, &"background".into(), &"#0a0a0a".into()).unwrap();
    js_sys::Reflect::set(&theme, &"foreground".into(), &"#cccccc".into()).unwrap();
    js_sys::Reflect::set(&opts, &"theme".into(), &theme).unwrap();

    let term = Terminal::new(&opts);
    term.open(&container);

    let tw = term.cols() as usize;
    let th = term.rows() as usize;

    let mut learner = CpuLearner::new();
    learner.load_from_storage();

    let app = App {
        sim: Sim::new(tw, th),
        learner,
        frame: 0,
        paused: false,
        show_help: true,
        learning_enabled: true,
        output_mode: 0,
        list_view: false,
        tw,
        th,
        out: String::with_capacity((tw + 32) * th),
    };

    // Wrap in Rc<RefCell> for shared ownership in closures
    let app = std::rc::Rc::new(std::cell::RefCell::new(app));

    // ── keyboard handler ──────────────────────────────────────────────────
    let app_key = app.clone();
    let key_cb = Closure::<dyn FnMut(JsValue)>::new(move |ev: JsValue| {
        let key = js_sys::Reflect::get(&ev, &"key".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let mut a = app_key.borrow_mut();
        match key.as_str() {
            " " => a.paused = !a.paused,
            "+" | "=" => {} // fps not applicable in rAF
            "-" => {}
            "s" => {
                let play_h = a.th.saturating_sub(2).max(1);
                let color = a.sim.next_color;
                a.sim.next_color += 1;
                let tw = a.tw;
                let frame = a.frame;
                let snake = Sim::make_snake(tw, play_h, color, frame, &mut a.sim.rng);
                a.sim.snakes.push(snake);
            }
            "k" => {
                if let Some(s) = a.sim.snakes.iter_mut().find(|s| s.alive) {
                    s.alive = false;
                }
            }
            "c" => {
                a.sim.trail.fill(0.0);
                a.sim.learn_age.fill(1024);
            }
            "r" => {
                a.sim = Sim::new(a.tw, a.th);
                a.frame = 0;
                a.learner = CpuLearner::new();
            }
            "l" => a.learning_enabled = !a.learning_enabled,
            "d" => a.list_view = !a.list_view,
            "v" => a.output_mode = (a.output_mode + 1) % NUM_OUTPUT_MODES,
            "p" => a.learner.save_to_storage(),
            "?" | "h" => a.show_help = !a.show_help,
            _ => {}
        }
    });
    term.on_key(&key_cb);
    key_cb.forget();

    // ── animation loop ────────────────────────────────────────────────────
    let term = std::rc::Rc::new(term);
    let app_loop = app.clone();
    let term_loop = term.clone();

    let frame_cb: std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    let frame_cb2 = frame_cb.clone();

    let mut tick_count: u32 = 0;

    *frame_cb2.borrow_mut() = Some(Closure::new(move || {
        tick_count += 1;
        // Run at ~8 fps (skip frames from 60fps rAF)
        if tick_count % 8 != 0 {
            request_animation_frame(frame_cb.borrow().as_ref().unwrap());
            return;
        }

        let mut a = app_loop.borrow_mut();

        if !a.paused {
            // Train inline (multiple steps, tiny model)
            if a.learning_enabled {
                let batch = 64;
                let (x, y, _marks) = a.sim.sample_batch(batch);
                let actual_batch = x.len() / FEATURE_DIM;
                if actual_batch > 0 {
                    let lr = 0.015 * (1.0 / (1.0 + a.learner.train_steps as f32 * 0.00003));
                    for _ in 0..5 {
                        a.learner.train_step(&x, &y, actual_batch, lr);
                    }
                }
            }

            let weights = a.learner.weights.clone();
            let cur_frame = a.frame;
            a.sim.step(&weights, cur_frame);
            a.frame += 1;

            // Auto-save checkpoint every 500 frames
            if a.frame % 500 == 0 {
                a.learner.save_to_storage();
            }
        }

        // Render
        let alive = a.sim.alive_count();
        let total = a.sim.snakes.len();
        let pause_tag = if a.paused { " \u{23f8}" } else { "" };
        let learn_tag = if a.learning_enabled {
            format!(" t:{}", a.learner.train_steps)
        } else {
            " \u{00b7}learn:off".to_string()
        };
        let out_tag = match a.output_mode {
            0 => "src", 1 => "out", 2 => "comments", 3 => "result", 4 => "stats", _ => "dna",
        };
        let view_tag = if a.list_view { "list" } else { "2D" };
        let hud = format!(
            " {alive}/{total} \u{00b7} f:{} \u{00b7} loss:{:.4} \u{00b7} meet:{} \u{00b7} {view_tag}:{out_tag}{pause_tag}{learn_tag} \u{00b7} [?]",
            a.frame, a.learner.last_loss, a.sim.total_meetings,
        );

        let tw = a.tw;
        let th = a.th;
        let output_mode = a.output_mode;
        let list_view = a.list_view;
        let show_help = a.show_help;
        let paused = a.paused;
        let learning_enabled = a.learning_enabled;
        let frame = a.frame;
        let total_meetings = a.sim.total_meetings;
        let last_loss = a.learner.last_loss;

        let mut out = std::mem::take(&mut a.out);
        if list_view {
            a.sim.render_list_view(&mut out, &hud, tw, th, output_mode);
        } else {
            a.sim.render(&mut out, &hud, frame, output_mode);
        }

        if show_help {
            let weights = a.learner.weights.clone();
            lisp_snakes_core::sim::render_help_overlay(
                &mut out, tw, th, paused, 8, output_mode, list_view, learning_enabled,
                alive, total, frame, total_meetings, last_loss, &weights,
            );
        }

        // Write to xterm — cursor home + content
        let out_ref: &str = &out;
        // Move cursor to top-left first
        term_loop.write("\x1b[H");
        term_loop.write(out_ref);
        a.out = out;

        drop(a);
        request_animation_frame(frame_cb.borrow().as_ref().unwrap());
    }));

    request_animation_frame(frame_cb2.borrow().as_ref().unwrap());
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}
