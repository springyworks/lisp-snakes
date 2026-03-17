pub mod lisp;
pub mod sim;

pub use sim::{Dir, LispSnake, Sim, snake_color, smoothstep, color_from_density_age};
pub use sim::{FEATURE_DIM, OUTPUT_DIM, INITIAL_SNAKES, MAX_SNAKES, MEET_DIST};
pub use sim::{SPAWN_COOLDOWN, MAX_BODY_LEN, LISP_BUDGET_PER_TICK, BOLD_SPAWN_FRAMES};
pub use sim::NUM_OUTPUT_MODES;
