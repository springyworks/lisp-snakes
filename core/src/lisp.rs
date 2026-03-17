//! Embedded multi-tasking Lisp interpreter for snake AI.
//!
//! Each snake carries a Lisp program that runs every frame with a step budget
//! (cooperative multitasking). When snakes meet, their programs evaluate with
//! each other's state to determine interaction outcomes.

use std::collections::HashMap;
use std::fmt;

// ── value type ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Num(f64),
    Sym(String),
    Bool(bool),
    List(Vec<Val>),
    Lambda(Vec<String>, Vec<Val>, HashMap<String, Val>),
    Nil,
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Num(n) => {
                if *n == (*n as i64 as f64) && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{:.3}", n)
                }
            }
            Val::Sym(s) => write!(f, "{s}"),
            Val::Bool(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Val::List(items) => {
                write!(f, "(")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Val::Lambda(p, ..) => write!(f, "(λ {})", p.join(" ")),
            Val::Nil => write!(f, "nil"),
        }
    }
}

impl Val {
    pub fn as_num(&self) -> f64 {
        match self {
            Val::Num(n) => *n,
            Val::Bool(b) => if *b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Val::Bool(false) | Val::Nil)
    }
}

// ── environment ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Env {
    pub data: HashMap<String, Val>,
}

impl Env {
    pub fn new() -> Self {
        let mut data = HashMap::new();
        data.insert("pi".into(), Val::Num(std::f64::consts::PI));
        data.insert("tau".into(), Val::Num(std::f64::consts::TAU));
        data.insert("nil".into(), Val::Nil);
        Env { data }
    }
    pub fn get(&self, k: &str) -> Option<&Val> {
        self.data.get(k)
    }
    pub fn set(&mut self, k: &str, v: Val) {
        self.data.insert(k.into(), v);
    }
}

// ── tokenizer ────────────────────────────────────────────────────────────────

fn tokenize(src: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut it = src.chars().peekable();
    while let Some(&ch) = it.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => { it.next(); }
            '(' | ')' | '\'' => { toks.push(ch.to_string()); it.next(); }
            ';' => { while it.next().is_some_and(|c| c != '\n') {} }
            '"' => {
                it.next();
                let mut s = String::from("\"");
                while let Some(&c) = it.peek() {
                    it.next();
                    if c == '"' { break; }
                    s.push(c);
                }
                s.push('"');
                toks.push(s);
            }
            _ => {
                let mut s = String::new();
                while let Some(&c) = it.peek() {
                    if " \t\n\r()".contains(c) { break; }
                    s.push(c);
                    it.next();
                }
                toks.push(s);
            }
        }
    }
    toks
}

// ── parser ───────────────────────────────────────────────────────────────────

fn parse_tok(toks: &[String], p: &mut usize) -> Result<Val, String> {
    if *p >= toks.len() {
        return Err("eof".into());
    }
    let t = &toks[*p];
    *p += 1;
    match t.as_str() {
        "(" => {
            let mut list = Vec::new();
            while *p < toks.len() && toks[*p] != ")" {
                list.push(parse_tok(toks, p)?);
            }
            if *p < toks.len() { *p += 1; }
            Ok(Val::List(list))
        }
        ")" => Err("unexpected )".into()),
        "'" => {
            let q = parse_tok(toks, p)?;
            Ok(Val::List(vec![Val::Sym("quote".into()), q]))
        }
        s if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 => {
            Ok(Val::Sym(s[1..s.len() - 1].to_string()))
        }
        "#t" | "true" => Ok(Val::Bool(true)),
        "#f" | "false" => Ok(Val::Bool(false)),
        "nil" => Ok(Val::Nil),
        s => s.parse::<f64>().map(Val::Num).or(Ok(Val::Sym(s.into()))),
    }
}

pub fn parse(src: &str) -> Result<Val, String> {
    let toks = tokenize(src);
    if toks.is_empty() {
        return Ok(Val::Nil);
    }
    let mut p = 0;
    parse_tok(&toks, &mut p)
}

// ── evaluator with step budget (cooperative multitasking) ────────────────────

/// Max captured print lines per tick
const MAX_PRINT_LINES: usize = 8;
const MAX_PRINT_LINE_LEN: usize = 40;

pub fn eval(v: &Val, env: &mut Env, budget: &mut usize, print_buf: &mut Vec<String>) -> Result<Val, String> {
    if *budget == 0 {
        return Ok(Val::Nil);
    }
    *budget -= 1;

    match v {
        Val::Num(_) | Val::Bool(_) | Val::Nil | Val::Lambda(..) => Ok(v.clone()),

        Val::Sym(name) => Ok(env
            .get(name)
            .cloned()
            .unwrap_or_else(|| Val::Sym(name.clone()))),

        Val::List(items) if items.is_empty() => Ok(Val::Nil),

        Val::List(items) => {
            // ── special forms ────────────────────────────────────────────
            if let Val::Sym(op) = &items[0] {
                match op.as_str() {
                    "quote" => return Ok(items.get(1).cloned().unwrap_or(Val::Nil)),

                    "if" => {
                        let cond = eval(items.get(1).unwrap_or(&Val::Nil), env, budget, print_buf)?;
                        return if cond.is_truthy() {
                            eval(items.get(2).unwrap_or(&Val::Nil), env, budget, print_buf)
                        } else {
                            eval(items.get(3).unwrap_or(&Val::Nil), env, budget, print_buf)
                        };
                    }

                    "define" => {
                        if let Some(Val::Sym(name)) = items.get(1) {
                            let val = eval(items.get(2).unwrap_or(&Val::Nil), env, budget, print_buf)?;
                            env.set(name, val.clone());
                            return Ok(val);
                        }
                        if let Some(Val::List(sig)) = items.get(1) {
                            if let Some(Val::Sym(name)) = sig.first() {
                                let params: Vec<String> = sig[1..]
                                    .iter()
                                    .filter_map(|v| {
                                        if let Val::Sym(s) = v { Some(s.clone()) } else { None }
                                    })
                                    .collect();
                                let body = items[2..].to_vec();
                                let lam = Val::Lambda(params, body, env.data.clone());
                                env.set(name, lam.clone());
                                return Ok(lam);
                            }
                        }
                        return Ok(Val::Nil);
                    }

                    "set!" => {
                        if let Some(Val::Sym(name)) = items.get(1) {
                            let val = eval(items.get(2).unwrap_or(&Val::Nil), env, budget, print_buf)?;
                            env.set(name, val.clone());
                            return Ok(val);
                        }
                        return Ok(Val::Nil);
                    }

                    "lambda" | "fn" => {
                        let params = match items.get(1) {
                            Some(Val::List(ps)) => ps
                                .iter()
                                .filter_map(|v| {
                                    if let Val::Sym(s) = v { Some(s.clone()) } else { None }
                                })
                                .collect(),
                            _ => vec![],
                        };
                        return Ok(Val::Lambda(
                            params,
                            items[2..].to_vec(),
                            env.data.clone(),
                        ));
                    }

                    "begin" | "do" => {
                        let mut result = Val::Nil;
                        for item in &items[1..] {
                            result = eval(item, env, budget, print_buf)?;
                        }
                        return Ok(result);
                    }

                    "let" => {
                        let mut local = env.clone();
                        if let Some(Val::List(bindings)) = items.get(1) {
                            for b in bindings {
                                if let Val::List(pair) = b {
                                    if let Some(Val::Sym(name)) = pair.first() {
                                        let val = eval(
                                            pair.get(1).unwrap_or(&Val::Nil),
                                            &mut local,
                                            budget,
                                            print_buf,
                                        )?;
                                        local.set(name, val);
                                    }
                                }
                            }
                        }
                        let mut result = Val::Nil;
                        for item in &items[2..] {
                            result = eval(item, &mut local, budget, print_buf)?;
                        }
                        return Ok(result);
                    }

                    "and" => {
                        let mut r = Val::Bool(true);
                        for item in &items[1..] {
                            r = eval(item, env, budget, print_buf)?;
                            if !r.is_truthy() {
                                return Ok(Val::Bool(false));
                            }
                        }
                        return Ok(r);
                    }

                    "or" => {
                        for item in &items[1..] {
                            let r = eval(item, env, budget, print_buf)?;
                            if r.is_truthy() {
                                return Ok(r);
                            }
                        }
                        return Ok(Val::Bool(false));
                    }

                    "cond" => {
                        for clause in &items[1..] {
                            if let Val::List(pair) = clause {
                                if pair.len() >= 2 {
                                    let test = eval(&pair[0], env, budget, print_buf)?;
                                    if test.is_truthy() {
                                        return eval(&pair[1], env, budget, print_buf);
                                    }
                                }
                            }
                        }
                        return Ok(Val::Nil);
                    }

                    _ => {} // fall through to function call
                }
            }

            // ── function call ────────────────────────────────────────────
            let func = eval(&items[0], env, budget, print_buf)?;
            let mut args = Vec::with_capacity(items.len() - 1);
            for item in &items[1..] {
                args.push(eval(item, env, budget, print_buf)?);
            }

            match func {
                Val::Lambda(params, body, closure) => {
                    let mut call_env = Env { data: closure };
                    for (i, p) in params.iter().enumerate() {
                        call_env.set(p, args.get(i).cloned().unwrap_or(Val::Nil));
                    }
                    let mut result = Val::Nil;
                    for expr in &body {
                        result = eval(expr, &mut call_env, budget, print_buf)?;
                    }
                    Ok(result)
                }
                Val::Sym(name) => builtin(&name, &args, print_buf),
                _ => Ok(Val::Nil),
            }
        }
    }
}

// ── built-in functions ───────────────────────────────────────────────────────

fn builtin(name: &str, args: &[Val], print_buf: &mut Vec<String>) -> Result<Val, String> {
    let a = |i: usize| args.get(i).map(|v| v.as_num()).unwrap_or(0.0);
    match name {
        "+" => Ok(Val::Num(args.iter().map(|v| v.as_num()).sum())),
        "-" => {
            if args.is_empty() { return Ok(Val::Num(0.0)); }
            if args.len() == 1 { return Ok(Val::Num(-a(0))); }
            Ok(Val::Num(args[1..].iter().fold(a(0), |acc, v| acc - v.as_num())))
        }
        "*" => Ok(Val::Num(args.iter().map(|v| v.as_num()).product())),
        "/" => {
            let d = a(1);
            Ok(Val::Num(if d == 0.0 { 0.0 } else { a(0) / d }))
        }
        "mod" | "%" => {
            let d = a(1);
            Ok(Val::Num(if d == 0.0 { 0.0 } else { a(0) % d }))
        }
        "abs" => Ok(Val::Num(a(0).abs())),
        "min" => Ok(Val::Num(
            args.iter().map(|v| v.as_num()).fold(f64::INFINITY, f64::min),
        )),
        "max" => Ok(Val::Num(
            args.iter()
                .map(|v| v.as_num())
                .fold(f64::NEG_INFINITY, f64::max),
        )),
        "sin" => Ok(Val::Num(a(0).sin())),
        "cos" => Ok(Val::Num(a(0).cos())),
        "sqrt" => Ok(Val::Num(a(0).max(0.0).sqrt())),
        "floor" => Ok(Val::Num(a(0).floor())),
        "ceil" => Ok(Val::Num(a(0).ceil())),
        "round" => Ok(Val::Num(a(0).round())),
        "pow" => Ok(Val::Num(a(0).powf(a(1)))),
        "log" => Ok(Val::Num(a(0).max(1e-30).ln())),
        "exp" => Ok(Val::Num(a(0).clamp(-50.0, 50.0).exp())),
        "tanh" => Ok(Val::Num(a(0).tanh())),
        "clamp" => Ok(Val::Num(a(0).clamp(a(1), a(2)))),

        "<" => Ok(Val::Bool(a(0) < a(1))),
        ">" => Ok(Val::Bool(a(0) > a(1))),
        "<=" => Ok(Val::Bool(a(0) <= a(1))),
        ">=" => Ok(Val::Bool(a(0) >= a(1))),
        "=" | "eq?" => Ok(Val::Bool((a(0) - a(1)).abs() < 1e-10)),
        "not" => Ok(Val::Bool(
            !args.first().map(|v| v.is_truthy()).unwrap_or(false),
        )),

        "car" | "first" => match args.first() {
            Some(Val::List(l)) => Ok(l.first().cloned().unwrap_or(Val::Nil)),
            _ => Ok(Val::Nil),
        },
        "cdr" | "rest" => match args.first() {
            Some(Val::List(l)) if l.len() > 1 => Ok(Val::List(l[1..].to_vec())),
            _ => Ok(Val::Nil),
        },
        "cons" => {
            let head = args.first().cloned().unwrap_or(Val::Nil);
            match args.get(1) {
                Some(Val::List(l)) => {
                    let mut v = vec![head];
                    v.extend(l.iter().cloned());
                    Ok(Val::List(v))
                }
                Some(Val::Nil) | None => Ok(Val::List(vec![head])),
                Some(tail) => Ok(Val::List(vec![head, tail.clone()])),
            }
        }
        "list" => Ok(Val::List(args.to_vec())),
        "length" | "len" => match args.first() {
            Some(Val::List(l)) => Ok(Val::Num(l.len() as f64)),
            _ => Ok(Val::Num(0.0)),
        },
        "null?" => match args.first() {
            Some(Val::Nil) => Ok(Val::Bool(true)),
            Some(Val::List(l)) => Ok(Val::Bool(l.is_empty())),
            _ => Ok(Val::Bool(false)),
        },
        "number?" => Ok(Val::Bool(matches!(args.first(), Some(Val::Num(_))))),
        "list?" => Ok(Val::Bool(matches!(args.first(), Some(Val::List(_))))),
        "append" => {
            let mut out = Vec::new();
            for arg in args {
                if let Val::List(l) = arg {
                    out.extend(l.iter().cloned());
                }
            }
            Ok(Val::List(out))
        }
        "print" | "println" | "display" => {
            if print_buf.len() < MAX_PRINT_LINES {
                let s = args.iter().map(|a| format!("{a}")).collect::<Vec<_>>().join(" ");
                let truncated: String = s.chars().take(MAX_PRINT_LINE_LEN).collect();
                print_buf.push(truncated);
            }
            Ok(args.first().cloned().unwrap_or(Val::Nil))
        }

        // unknown → return as symbol
        _ => Ok(Val::Sym(name.into())),
    }
}

// ── extract comments from Lisp source ────────────────────────────────────────

/// Extract semicolon comments from Lisp source as natural language lines.
pub fn extract_comments(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|line| {
            if let Some(pos) = line.find(';') {
                let comment = line[pos + 1..].trim();
                if !comment.is_empty() {
                    Some(comment.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

// ── hardwired Lisp kernel ─────────────────────────────────────────────────────

/// The kernel is evaluated once when a snake is born, defining utility functions
/// in its persistent `Env`. Personality code (snippets) can call these.
/// This is the "brainstem" — instincts that don't change, even when
/// Lisp-eats-Lisp recombines the personality layer.
pub const KERNEL: &str = r#"
(begin
  ; clamp a value to 0..1
  (define (clamp01 v) (clamp v 0 1))

  ; normalise a value from 0..max into 0..1
  (define (norm v lo hi)
    (clamp01 (/ (- v lo) (+ (- hi lo) 0.001))))

  ; sigmoid squash — keeps outputs smooth
  (define (squash v) (/ 1 (+ 1 (exp (- v)))))

  ; safe division — never divide by zero
  (define (safe/ a b) (if (< (abs b) 0.001) 0 (/ a b)))

  ; energy urgency — high when energy is low
  (define (urgency) (- 1 (clamp01 e)))

  ; am I old?
  (define (elder?) (> age 200))

  ; mood descriptor for println
  (define (mood-word)
    (if (> mood 0.7) "happy"
      (if (< mood 0.3) "grim" "calm")))

  ; energy status word
  (define (energy-word)
    (if (> e 0.8) "strong"
      (if (< e 0.3) "fading" "okay")))

  ; standard status report — call from personality code
  (define (status)
    (println (mood-word) (energy-word)))

  ; weighted average — useful for blending decisions
  (define (mix a b t) (+ (* a (- 1 t)) (* b t)))

  ; approach/avoid score based on distance
  (define (approach-score)
    (if (< d 2) 1.5 (if (< d 5) 0.5 -0.5)))

  ; pulse — oscillates with age, useful for rhythm
  (define (pulse freq) (sin (* age freq 0.01)))

  nil)
"#;

/// Boot the kernel into a fresh Env.  Called once per snake at birth.
pub fn boot_kernel(env: &mut Env) {
    let mut budget: usize = 500;
    let mut print_buf = Vec::new();
    if let Ok(ast) = parse(KERNEL) {
        let _ = eval(&ast, env, &mut budget, &mut print_buf);
    }
}

// ── Lisp code snippets for snakes ────────────────────────────────────────────

pub const SNIPPETS: &[&str] = &[
    // 0 – the optimist (uses kernel: status, urgency)
    "; the optimist\n(begin\n  (status)\n  (if (> mood 0.7) (println \"everything is wonderful!\")\n    (println \"even now, I believe\"))\n  (- 1 (urgency)))",
    // 1 – the coward
    "; the coward\n(begin\n  (if (< mood 0.3) (println \"I am terrified\")\n    (if (> mood 0.7) (println \"maybe it is safe...\")\n      (println \"nervous...\")))\n  (if (> d 3) (println \"too far, I run\") (println \"oh no, so close!\"))\n  (if (> d 3) 1 -1))",
    // 2 – the monk (uses kernel: status, squash, mix)
    "; the monk\n(begin\n  (status)\n  (println \"balance in all things\")\n  (mix e oe (squash mood)))",
    // 3 – the taxman
    "; the taxman\n(begin\n  (if (> mood 0.7) (println \"revenues are up!\")\n    (if (< mood 0.3) (println \"austerity measures...\")\n      (println \"paying my dues\")))\n  (if (> d 5) (println \"distance is costly\"))\n  (- e (/ d 10)))",
    // 4 – the gambler
    "; the gambler\n(begin\n  (if (> mood 0.7) (println \"I am on a hot streak!\")\n    (if (< mood 0.3) (println \"the house always wins\")\n      (println \"playing it safe\")))\n  (if (< e 0.3) (println \"all in!\") (println \"holding cards\"))\n  (if (< e 0.3) -1 1))",
    // 5 – the diplomat (uses kernel: approach-score, status)
    "; the diplomat\n(begin\n  (status)\n  (if (> (approach-score) 0) (println \"we can work together\")\n    (println \"let us negotiate\"))\n  (safe/ e (+ d 1)))",
    // 6 – the gardener
    "; the gardener\n(begin\n  (if (> mood 0.7) (println \"the garden is paradise!\")\n    (if (< mood 0.3) (println \"weeds everywhere...\")\n      (println \"tending my garden\")))\n  (if (> e 0.6) (println \"blooming nicely\") (println \"needs more rain\"))\n  (* (+ e 1) 0.8))",
    // 7 – the magnet
    "; the magnet\n(begin\n  (if (> mood 0.7) (println \"the attraction is electric!\")\n    (if (< mood 0.3) (println \"repelling everything\")\n      (println \"searching for company\")))\n  (if (< d 2) (println \"come closer\"))\n  (if (< d 2) 2 -2))",
    // 8 – the warrior (uses kernel: urgency, approach-score)
    "; the warrior\n(begin\n  (println (mood-word) \"in battle\")\n  (if (> (urgency) 0.5) (println \"wounded but fighting\")\n    (println \"FORWARD!\"))\n  (+ (approach-score) (* e 2)))",
    // 9 – the dreamer
    "; the dreamer\n(begin\n  (if (> mood 0.7)\n    (begin (println \"I see the stars\") (println \"anything is possible\"))\n    (if (< mood 0.3) (println \"nightmares haunt me\")\n      (println \"waiting for dawn\")))\n  (if (> e 0.7) 3 0))",
    // 10 – the cartographer
    "; the cartographer\n(begin\n  (if (> mood 0.7) (println \"discovered new lands!\")\n    (if (< mood 0.3) (println \"lost in the fog\")\n      (println \"mapping the world\")))\n  (println (if (> d 5) \"uncharted territory\" \"known ground\"))\n  (+ e (* d 0.1)))",
    // 11 – the poet (uses kernel: pulse, status)
    "; the poet\n(begin\n  (status)\n  (if (> (pulse 7) 0) (println \"rising verse\") (println \"falling refrain\"))\n  (* (pulse 7) d))",
    // 12 – the rival
    "; the rival\n(begin\n  (if (> mood 0.7) (println \"I am supreme!\")\n    (if (< mood 0.3) (println \"they mock me\")\n      (println \"the contest continues\")))\n  (if (> oe e) (println \"I must retreat\") (println \"I press forward\"))\n  (if (> oe e) -1 1))",
    // 13 – the communist (uses kernel: mix, status)
    "; the communist\n(begin\n  (status)\n  (println \"from each, to each\")\n  (mix e oe 0.5))",
    // 14 – the thief
    "; the thief\n(begin\n  (if (> mood 0.7) (println \"the heist went perfectly!\")\n    (if (< mood 0.3) (println \"caught red-handed\")\n      (println \"eyeing the goods\")))\n  (if (> e oe) (println \"what is yours is mine\"))\n  (- e oe))",
    // 15 – the hunter
    "; the hunter\n(begin\n  (if (> mood 0.7) (println \"the prey is mine!\")\n    (if (< mood 0.3) (println \"the trail has gone cold\")\n      (println \"stalking quietly\")))\n  (if (< d 1) (println \"GOTCHA!\"))\n  (if (< d 1) (* e 3) e))",
    // 16 – the sailor
    "; the sailor\n(begin\n  (if (> mood 0.7) (println \"fair winds and following seas!\")\n    (if (< mood 0.3) (println \"storm on the horizon\")\n      (println \"drifting on the current\")))\n  (if (> (cos d) 0) (println \"wind at my back\") (println \"tacking against\"))\n  (* e (cos d)))",
    // 17 – the musician
    "; the musician\n(begin\n  (if (> mood 0.7) (println \"standing ovation!\")\n    (if (< mood 0.3) (println \"out of tune...\")\n      (println \"playing my song\")))\n  (if (> (sin (* d 3)) 0) (println \"crescendo!\") (println \"soft passage\"))\n  (+ e (sin (* d 3))))",
    // 18 – the snake charmer
    "; the snake charmer\n(begin\n  (if (> mood 0.7) (println \"the serpent dances!\")\n    (if (< mood 0.3) (println \"the rhythm falters\")\n      (println \"hypnotic rhythm\")))\n  (if (> (* e d) 1) (println \"trance deepens\") (println \"swaying gently\"))\n  (tanh (* e d)))",
    // 19 – the elder (uses kernel: elder?, status, squash)
    "; the elder\n(begin\n  (status)\n  (if (elder?) (println \"I have seen much\")\n    (println \"still learning\"))\n  (if (elder?) (squash oe) e))",
    // 20 – the explorer
    "; the explorer\n(begin\n  (if (> mood 0.7) (println \"what a discovery!\")\n    (if (< mood 0.3) (println \"hopelessly lost\")\n      (println \"mind the gap\")))\n  (if (> (abs (- e oe)) 0.5) (println \"vast difference here\") (println \"we are alike\"))\n  (* (abs (- e oe)) d))",
    // 21 – the scientist
    "; the scientist\n(let ((x (+ e d)))\n  (begin\n    (if (> mood 0.7) (println \"eureka!\")\n      (if (< mood 0.3) (println \"experiment failed\")\n        (println \"measuring...\")))\n    (println x)\n    (if (> x 1) (println \"above threshold\") (println \"below threshold\"))\n    (* x 0.3)))",
    // 22 – the strategist (uses kernel: approach-score, urgency)
    "; the strategist\n(begin\n  (println (mood-word) \"calculating\")\n  (let ((score (approach-score))\n        (need (urgency)))\n    (println score need)\n    (+ score (* need 2))))",
    // 23 – the peacemaker
    "; the peacemaker\n(begin\n  (define x (+ e oe))\n  (if (> mood 0.7) (println \"harmony achieved!\")\n    (if (< mood 0.3) (println \"conflict everywhere\")\n      (println \"seeking harmony\")))\n  (if (> x 1) (println \"together we thrive\") (println \"we must conserve\"))\n  (/ x 2))",
];

pub fn random_snippet(idx: usize) -> &'static str {
    SNIPPETS[idx % SNIPPETS.len()]
}

/// Combine two parent snake codes using homoiconicity — lisp eats lisp.
/// Parses both codes to Val (AST), then uses code-as-data operations
/// (cons, car, cdr, append) to produce a new valid Lisp expression.
/// The child's comment is synthesised from both parents.
pub fn combine_snippets(a: &str, b: &str, seed: usize) -> String {
    // Strip comments for code parsing, but gather parent comments for child
    let comment_a = extract_comments(a);
    let comment_b = extract_comments(b);
    let child_comment = match (comment_a.first(), comment_b.first()) {
        (Some(ca), Some(cb)) => {
            let merged: String = format!("{ca} + {cb}").chars().take(38).collect();
            format!("; {merged}\n")
        }
        (Some(ca), None) => format!("; {ca} child\n"),
        (None, Some(cb)) => format!("; {cb} child\n"),
        (None, None) => "; offspring\n".to_string(),
    };

    let code_a = strip_comments(a);
    let code_b = strip_comments(b);

    let ast_a = parse(&code_a).unwrap_or(Val::Sym("e".into()));
    let ast_b = parse(&code_b).unwrap_or(Val::Sym("oe".into()));

    let child_expr = match seed % 8 {
        // cons: prepend operator from one parent onto the other's arg list
        0 => {
            let op = head_sym(&ast_a);
            Val::List(vec![op, ast_a.clone(), ast_b.clone()])
        }
        // car: take the core (first sub-expr) of one, wrap with the other
        1 => {
            let core = car(&ast_a);
            let shell_op = head_sym(&ast_b);
            Val::List(vec![shell_op, core, Val::Sym("e".into())])
        }
        // cdr: take the tail args of parent a, apply parent b's operator
        2 => {
            let tail = cdr(&ast_a);
            let op = head_sym(&ast_b);
            let mut new = vec![op];
            if let Val::List(items) = tail {
                new.extend(items);
            } else {
                new.push(tail);
            }
            Val::List(new)
        }
        // append: merge argument lists under a new operator
        3 => {
            let args_a = cdr(&ast_a);
            let args_b = cdr(&ast_b);
            let mut merged = vec![Val::Sym("+".into())];
            if let Val::List(items) = args_a { merged.extend(items); }
            if let Val::List(items) = args_b { merged.extend(items); }
            // limit arg count to keep expressions short
            merged.truncate(4);
            Val::List(merged)
        }
        // if-gate: wrap both parent expressions in a conditional
        4 => Val::List(vec![
            Val::Sym("if".into()),
            Val::List(vec![Val::Sym(">".into()), Val::Sym("e".into()), Val::Num(0.5)]),
            ast_a,
            ast_b,
        ]),
        // quote-eval: literally quote one parent and eval the other
        5 => Val::List(vec![
            Val::Sym("begin".into()),
            Val::List(vec![Val::Sym("define".into()), Val::Sym("parent".into()),
                Val::List(vec![Val::Sym("quote".into()), ast_a])]),
            ast_b,
        ]),
        // let-bind both parent results then combine
        6 => Val::List(vec![
            Val::Sym("let".into()),
            Val::List(vec![
                Val::List(vec![Val::Sym("pa".into()), ast_a]),
                Val::List(vec![Val::Sym("pb".into()), ast_b]),
            ]),
            Val::List(vec![Val::Sym("/".into()),
                Val::List(vec![Val::Sym("+".into()), Val::Sym("pa".into()), Val::Sym("pb".into())]),
                Val::Num(2.0)]),
        ]),
        // random fresh snippet (mutation)
        _ => {
            let fresh = SNIPPETS[seed / 8 % SNIPPETS.len()];
            let code = strip_comments(fresh);
            parse(&code).unwrap_or(Val::Sym("e".into()))
        }
    };

    // Render the child Val back to Lisp source (code as data → data as code)
    let code_str = format!("{child_expr}");
    // Truncate very long expressions to keep display manageable
    let truncated: String = code_str.chars().take(80).collect();
    format!("{child_comment}{truncated}")
}

/// Extract the head symbol (operator) from a Lisp expression
fn head_sym(v: &Val) -> Val {
    match v {
        Val::List(items) if !items.is_empty() => items[0].clone(),
        _ => Val::Sym("+".into()),
    }
}

/// car: first element of a list, or the value itself
fn car(v: &Val) -> Val {
    match v {
        Val::List(items) if !items.is_empty() => items[0].clone(),
        _ => v.clone(),
    }
}

/// cdr: rest of a list, or nil
fn cdr(v: &Val) -> Val {
    match v {
        Val::List(items) if items.len() > 1 => Val::List(items[1..].to_vec()),
        _ => Val::Nil,
    }
}

/// Strip ; comments from source, keeping only code lines
fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|line| {
            if let Some(pos) = line.find(';') {
                &line[..pos]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Run a snake's Lisp code for one frame tick (cooperative multitasking).
/// Returns (numeric result, captured print lines).
pub fn tick(
    code: &str,
    env: &mut Env,
    energy: f64,
    x: f64,
    y: f64,
    age: f64,
    mood: f64,
    budget: usize,
) -> (f64, Vec<String>) {
    env.set("e", Val::Num(energy));
    env.set("x", Val::Num(x));
    env.set("y", Val::Num(y));
    env.set("age", Val::Num(age));
    env.set("mood", Val::Num(mood));
    if env.get("d").is_none() {
        env.set("d", Val::Num(99.0));
    }
    if env.get("oe").is_none() {
        env.set("oe", Val::Num(0.0));
    }

    let val = match parse(code) {
        Ok(v) => v,
        Err(_) => return (0.0, vec![]),
    };
    let mut b = budget;
    let mut print_buf = Vec::new();
    match eval(&val, env, &mut b, &mut print_buf) {
        Ok(r) => (r.as_num().clamp(-10.0, 10.0), print_buf),
        Err(_) => (0.0, print_buf),
    }
}

/// Evaluate interaction between two meeting snakes.
/// Returns energy delta from the perspective of the evaluating snake.
pub fn eval_interaction(
    code: &str,
    own_energy: f64,
    other_energy: f64,
    dist: f64,
    age: f64,
) -> f64 {
    let mut env = Env::new();
    env.set("e", Val::Num(own_energy));
    env.set("oe", Val::Num(other_energy));
    env.set("d", Val::Num(dist));
    env.set("age", Val::Num(age));

    let val = match parse(code) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };
    let mut budget: usize = 200;
    let mut print_buf = Vec::new();
    match eval(&val, &mut env, &mut budget, &mut print_buf) {
        Ok(r) => r.as_num().clamp(-10.0, 10.0),
        Err(_) => 0.0,
    }
}
