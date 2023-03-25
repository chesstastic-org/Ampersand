use std::{env, time::{SystemTime, UNIX_EPOCH}};

use monster_chess::games::chess::Chess;
use monster_ugi::{ugi::run_ugi, engine::Engine, random::RandomEngine};
use nnue::{load_nnue};
use rand::thread_rng;
use train::get_features;

use crate::{negamax::evaluate, nnue::eval_nnue, train::generate_random_data, engine::SimpleEngine};

mod negamax;
mod train;
mod nnue;
mod engine;

fn get_time_ms() -> u128 {
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards.")
        .as_millis()
}

fn main() {
    generate_random_data();
    return;

    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args.contains(&"random".to_string()) {
        let engine = Engine {
            behavior: Box::new(RandomEngine(thread_rng())),
            game: Chess::create()
        };
    
        run_ugi(engine)
    } else {
        let engine = Engine {
            behavior: Box::new(SimpleEngine(None)),
            game: Chess::create()
        };
    
        run_ugi(engine);
    }
}