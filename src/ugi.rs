use std::{thread::{Thread, self}, time::Duration};

use monster_chess::board::{Board, actions::Move, game::NORMAL_MODE};
use monster_ugi::engine::{EngineBehavior, EngineInfo, TimeControl, MoveSelectionResults, Info};

use crate::{
    engine::{negamax::{negamax_iid}, nnue::{NNUE, eval_nnue, load_nnue, alloc_layers, apply_hidden}, pv::{PV, MAX_DEPTH}, features::save_features, search_info::create_search_info},
    get_time_ms
};

pub struct SimpleEngine(pub NNUE);

impl<const T: usize> EngineBehavior<T> for SimpleEngine {
    fn get_engine_info(&mut self) -> EngineInfo {
        EngineInfo {
            name: "Ampersand",
            author: "Corman"
        }
    }

    fn is_ready(&mut self) -> bool {
        true
    }

    fn select_move(&mut self, board: &mut Board<T>, time_control: TimeControl, hashes: &Vec<u64>) -> MoveSelectionResults {
        let nnue = &self.0;
        let mut search_info = create_search_info(board, nnue);

        search_info.hashes = hashes.clone();
        
        for square in 0..search_info.layers[0].len() {
            search_info.layers[0][square] = 0;
        }
        save_features(&mut search_info.layers[0], board, &search_info.flips);
        apply_hidden(&mut search_info);
        let eval = negamax_iid(&mut search_info, board, 3000) as u64;

        MoveSelectionResults {
            best_move: search_info.best_move.expect("Could not find best move."),
            evaluation: eval
        }
    }

    fn stop_search(&mut self) {
        
    }
}