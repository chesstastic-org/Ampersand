use std::{thread::{Thread, self}, time::Duration};

use monster_chess::board::{Board, actions::Move, game::NORMAL_MODE};
use monster_ugi::engine::{EngineBehavior, EngineInfo, TimeControl, MoveSelectionResults, Info};

use crate::{negamax::{SearchInfo, negamax, MIN_SCORE, MAX_SCORE, negamax_iid}, nnue::{NNUE, eval_nnue, load_nnue, alloc_layers}, train::{get_features, save_features, create_flips}, get_time_ms};

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

    fn select_move(&mut self, board: &mut Board<T>, time_control: TimeControl) -> MoveSelectionResults {
        let nnue = &self.0;

        let mut search_info = SearchInfo {
            best_move: None,
            nnue,
            nodes: 0,
            flips: create_flips(board),
            layers: alloc_layers(nnue),
            transposition_table: vec![ None; 1_000_000 ],
            transposition_size: 1_000_000
        };
        
        let eval = negamax_iid(&mut search_info, board, 1500) as u64;

        MoveSelectionResults {
            best_move: search_info.best_move.expect("Could not find best move."),
            evaluation: eval
        }
    }

    fn stop_search(&mut self) {
        
    }
}