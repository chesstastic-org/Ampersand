use std::{thread::{Thread, self}, time::Duration};

use monster_chess::board::{Board, actions::Move, game::NORMAL_MODE};
use monster_ugi::engine::{EngineBehavior, EngineInfo, TimeControl, MoveSelectionResults, Info};

use crate::{negamax::{SearchInfo, negamax, MIN_SCORE, MAX_SCORE}, nnue::{NNUE, eval_nnue, load_nnue, alloc_layers}, train::{get_features, save_features, create_flips}, get_time_ms};

pub struct SimpleEngine(pub Option<NNUE>);

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
        if self.0.is_none() {
            self.0 = Some(load_nnue("./model_weights.json"));
        }

        let nnue = self.0.as_ref().expect("Must have NNUE!");

        let start = get_time_ms();
        let mut search_info = SearchInfo {
            best_move: None,
            nnue,
            nodes: 0,
            flips: create_flips(board),
            layers: alloc_layers(nnue)
        };
        
        let eval = negamax(&mut search_info, board, MIN_SCORE, MAX_SCORE, 4, 0) as u64;
        let end = get_time_ms();

        MoveSelectionResults {
            best_move: search_info.best_move.expect("Could not find best move."),
            evaluation: eval
        }
    }

    fn stop_search(&mut self) {
        
    }
}