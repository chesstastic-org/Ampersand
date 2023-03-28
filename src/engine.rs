use std::{thread::{Thread, self}, time::Duration};

use monster_chess::board::{Board, actions::Move, game::NORMAL_MODE};
use monster_ugi::engine::{EngineBehavior, EngineInfo, TimeControl, MoveSelectionResults, Info};

use crate::{negamax::{SearchInfo, negamax, MIN_SCORE, MAX_SCORE, negamax_iid, MAX_KILLER_MOVES}, nnue::{NNUE, eval_nnue, load_nnue, alloc_layers, apply_hidden}, train::{get_features, save_features, create_flips}, get_time_ms, pv_table::{PV, MAX_DEPTH}};

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
        let squares = board.game.squares as usize;

        let mut search_info = SearchInfo {
            best_move: None,
            nnue,
            nodes: 0,
            flips: create_flips(board),
            layers: alloc_layers(nnue),
            transposition_table: vec![ None; 1_000_000 ],
            transposition_size: 1_000_000,
            history_info: vec![ 
                vec![ vec![ None; squares ]; squares ]; 2  
            ],
            pv_table: PV {
                table: [[None; MAX_DEPTH]; MAX_DEPTH],
                length: [0; MAX_DEPTH],
            },
            killer_moves: [ [ None; MAX_KILLER_MOVES ]; MAX_DEPTH ],
            hashes: Vec::with_capacity(64)
        };

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