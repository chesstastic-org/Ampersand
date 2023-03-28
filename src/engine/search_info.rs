use monster_chess::board::{actions::Move, Board};

use super::{nnue::{NNUE, alloc_layers}, ordering::{KillerMoves, HistoryInfo, MAX_KILLER_MOVES, TranspositionEntry}, pv::{PV, MAX_DEPTH}, features::create_flips};

pub struct SearchInfo<'a, const T: usize> {
    pub best_move: Option<Move>,
    pub nnue: &'a NNUE,
    pub layers: Vec<Vec<i16>>,
    pub flips: Vec<usize>,
    pub nodes: u128,
    pub transposition_table: Vec<Option<TranspositionEntry>>,
    pub transposition_size: usize,
    pub pv_table: PV<T>,
    pub history_info: Vec<Vec<Vec<Option<HistoryInfo>>>>,
    pub killer_moves: KillerMoves,
    pub hashes: Vec<u64>
}

pub fn create_search_info<'a, const T: usize>(board: &Board<T>, nnue: &'a NNUE) -> SearchInfo<'a, T> {
    let squares = board.game.squares as usize;

    SearchInfo {
        best_move: None,
        nnue,
        nodes: 0,
        flips: create_flips(&board),
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
    }
}