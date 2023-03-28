use monster_chess::board::actions::Move;

use crate::{engine::{util::compare_moves, search_info::SearchInfo, pv::MAX_DEPTH}};

pub const MAX_KILLER_MOVES: usize = 5;
pub type KillerMoves = [[Option<Move>; MAX_KILLER_MOVES]; MAX_DEPTH];

pub fn store_killer_move<const T: usize>(search_info: &mut SearchInfo<T>, ply: u32, action: Move) {
    let ply = ply as usize;
    let first_killer = search_info.killer_moves[ply][0];

    // First killer must not be the same as the move being stored.
    if let Some(first_killer) = first_killer {
        if !compare_moves(&action, &first_killer) {
            // Shift all the moves one index upward...
            for i in (1..MAX_KILLER_MOVES).rev() {
                let n = i as usize;
                let previous = search_info.killer_moves[ply][n - 1];
                search_info.killer_moves[ply][n] = previous;
            }

            // and add the new killer move in the first spot.
            search_info.killer_moves[ply][0] = Some(action);
        }
    }
}