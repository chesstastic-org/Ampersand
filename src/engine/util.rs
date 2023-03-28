use monster_chess::board::{actions::Move, Board};

use crate::{engine::{search_info::SearchInfo, nnue::{NNUE, alloc_layers}, ordering::MAX_KILLER_MOVES, pv::{MAX_DEPTH, PV}}};

use super::features::create_flips;

pub const MAX_SCORE: i32 = 1_000_000_000;
pub const MIN_SCORE: i32 = -1_000_000_000;

pub fn compare_moves(action: &Move, other: &Move) -> bool {
    match action {
        Move::Pass => other.is_pass(),
        Move::Action(action) => {
            match other {
                Move::Pass => false,
                Move::Action(other) => {
                    action.from == other.from && action.to == other.to && action.info == other.info
                }
            }
        }
    }
}