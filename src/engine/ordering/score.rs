use monster_chess::board::{actions::Move, Board};

use crate::engine::{util::compare_moves, search_info::SearchInfo};

use super::{MAX_KILLER_MOVES, TranspositionEntry};

const KILLER_DECAY: [ u32; 5 ] = [ 0, 1, 2, 3, 4 ];

pub fn score_action<const T: usize>(board: &Board<T>, search_info: &mut SearchInfo<T>, action: &Move, tt_entry: &Option<TranspositionEntry>, ply: u32) -> u32 {
    if let Some(entry) = tt_entry {
        let best_move = entry.best_move;

        let is_tt_move = compare_moves(action, &best_move);
        if is_tt_move {
            return 1_000_000;
        }
    }

    
    let mut score = 0;

    let ply = ply as usize;
    let mut i = 0;
    while i < MAX_KILLER_MOVES {
        let killer = search_info.killer_moves[ply][i];
        if let Some(killer) = killer {
            if compare_moves(action, &killer) {
                // Killer Moves that happen later should be much less resistant to history/countermove changes.
                let killer_loss = KILLER_DECAY[i];

                score += 100_000 - killer_loss;
            }
        }
        i += 1;
    }

    let base_move = action;
    if let Move::Action(action) = action {
        if let Some(from) = action.from {
            let history_entry = search_info.history_info[board.state.moving_team as usize][from as usize][action.to as usize];
            
            if let Some(history_entry) = history_entry {
                score += history_entry.inc;
                
                if let Some(counter_move) = history_entry.counter_move {
                    if compare_moves(base_move, &counter_move) {
                        score += 1_000;
                    }
                }
            }
        }
    }

    score
}