use monster_chess::board::{game::{NORMAL_MODE, GameResults}, Board, actions::Move};

use crate::{nnue::{NNUE, eval_nnue}, train::{get_features, save_features}};

pub const MAX_SCORE: i32 = 1_000_000_000;
pub const MIN_SCORE: i32 = -1_000_000_000;

pub fn evaluate<const T: usize>(
    search_info: &mut SearchInfo, 
    board: &mut Board<T>
) -> i32 {
    save_features(&mut search_info.layers[0], board, &search_info.flips);
    eval_nnue(search_info)[0] as i32
}

pub struct SearchInfo<'a> {
    pub best_move: Option<Move>,
    pub nnue: &'a NNUE,
    pub layers: Vec<Vec<i16>>,
    pub flips: Vec<usize>,
    pub nodes: u128
}

pub fn negamax<const T: usize>(
    search_info: &mut SearchInfo, 
    board: &mut Board<T>, 
    mut alpha: i32, mut beta: i32,
    depth: u32, ply: u32
) -> i32 {
    if depth == 0 { return evaluate(search_info, board); }

    let moves = board.generate_legal_moves(NORMAL_MODE);

    let mut best_score: i32 = MIN_SCORE;
    let mut best_move: Option<Move> = None;

    // This line is to fix a bug where sometimes, all the move scores are MIN_SCORE, so it can't pick any.
    // In this case, we'll just pick the first one.
    // I'll find a true fix for this later.
    if moves.len() > 0 {
        best_move = Some(moves[0]);
    }

    for action in moves {
        search_info.nodes += 1;
        let undo = board.make_move(&action);
        let score = -negamax(search_info, board, -beta, -alpha, depth - 1, ply + 1);
        board.undo_move(undo);
        if score > best_score {
            best_score = score;
            best_move = Some(action);
            if score > alpha {
                alpha = score;
            }
        }
        if score > beta {
            best_move = Some(action);
            break;
        }
    }

    if ply == 0 {
        search_info.best_move = best_move;
    }

    return alpha;
}