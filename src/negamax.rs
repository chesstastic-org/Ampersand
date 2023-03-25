use std::cmp::Reverse;

use monster_chess::board::{game::{NORMAL_MODE, GameResults}, Board, actions::{Move, Action}};

use crate::{nnue::{NNUE, eval_nnue}, train::{get_features, save_features}, get_time_ms};

pub const MAX_SCORE: i32 = 1_000_000_000;
pub const MIN_SCORE: i32 = -1_000_000_000;

pub fn evaluate<const T: usize>(
    search_info: &mut SearchInfo, 
    board: &mut Board<T>
) -> i32 {
    for square in 0..search_info.layers[0].len() {
        search_info.layers[0][square] = 0;
    }
    save_features(&mut search_info.layers[0], board, &search_info.flips);
    eval_nnue(search_info)[0] as i32
}

#[derive(Clone, Copy, Debug)]
pub struct TranspositionEntry {
    pub depth: u32,
    pub eval: i32,
    pub best_move: Move
}

pub struct SearchInfo<'a> {
    pub best_move: Option<Move>,
    pub nnue: &'a NNUE,
    pub layers: Vec<Vec<i16>>,
    pub flips: Vec<usize>,
    pub nodes: u128,
    pub transposition_table: Vec<Option<TranspositionEntry>>,
    pub transposition_size: usize
}

fn compare_moves(action: &Move, other: &Move) -> bool {
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

pub fn score_action(search_info: &mut SearchInfo, action: &Move, entry: &Option<TranspositionEntry>) -> i32 {
    if let Some(entry) = entry {
        let best_move = entry.best_move;

        let is_tt_move = compare_moves(action, &best_move);
        if is_tt_move {
            return 0;
        }
    }

    0
}

pub struct ScoredMove {
    action: Move,
    score: i32
}

pub fn negamax<const T: usize>(
    search_info: &mut SearchInfo, 
    board: &mut Board<T>, 
    mut alpha: i32, mut beta: i32,
    depth: u32, ply: u32
) -> i32 {
    if depth == 0 { return evaluate(search_info, board); }

    let hash = (board.game.zobrist.compute(&board) as usize) % search_info.transposition_size;

    let entry = search_info.transposition_table[hash].clone();
    if let Some(entry) = entry {
        if entry.depth >= depth {
            if ply == 0 {
                search_info.best_move = Some(entry.best_move);
            }
            return entry.eval;
        }
    }

    let mut moves = board.generate_legal_moves(NORMAL_MODE)
        .iter()
        .map(|action| ScoredMove { action: *action, score: score_action(search_info, action, &entry) })
        .collect::<Vec<_>>();

    moves.sort_by(|a, b| b.score.cmp(&a.score));

    let mut best_score: i32 = MIN_SCORE;
    let mut best_move: Option<Move> = None;

    // This line is to fix a bug where sometimes, all the move scores are MIN_SCORE, so it can't pick any.
    // In this case, we'll just pick the first one.
    // I'll find a true fix for this later.
    if moves.len() > 0 {
        best_move = Some(moves[0].action);
    }

    for ScoredMove { action, .. } in moves {
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
    
    if let Some(best_move) = best_move {
        search_info.transposition_table[hash] = Some(TranspositionEntry {
            depth,
            eval: alpha,
            best_move: best_move
        });
    }

    return alpha;
}

pub fn negamax_iid<const T: usize>(
    search_info: &mut SearchInfo, 
    board: &mut Board<T>,
    max_nodes: u32
) -> i32 {
    let mut out: i32 = MIN_SCORE;
    for depth in 1..1000 {

        let start = get_time_ms();
        out = negamax(search_info, board, MIN_SCORE, MAX_SCORE, depth, 0);
        let end = get_time_ms();

        let mut time = end - start;
        if time == 0 { time = 1; }
        let nodes = search_info.nodes;
        let npms = nodes / time;
        let nps = npms * 1_000;

        println!("info depth {depth} time {} nodes {} nps {}", time, nodes, nps);

        if nodes > (max_nodes as u128) {
            break;
        }
    }

    return out;
}