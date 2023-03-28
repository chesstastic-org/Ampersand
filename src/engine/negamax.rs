use std::cmp::Reverse;

use monster_chess::{board::{game::{NORMAL_MODE, GameResults}, Board, actions::{Move, Action, HistoryMove, HistoryState, HistoryUpdate, IndexedPreviousBoard}}, games::chess::ATTACKS_MODE};
use serde_json::value::Index;

use super::{
    nnue::{NNUE, eval_nnue, apply_hidden, update_hidden, relu, register_hidden_updates}, pv::{PV, MAX_DEPTH}, util::{MIN_SCORE, MAX_SCORE, compare_moves}, ordering::{KillerMoves, store_killer_move, MAX_KILLER_MOVES, TranspositionEntry, HistoryInfo, score_action},
    super::{get_time_ms}, features::save_features, search_info::SearchInfo
};

pub fn evaluate<const T: usize>(
    search_info: &mut SearchInfo<T>, 
    board: &mut Board<T>
) -> i32 {
    (eval_nnue(search_info)[0] as i32) / (64 * 64)
}

pub struct ScoredMove {
    action: Move,
    score: u32
}

fn make_move<const T: usize>(board: &mut Board<T>, search_info: &mut SearchInfo<T>, action: &Move) -> Option<HistoryMove<T>> {
    let undo = board.make_move(&action);
    search_info.hashes.push(board.game.zobrist.compute(&board));

    register_hidden_updates(search_info, board, &undo, false);

    undo
}

fn undo_move<const T: usize>(board: &mut Board<T>, search_info: &mut SearchInfo<T>, undo: Option<HistoryMove<T>>) {
    search_info.hashes.pop();

    register_hidden_updates(search_info, board, &undo, true);

    board.undo_move(undo);
}

pub fn negamax<const T: usize>(
    search_info: &mut SearchInfo<T>, 
    board: &mut Board<T>, 
    mut alpha: i32, mut beta: i32,
    depth: u32, ply: u32
) -> i32 {
    if depth == 0 { return evaluate(search_info, board); }
    
    search_info.pv_table.init_pv(ply);

    let len = search_info.hashes.len();


    if ply > 0 && len >= 5 && search_info.hashes[len - 1] == search_info.hashes[len - 5] {
        return 0;
    }

    let moves = board.generate_legal_moves(NORMAL_MODE);

    match board.game.resolution.resolve(board, &moves) {
        GameResults::Draw => {
            return 0;
        },
        GameResults::Win(team) => {
            if board.state.moving_team == team {
                return MAX_SCORE - (ply as i32);
            }

            return MIN_SCORE + (ply as i32);
        },
        GameResults::Ongoing => {}
    };

    let hash = (search_info.hashes[len - 1] as usize) % search_info.transposition_size;

    let tt_entry = search_info.transposition_table[hash].clone();
    if let Some(tt_entry) = tt_entry {
        if tt_entry.depth >= depth {
            if ply == 0 {
                search_info.pv_table.update_pv(ply, Some(tt_entry.best_move));
                search_info.best_move = Some(tt_entry.best_move);
            }
            return tt_entry.eval;
        }
    }

        
    let mut moves = moves.iter()
        .map(|action| ScoredMove { action: *action, score: score_action(&board, search_info, action, &tt_entry, ply) })
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

        let undo = make_move(board, search_info, &action);
        let score = -negamax(search_info, board, -beta, -alpha, depth - 1, ply + 1);
        undo_move(board, search_info, undo);
        
        if score > best_score {
            best_score = score;
            best_move = Some(action);
            search_info.pv_table.update_pv(ply, best_move);
            if score > alpha {
                alpha = score;
            }
        }
        if score >= beta {
            best_move = Some(action);

            store_killer_move(search_info, ply, action);

            if let Move::Action(action) = action {
                if let Some(from) = action.from {
                    let moving_team = board.state.moving_team as usize;
                    let from = from as usize;
                    let to = action.to as usize;
                    let history_entry = search_info.history_info[moving_team][from][to];
                    match history_entry {
                        None => {
                            let inc = depth * depth;
                            let counter_move = Move::Action(action);
                            search_info.history_info[moving_team][from][to] = Some(HistoryInfo {
                                inc,
                                counter_move: Some(counter_move)
                            });
                        },
                        Some(mut history_entry) => {
                            history_entry.inc += depth * depth;
                            history_entry.counter_move = Some(Move::Action(action));
                        }
                    }
                }
            }

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
    search_info: &mut SearchInfo<T>, 
    board: &mut Board<T>,
    max_nodes: u32
) -> i32 {
    let mut out: i32 = MIN_SCORE;
    for depth in 1..30 {
        let start = get_time_ms();
        out = negamax(search_info, board, MIN_SCORE, MAX_SCORE, depth, 0);
        let end = get_time_ms();

        let mut time = end - start;
        if time == 0 { time = 1; }
        let nodes = search_info.nodes;
        let npms = nodes / time;
        let nps = npms * 1_000;

        let pv = search_info.pv_table.display_pv(board);
        println!("info depth {depth} cp {} time {} nodes {} nps {} pv {}", out, time, nodes, nps, pv);

        if nodes > (max_nodes as u128) {
            break;
        }
    }

    return out;
}