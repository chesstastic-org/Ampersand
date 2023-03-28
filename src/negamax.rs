use std::cmp::Reverse;

use monster_chess::{board::{game::{NORMAL_MODE, GameResults}, Board, actions::{Move, Action, HistoryMove, HistoryState, HistoryUpdate, IndexedPreviousBoard}}, games::chess::ATTACKS_MODE};
use serde_json::value::Index;

use crate::{nnue::{NNUE, eval_nnue, apply_hidden, update_hidden, relu}, train::{get_features, save_features}, get_time_ms, pv_table::{PV, MAX_DEPTH}};

pub const MAX_SCORE: i32 = 1_000_000_000;
pub const MIN_SCORE: i32 = -1_000_000_000;

pub fn evaluate<const T: usize>(
    search_info: &mut SearchInfo<T>, 
    board: &mut Board<T>
) -> i32 {
    (eval_nnue(search_info)[0] as i32) / (64 * 64)
}

#[derive(Clone, Copy, Debug)]
pub struct TranspositionEntry {
    pub depth: u32,
    pub eval: i32,
    pub best_move: Move
}

#[derive(Copy, Clone, Debug)]
pub struct HistoryInfo {
    pub inc: u32,
    pub counter_move: Option<Move>
}

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
                score += 100_000 - (i as u32);
                break;
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

fn register_hidden_updates<const T: usize>(search_info: &mut SearchInfo<T>, board: &Board<T>, undo: &Option<HistoryMove<T>>, reverse: bool) {
    if let Some(undo) = undo {
        match &undo.state {
            HistoryState::Single { all_pieces, first_move, team, piece } => {
                let old_board = team.1 & piece.1;
                let current_board = board.state.teams[team.0] & board.state.pieces[piece.0];

                let changed_features = old_board ^ current_board;

                let removed_features = old_board & changed_features;
                let added_features = current_board & changed_features;

                let removed_features = removed_features.iter_set_bits(board.game.squares)
                    .map(|square| square + board.game.squares * (team.0 as u16) + (board.game.squares * board.game.teams) * (piece.0 as u16))
                    .collect::<Vec<_>>();
                let added_features = added_features.iter_set_bits(board.game.squares)
                    .map(|square| square + board.game.squares * (team.0 as u16) + (board.game.squares * board.game.teams) * (piece.0 as u16))
                    .collect::<Vec<_>>();

                if reverse {
                    update_hidden(search_info, &removed_features, &added_features, relu);
                } else {
                    update_hidden(search_info, &added_features, &removed_features, relu);
                }
            },
            HistoryState::Any { all_pieces, first_move, updates } => {
                let mut piece_updates: Vec<&IndexedPreviousBoard<T>> = vec![];
                let mut team_updates: Vec<&IndexedPreviousBoard<T>> = vec![];

                let mut added_features: Vec<u16> = Vec::with_capacity(3);
                let mut removed_features: Vec<u16> = Vec::with_capacity(3);

                for update in updates {
                    match update {
                        HistoryUpdate::Piece(piece) => {
                            piece_updates.push(piece);
                        }
                        HistoryUpdate::Team(team) => {
                            team_updates.push(team);
                        }
                    }
                }

                for piece in piece_updates {
                    for team in &team_updates {
                        let old_board = team.1 & piece.1;
                        let current_board = board.state.teams[team.0] & board.state.pieces[piece.0];
        
                        let changed_features = old_board ^ current_board;
        
                        let sub_removed_features = old_board & changed_features;
                        let sub_added_features = current_board & changed_features;
        
                        sub_removed_features.iter_set_bits(board.game.squares)
                            .map(|square| square + board.game.squares * (team.0 as u16) + (board.game.squares * board.game.teams) * (piece.0 as u16))
                            .for_each(|square| removed_features.push(square));
                        sub_added_features.iter_set_bits(board.game.squares)
                            .map(|square| square + board.game.squares * (team.0 as u16) + (board.game.squares * board.game.teams) * (piece.0 as u16))
                            .for_each(|square| added_features.push(square));
                    }
                }

                if reverse {
                    update_hidden(search_info, &removed_features, &added_features, relu);
                } else {
                    update_hidden(search_info, &added_features, &removed_features, relu);
                }
            },
            _ => {
                for square in 0..search_info.layers[0].len() {
                    search_info.layers[0][square] = 0;
                }
                save_features(&mut search_info.layers[0], board, &search_info.flips);
                if !reverse {
                    apply_hidden(search_info);
                }
            }
        }
    }
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