use monster_chess::board::{actions::{HistoryMove, IndexedPreviousBoard, HistoryUpdate, HistoryState}, Board};

use crate::engine::{features::save_features, search_info::SearchInfo};

use super::{update_hidden, apply_hidden, relu, identity};

pub fn register_hidden_updates<const T: usize>(search_info: &mut SearchInfo<T>, board: &Board<T>, undo: &Option<HistoryMove<T>>, reverse: bool) {
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