use monster_chess::{board::Board, bitboard::BitBoard};

pub fn save_features<const T: usize>(features: &mut Vec<i16>, board: &Board<T>, flips: &Vec<usize>) {
    let piece_types = board.game.pieces.len() as usize;
    let teams = board.game.teams as usize;
    let squares = board.game.squares as usize;
    for mut square in 0..squares {
        square = flip(square, board.state.moving_team, flips);
        let bitboard = BitBoard::<T>::from_lsb(square as u16);

        if (board.state.all_pieces & bitboard).is_empty() {
            continue;
        }

        let mut team = usize::MAX;
        for ind in 0..teams {
            if (board.state.teams[ind as usize] & bitboard).is_set() {
                team = ind as usize;
            }
        }

        team = if team == board.state.moving_team as usize { 0 } else { 1 };

        let mut piece_type = usize::MAX;
        for ind in 0..piece_types {
            if (board.state.pieces[ind] & bitboard).is_set() {
                piece_type = ind;
                break;
            }
        }

        features[square + squares * team + (squares * teams) * piece_type] = 1;
    }
}

pub fn get_features<const T: usize>(board: &Board<T>, flips: &Vec<usize>) -> Vec<i16> {
    let mut features: Vec<i16> = vec![0; board.game.pieces.len() * (board.game.teams * board.game.squares) as usize];
    save_features(&mut features, board, flips);
    features
}

pub fn create_flips<const T: usize>(board: &Board<T>) -> Vec<usize> {
    let mut flips: Vec<usize> = vec![];
    for square in 0..board.game.squares {
        let mut bitboard = BitBoard::<T>::from_lsb(square);
        bitboard = bitboard.flip_vertically(&board.state.ranks, board.game.cols, board.game.rows);
        flips.insert(square as usize, bitboard.bitscan_forward() as usize);
    }

    flips
}

pub fn flip(mut square: usize, moving_team: u16, flips: &Vec<usize>) -> usize {
    if moving_team == 1 {
        square = flips[square];
    } 

    return square;
}