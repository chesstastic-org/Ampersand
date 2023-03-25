use monster_chess::{
    games::{chess::Chess}, 
    board::{game::{NORMAL_MODE, GameResults}, Board}, bitboard::BitBoard
};
use rand::{rngs::ThreadRng, thread_rng};
use rand::prelude::SliceRandom;

use std::time::{SystemTime, UNIX_EPOCH};
use std::fs;
use std::io::Write;

fn get_time_ms() -> u128 {
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards.")
        .as_millis()
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

pub fn save_features<const T: usize>(features: &mut Vec<i16>, board: &Board<T>, flips: &Vec<usize>) {
    let piece_types = board.game.pieces.len() as usize;
    let teams = board.game.teams as usize;
    let squares = board.game.squares as usize;
    for mut square in 0..squares {
        if board.state.moving_team == 1 {
            square = flips[square];
        } 
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

fn generate_random_game(rng: &mut ThreadRng) -> Vec<(Vec<i16>, i32)> {
    let game = Chess::create();
    let mut board = game.default();
    let mut move_count = 0;

    let mut uci_positions: Vec<(Vec<i16>, u16)> = vec![];

    let mut result = 0;

    let flips = &create_flips(&board);

    loop {
        let moves = board.generate_legal_moves(NORMAL_MODE);
        match game.resolution.resolve(&mut board, &moves) {
            GameResults::Win(team) => {
                result = if team == 0 { 1 } else { -1 };
                break;
            },
            GameResults::Draw => { break; },
            GameResults::Ongoing => {}
        }
        if board.state.sub_moves >= 100 {
            break;
        }

        let action = moves.choose(rng).copied().expect("Could not get random move.");

        uci_positions.push((get_features(&board, flips), board.state.moving_team));
        board.make_move(&action);
        move_count += 1;
    }

    uci_positions.choose_multiple(rng, 20)
        .map(|(fen, moving_team)| (fen.clone(), match moving_team {
            0 => result,
            1 => -result,
            _ => result
        }))
        .collect::<Vec<_>>()
}

pub fn generate_random_data(){ 
    let mut rng = thread_rng();

    let mut file = fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open("positions.txt")
        .unwrap();
    

    let mut position_count = 0;
    let mut iter = 0;
    while position_count < 2_000_000 {
        let start = get_time_ms();
        let mut positions: Vec<(Vec<i16>, i32)> = vec![];
        for _ in 0..1000 {       
            let game = generate_random_game(&mut rng); 
            positions.extend(game);
        }

        let mut draws = positions.iter().filter(|(_, result)| result == &0).count();
        let mut wins = positions.len() - draws;

        while draws > (wins / 2) {
            let draw_index = positions.iter().position(|(_, result)| result == &0).expect("Could not find a draw.");
            positions.swap_remove(draw_index);

            draws -= 1;
        }

        while (2 * wins) > draws  {
            let win_index = positions.iter().position(|(_, result)| result != &0).expect("Could not find a win.");
            positions.swap_remove(win_index);

            wins -= 1;
        }

        for (features, score) in positions {
            let features = features.iter().map(|el| el.to_string()).collect::<Vec<_>>().join(".");
            writeln!(file, "{features}: {:.1}", score).unwrap();
            position_count += 1;
        }
        let end = get_time_ms();
        println!("{iter}: {} ({} positions)", end - start, position_count);
        iter += 1;
    }
}