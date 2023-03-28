use monster_chess::{
    games::{chess::Chess}, 
    board::{game::{NORMAL_MODE, GameResults}, Board}, bitboard::BitBoard
};
use rand::{rngs::ThreadRng, thread_rng};
use rand::prelude::SliceRandom;

use std::time::{SystemTime, UNIX_EPOCH};
use std::fs;
use std::io::Write;

use crate::{engine::{pv::{MAX_DEPTH, PV}, search_info::{SearchInfo, create_search_info, SearchEnd}, negamax::{negamax_iid}, nnue::{NNUE, alloc_layers, apply_hidden}, ordering::MAX_KILLER_MOVES, features::{get_features, save_features, create_flips}}};

fn get_time_ms() -> u128 {
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards.")
        .as_millis()
}

fn generate_random_game(rng: &mut ThreadRng, nnue: &NNUE) -> Vec<(Vec<i16>, i32)> {
    let game = Chess::create();
    let mut board = game.default();
    let mut move_count = 0;

    let mut uci_positions: Vec<(Vec<i16>, u16)> = vec![];

    let mut result = 0;

    let flips = &create_flips(&board);

    let mut hashes: Vec<u64> = vec![];

    loop {
        hashes.push(game.zobrist.compute(&board));

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

        let mut search_info = create_search_info(&board, nnue, SearchEnd::Nodes(3000));

        search_info.hashes = hashes.clone();
        
        for square in 0..search_info.layers[0].len() {
            search_info.layers[0][square] = 0;
        }
        save_features(&mut search_info.layers[0], &mut board, &search_info.flips);
        apply_hidden(&mut search_info);
        let eval = negamax_iid(&mut search_info, &mut board) as u64;

        let action = search_info.best_move.expect("Could not find NNUE move");

        //let action = moves.choose(rng).copied().expect("Could not get random move.");

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

pub fn generate_random_data(nnue: &NNUE) { 
    let mut rng = thread_rng();

    let mut file = fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open("positions.txt")
        .unwrap();
    

    let mut position_count = 0;
    let mut iter = 0;
    while position_count < 1_000_000 {
        let start = get_time_ms();
        let mut positions: Vec<(Vec<i16>, i32)> = vec![];
        for _ in 0..100 {       
            let game = generate_random_game(&mut rng, nnue); 
            positions.extend(game);
        }

        let mut draws = positions.iter().filter(|(_, result)| result == &0).count();
        let mut wins = positions.len() - draws;

        while draws > (wins / 2) {
            let draw_index = positions.iter().position(|(_, result)| result == &0).expect("Could not find a draw.");
            positions.swap_remove(draw_index);

            draws -= 1;
        }

        /*while (2 * wins) > draws  {
            let win_index = positions.iter().position(|(_, result)| result != &0).expect("Could not find a win.");
            positions.swap_remove(win_index);

            wins -= 1;
        }*/

        for (features, score) in positions {
            let features = features.iter().map(|el| el.to_string()).collect::<Vec<_>>().join("");
            writeln!(file, "{features}: {:.1}", score).unwrap();
            position_count += 1;
        }
        let end = get_time_ms();
        println!("{iter}: {} ({} positions)", end - start, position_count);
        iter += 1;
    }
}