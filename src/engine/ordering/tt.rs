use monster_chess::board::actions::Move;

#[derive(Clone, Copy, Debug)]
pub struct TranspositionEntry {
    pub depth: u32,
    pub eval: i32,
    pub best_move: Move
}