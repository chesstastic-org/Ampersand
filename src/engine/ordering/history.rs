use monster_chess::board::actions::Move;

#[derive(Copy, Clone, Debug)]
pub struct HistoryInfo {
    pub inc: u32,
    pub counter_move: Option<Move>
}