use monster_chess::board::{actions::{Move, HistoryMove}, Board};

pub const MAX_DEPTH: usize = 100;

pub struct PV<const T: usize> {
    pub length: [u32; MAX_DEPTH],
    pub table: [[Option<Move>; MAX_DEPTH]; MAX_DEPTH],
}

impl<const T: usize> PV<T> {
    pub fn init_pv(&mut self, ply: u32) {
        self.length[ply as usize] = ply;
    }

    pub fn update_pv(&mut self, ply: u32, best_move: Option<Move>) {
        self.table[ply as usize][ply as usize] = best_move;
        for next_ply in (ply + 1)..(self.length[(ply as usize) + 1]) {
            self.table[ply as usize][next_ply as usize] =
                self.table[(ply + 1) as usize][next_ply as usize];
        }
        self.length[ply as usize] = self.length[(ply + 1) as usize];
    }

    pub fn display_pv(&mut self, board: &mut Board<T>) -> String {
        let mut pv_actions: Vec<String> = Vec::with_capacity(self.table[0].len());
        let pv_table = self.table[0].clone();
        let mut undos: Vec<Option<HistoryMove<T>>> = Vec::with_capacity(self.table[0].len());
        for action in &pv_table {
            if action.is_none() {
                break;
            }
            if let Some(action) = action {
                pv_actions.push(board.encode_action(action));
                let undo = board.make_move(action);
                undos.push(undo);
            }
        }

        undos.reverse();

        for undo in undos {
            board.undo_move(undo);
        }

        pv_actions.join(" ")
    }
}