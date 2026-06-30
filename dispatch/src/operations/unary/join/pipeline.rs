//
//
// struct Pipeline<'a, F: FnMut(), T> {
//     func: F,
//     values: &'a [T]
// }
//
// impl<'a, F, T> Pipeline<'a, F, T> {
//
//     pub fn advance(&mut self) {
//         self.func()
//     }
// }
