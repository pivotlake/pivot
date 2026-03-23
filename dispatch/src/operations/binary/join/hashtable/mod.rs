// use crate::memory::MultiSlabBuffer;
//
pub struct Directory {
    entries: Vec<u64>
}

impl Directory {

    #[inline(always)]
    fn matches_bloom(&self, key: u64) -> bool {
        true
    }
}
//
//
// pub struct HashTable<K, T> {
//     directory: Directory,
//     values: MultiSlabBuffer<(K, T)>,
// }
//
// impl<K, T> HashTable<K, T> {
//
//     fn get(&self, hash: u64, key: K) -> impl Iterator<Item=T> {
//
//         todo!()
//     }
// }

mod builder;