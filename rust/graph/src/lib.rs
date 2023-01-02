// https://github.com/cexen/procon-cexen/blob/main/rust/graph/src/lib.rs
pub mod graph {
    /// v1.2 @cexen
    #[derive(Default)]
    pub struct Graph {
        n: usize,
        adjs: Vec<Vec<usize>>,
        revs: Vec<Vec<usize>>,
        idxs: Option<Vec<usize>>,
    }
    impl Graph {
        /// O(n).
        pub fn new(n: usize) -> Self {
            Self {
                n,
                adjs: vec![vec![]; n],
                revs: vec![vec![]; n],
                ..Default::default()
            }
        }
        pub fn len(&self) -> usize {
            self.n
        }
        pub fn adjs<'a>(&'a self) -> &'a Vec<Vec<usize>> {
            &self.adjs
        }
        pub fn revs<'a>(&'a self) -> &'a Vec<Vec<usize>> {
            &self.revs
        }
        /// O(n).
        pub fn connect(&mut self, i: usize, j: usize) {
            assert!(i < self.n);
            assert!(j < self.n);
            self.adjs[i].push(j);
            self.revs[j].push(i);
        }
        pub fn sort<'a>(&'a mut self) -> &'a Vec<usize> {
            let mut idxs = vec![0usize; 0];
            let mut visited = vec![false; self.n];
            for i in 0..self.n {
                self._dfs(i, &mut idxs, &mut visited);
            }
            idxs.reverse();
            self.idxs = Some(idxs);
            &self.idxs.as_ref().unwrap()
        }
        fn _dfs(&self, i: usize, idxs: &mut Vec<usize>, visited: &mut Vec<bool>) {
            if visited[i] {
                return;
            }
            visited[i] = true;
            for &j in &self.adjs[i] {
                self._dfs(j, idxs, visited);
            }
            idxs.push(i);
        }
        pub fn find_sccs(&mut self) -> Vec<Vec<usize>> {
            if self.idxs.is_none() {
                self.sort();
            }
            let mut sccs = vec![vec![]; 0];
            let mut visited = vec![false; self.n];
            for &i0 in self.idxs.as_ref().unwrap() {
                if visited[i0] {
                    continue;
                }
                visited[i0] = true;
                let mut scc = vec![];
                let mut q = vec![i0];
                while let Some(i) = q.pop() {
                    scc.push(i);
                    for &j in &self.revs[i] {
                        if visited[j] {
                            continue;
                        }
                        visited[j] = true;
                        q.push(j);
                    }
                }
                sccs.push(scc);
            }
            sccs
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_sort() {
            let mut g = Graph::new(4);
            g.connect(0, 2);
            g.connect(2, 1);
            g.connect(1, 3);
            assert_eq!(g.sort(), &[0, 2, 1, 3]);
        }
        #[test]
        fn test_find_sccs() {
            let mut g = Graph::new(4);
            g.connect(0, 2);
            g.connect(2, 0);
            g.connect(0, 1);
            g.connect(1, 3);
            assert_eq!(g.find_sccs(), vec![vec![0, 2], vec![1], vec![3]]);
        }
    }
}
#[allow(unused_imports)]
use graph::Graph;
