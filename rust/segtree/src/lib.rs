// https://github.com/cexen/procon-cexen/blob/main/rust/segtree/src/lib.rs
pub mod segtree {
    /// v1.4 @cexen
    pub struct SegtreePrimal<T, F: Fn(&T, &T) -> T, E: Fn() -> T> {
        n: usize,
        tree: Vec<T>,
        f: F,
        e: E,
    }
    impl<T, F: Fn(&T, &T) -> T, E: Fn() -> T> SegtreePrimal<T, F, E> {
        /// O(n).
        pub fn new(n: usize, f: F, e: E) -> Self {
            fn bit_width(mut n: usize) -> usize {
                let mut w = 0;
                while n > 0 {
                    w += 1;
                    n >>= 1;
                }
                w
            }
            let sz = 1 << (bit_width(n - 1) + 1);
            Self {
                n,
                tree: (0..sz).map(|_| e()).collect(),
                f,
                e,
            }
        }
        /// O(n).
        pub fn from_vec(v: Vec<T>, f: F, e: E) -> Self {
            let mut _self = Self::new(v.len(), f, e);
            let idx_leaf = _self._idx_leaf();
            for (i, x) in v.into_iter().enumerate() {
                _self.tree[idx_leaf + i] = x;
            }
            _self._build();
            _self
        }
        /// O(n).
        fn _build(&mut self) {
            for i in (0..self._idx_leaf()).rev() {
                self.tree[i] = (self.f)(&self.tree[i << 1], &self.tree[(i << 1) + 1]);
            }
        }
        /// O(1).
        pub fn len(&self) -> usize {
            self.n
        }
        /// O(1).
        fn _idx_leaf(&self) -> usize {
            self.tree.len() / 2
        }
        /// O(1).
        pub fn get(&self, i: usize) -> &T {
            assert!(i < self.n);
            &self.tree[i + self._idx_leaf()]
        }
        /// O(log n).
        pub fn set(&mut self, mut i: usize, v: T) {
            assert!(i <= self.n);
            i += self._idx_leaf();
            self.tree[i] = v;
            i >>= 1;
            while i > 0 {
                self.tree[i] = (self.f)(&self.tree[i << 1], &self.tree[(i << 1) + 1]);
                i >>= 1;
            }
        }
        /// O(log n).
        pub fn grasp(&self, mut l: usize, mut r: usize) -> T {
            assert!(l <= r && r <= self.n);
            l += self._idx_leaf();
            r += self._idx_leaf();
            let mut vl = vec![];
            let mut vr = vec![];
            while l < r {
                if l & 1 > 0 {
                    vl.push(&self.tree[l]);
                    l += 1;
                }
                if r & 1 > 0 {
                    vr.push(&self.tree[r - 1]);
                    r -= 1;
                }
                l >>= 1;
                r >>= 1;
            }
            let mut ans = (self.e)();
            for &v in vl.iter() {
                ans = (self.f)(&ans, &v);
            }
            for &v in vr.iter().rev() {
                ans = (self.f)(&ans, &v);
            }
            ans
        }
        pub fn max_right<G: Fn(&T) -> bool>(&self, l: usize, g: G) -> usize {
            assert!(l <= self.n);
            let mut r = l + self._idx_leaf();
            let mut s = (self.e)();
            loop {
                r /= r & (!r + 1);
                let ns = (self.f)(&s, &self.tree[r]);
                if !g(&ns) {
                    break;
                }
                let nr = (r >> 1) + 1;
                if (nr & (!nr + 1)) == nr {
                    return self.n;
                }
                s = ns;
                r = nr;
            }
            while r < self._idx_leaf() {
                r <<= 1;
                let ns = (self.f)(&s, &self.tree[r]);
                if g(&ns) {
                    r += 1;
                    s = ns;
                }
            }
            (r - self._idx_leaf()).min(self.n)
        }
        pub fn partition_point<G: Fn(&T) -> bool>(&self, g: G) -> usize {
            self.max_right(0, g)
        }
    }
    impl<T, F: Fn(&T, &T) -> T, E: Fn() -> T> std::ops::Index<usize> for SegtreePrimal<T, F, E> {
        type Output = T;
        fn index(&self, index: usize) -> &Self::Output {
            self.get(index)
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_segtree_primal() {
            let f = |a: &_, b: &_| a + b;
            let e = || 0;
            let mut segs = vec![
                SegtreePrimal::new(6, f, e),
                SegtreePrimal::from_vec(vec![0, 0, 10, 1, 40, 0], f, e),
            ];
            segs[0].set(2, 10);
            segs[0].set(3, 1);
            segs[0].set(4, 40);
            for seg in segs {
                assert_eq!(seg.len(), 6);
                assert_eq!(seg[0], 0);
                assert_eq!(seg[2], 10);
                assert_eq!(seg.get(2), &10);
                assert_eq!(seg[3], 1);
                assert_eq!(seg[5], 0);
                assert_eq!(seg.grasp(0, seg.n), 51);
                assert_eq!(seg.grasp(0, 0), 0);
                assert_eq!(seg.grasp(0, 1), 0);
                assert_eq!(seg.grasp(0, 2), 0);
                assert_eq!(seg.grasp(0, 3), 10);
                assert_eq!(seg.grasp(1, 3), 10);
                assert_eq!(seg.grasp(2, 3), 10);
                assert_eq!(seg.grasp(3, 3), 0);
                assert_eq!(seg.grasp(2, 4), 11);
                assert_eq!(seg.grasp(2, 5), 51);
                assert_eq!(seg.grasp(2, 6), 51);
                assert_eq!(seg.grasp(3, 4), 1);
                assert_eq!(seg.grasp(3, 5), 41);
                assert_eq!(seg.grasp(4, 5), 40);
                assert_eq!(seg.grasp(4, 6), 40);
                assert_eq!(seg.partition_point(|&s| s < 0), 0);
                assert_eq!(seg.partition_point(|&s| s <= 0), 2);
                assert_eq!(seg.partition_point(|&s| s < 11), 3);
                assert_eq!(seg.partition_point(|&s| s <= 11), 4);
                assert_eq!(seg.partition_point(|&s| s < 51), 4);
                assert_eq!(seg.partition_point(|&s| s <= 51), 6);
                assert_eq!(seg.max_right(1, |&s| s < 0), 1);
                assert_eq!(seg.max_right(1, |&s| s <= 0), 2);
                assert_eq!(seg.max_right(2, |&s| s < 0), 2);
                assert_eq!(seg.max_right(2, |&s| s < 10), 2);
                assert_eq!(seg.max_right(2, |&s| s < 11), 3);
                assert_eq!(seg.max_right(3, |&s| s < 1), 3);
                assert_eq!(seg.max_right(3, |&s| s <= 1), 4);
                assert_eq!(seg.max_right(3, |&s| s < 41), 4);
                assert_eq!(seg.max_right(3, |&s| s <= 41), 6);
                assert_eq!(seg.max_right(5, |&s| s < 0), 5);
                assert_eq!(seg.max_right(5, |&s| s < 100), 6);
                assert_eq!(seg.max_right(6, |_| false), 6);
                assert_eq!(seg.max_right(6, |_| true), 6);
            }
        }
    }
}
#[allow(unused_imports)]
use segtree::SegtreePrimal;
