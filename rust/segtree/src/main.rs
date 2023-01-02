use lib::segtree::SegtreePrimal;

// --------------------

use proconio::input;

/// AtCoder ACL Practice Contest J - Segment Tree
/// https://atcoder.jp/contests/practice2/tasks/practice2_j
#[allow(non_snake_case, dead_code)]
fn solve_atcoder_practice2_j() {
    input! {
        N: usize,
        Q: usize,
        A: [usize; N],
        qs: [(i8, usize, usize); Q],
    }
    let mut seg = SegtreePrimal::from_vec(A, |a, b| *a.max(b), || 0);
    for (i, mut u, mut v) in qs {
        match i {
            1 => {
                u -= 1;
                seg.set(u, v);
            }
            2 => {
                u -= 1;
                v -= 1;
                println!("{}", seg.grasp(u, v + 1));
            }
            3 => {
                u -= 1;
                println!("{}", 1 + seg.max_right(u, |&m| m < v));
            }
            _ => {
                unreachable!();
            }
        };
    }
}

/// AtCoder ABC231 F - Jealous Two
/// https://atcoder.jp/contests/abc231/tasks/abc231_f
#[allow(non_snake_case, dead_code)]
fn solve_atcoder_abc231_f() {
    use std::collections::HashMap;
    input! {
        N: usize,
        mut A: [i32; N],
        mut B: [i32; N],
    }
    let mut AB = HashMap::new();
    for (&a, &b) in A.iter().zip(&B) {
        AB.entry(a).or_insert(vec![]).push(b);
    }
    A.sort();
    A.dedup();
    B.sort();
    B.dedup();
    let mut K = HashMap::new();
    for (i, &v) in B.iter().enumerate() {
        K.insert(v, i);
    }
    let mut seg = SegtreePrimal::new(K.len(), |a, b| a + b, || 0i64);
    let mut ans = 0;
    for &a in &A {
        let bs = &AB[&a];
        for &b in bs {
            let k = K[&b];
            seg.set(k, seg[k] + 1);
        }
        for &b in bs {
            let k = K[&b];
            ans += seg.grasp(k, seg.len());
        }
    }
    println!("{}", ans);
}

fn main() {
    // solve_atcoder_practice2_j();
    // solve_atcoder_abc231_f();
}
