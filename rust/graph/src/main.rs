use lib::graph::Graph;

// --------------------

use proconio::input;

/// AtCoder ACL Practice Contest G - SCC
/// https://atcoder.jp/contests/practice2/tasks/practice2_g
#[allow(non_snake_case, dead_code)]
fn solve_atcoder_practice2_g() {
    input! {
        N: usize,
        M: usize,
        AB: [(usize, usize); M],
    }
    let mut g = Graph::new(N);
    for (a, b) in AB {
        g.connect(a, b);
    }
    let sccs = g.find_sccs();
    println!("{}", sccs.len());
    for scc in sccs {
        print!("{}", scc.len());
        for i in scc {
            print!(" {}", i);
        }
        println!();
    }
}

fn main() {
    // solve_atcoder_practice2_g();
}
