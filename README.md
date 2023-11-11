# procon-cexen

Libraries for competitive programming written by @cexen.
Primarily for [AtCoder](https://atcoder.jp/).

The codes are not fully tested and may contain bugs.
Use the information in this repository at your own risk.

## cpp

Target: Clang++23 on Windows.

### Additional info

[cpp/README.md](cpp/README.md)

## py

Target: PyPy3.9.
(Most of the 3.9 dependencies are GenericAlias, so most of the code should support PyPy3.6 if you make the necessary changes, such as rewriting `x = list[int]()` to `x: typing.List[int] = []`)

### Additional info

For `std::set` and `std::multiset` in Python,
I recommend [tatyam-prime/SortedSet](https://github.com/tatyam-prime/SortedSet).
