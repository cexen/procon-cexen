# https://github.com/cexen/procon-cexen/blob/main/py/recurse.py
import sys

sys.setrecursionlimit(10**9)

try:
    # https://qiita.com/shoji9x9/items/e7d19bd6f54e960f46be#2022226%E8%BF%BD%E8%A8%98
    import pypyjit  # type: ignore

    # =1 seems best?
    # https://atcoder.jp/contests/abc165/submissions?f.Task=abc165_f&f.LanguageName=Python3&f.User=cexen
    pypyjit.set_param("max_unroll_recursion=1")
except ImportError:
    pass
