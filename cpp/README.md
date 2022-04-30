# procon-cexen/cpp

## building boost 1.72.0 with clang on Windows

Fetch `boost_1_72_0.7z` from [boost.org](https://www.boost.org/users/history/version_1_72_0.html).
Extract into `~/Downloads/boost_1_72_0`.

Install MSVC (C++ Build Tools). Available via Visual Studio Installer or directly from microsoft website.

Install llvm. Available from [Chocolatey](https://community.chocolatey.org/packages/llvm) or [GitHub](https://github.com/llvm/llvm-project/releases).

Open a Developer PowerShell of MSVC.

```pwsh
cd ~/Downloads/boost_1_72_0
./bootstrap
$threads = 1 + (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors  # OR SPECIFY WHAT YOU LIKE
./b2 stage toolset=clang-win toolset=clang-win link=static threading=single variant=release runtime-link=static -j"$threads"
```
