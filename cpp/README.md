# procon-cexen/cpp

## building boost 1.83.0 with clang on Windows

AtCoder's 2023/1 language update adopted boost 1.82.0, but the bootstrap included with that version does not work properly. Therefore, I recommend using 1.83.0 for local builds instead.

Fetch `boost_1_83_0.7z` from [boost.org](https://www.boost.org/users/history/version_1_83_0.html).
Extract into `~/Downloads/boost_1_83_0`.

Install MSVC (C++ Build Tools). Available via Visual Studio Installer or directly from microsoft website.

Install llvm. Available from [Chocolatey](https://community.chocolatey.org/packages/llvm) or [GitHub](https://github.com/llvm/llvm-project/releases).

Open a Developer PowerShell of MSVC.

```pwsh
cd ~/Downloads/boost_1_83_0
./bootstrap vc143
$num_threads = 1 + (Get-CimInstance -ClassName Win32_Processor).NumberOfLogicalProcessors  # OR SPECIFY WHAT YOU LIKE
./b2 stage toolset=clang-win link=static runtime-link=static threading=single variant=release address-model=64 -j"$num_threads"
```
