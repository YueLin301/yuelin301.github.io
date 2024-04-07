---
title: Python Project Template
date: 2024-03-31 14:00:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code Utils, Toolbox, Template]
math: True
---


## How to Use

1. Download [`LyPythonProjectTemplate`](https://github.com/YueLin301/LyPythonProjectTemplate/archive/refs/heads/main.zip)
2. Decompress it.
3. Create a new Github project. See [my blog]({{site.baseurl}}/posts/Github-Memo/#create-a-repo).
4. Copy the contents of `LyPythonProjectTemplate` into the root folder of your new project.
5. Commit it. See [my blog]({{site.baseurl}}/posts/Github-Memo/#lazy-commit).

## File Structure

```bash
brew install tree
```

```bash
tree -a -I '.DS_Store|.git' .
```

- `-a` means all, including hidden files.
- `-I` means ignore. Use `|` to separate multiple file names.

```
.
├── .gitignore
├── .vscode
│   └── launch.json
├── LICENSE
├── README.md
├── bin
│   └── print_haha.sh
├── demo
│   └── print_haha.py
├── docs
│   └── haha.md
├── draft
│   └── draft_0x00.md
├── requirements.txt
├── src
│   ├── project_name
│   │   └── __init__.py
│   └── utils
│       ├── Util_import.py
│       └── __init__.py
└── tests
    └── mytest_0x00.py

10 directories, 13 files
```

## License

To use a license, all you need to do is to create a file named `LICENSE` in the root directory of your project, and then copy the contents of the licence you wanted into it.

### WTFPL

WTFPL = Do <u>W</u>hat <u>T</u>he <u>F</u>uck You Want To <u>P</u>ublic <u>L</u>icense.

No limitations at all.

```
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004
 
Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.
 
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
```

### MIT License

They who use your work should copy this file into their projects.

1. This will let others know that the new projects have incorporated your open-source work.
2. No other limitations.

```
MIT License

Copyright (c) [year] Yue Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Apache License 2.0

下面是一个表格，对比了Apache License 2.0和MIT License的关键特性和要求：

| 特性/要求           | Apache License 2.0                                                  | MIT License                          |
|-------------------|--------------------------------------------------------------------|--------------------------------------|
| **版权声明**         | 必须在每个副本中包含版权声明。                                          | 必须在每个副本中包含版权声明。            |
| **许可证文本**       | 必须在每个副本中包含完整的许可证文本。                                    | 必须在每个副本中包含完整的许可证文本。      |
| **变更记录**         | 如果修改了代码，必须在一个通知文件中提供修改的描述。                       | 无此要求。                             |
| **专利授权**        | 明确授予专利使用权，保护贡献者和使用者免受专利诉讼。                        | 无明确的专利授权条款。                    |
| **商标使用**         | 禁止使用贡献者的商标。                                                 | 无明确条款。                           |
| **分发要求**         | 分发作品时，必须包含许可证副本和变更记录（如果有修改）。                      | 分发作品时，必须包含许可证副本。             |
| **无担保声明**       | 包含详细的无担保声明，明确指出软件“按原样”提供，不承担任何担保责任。               | 简短的无担保声明，通常一句话说明软件不提供担保。 |
| **软件使用和再分发** | 允许广泛使用、修改、再分发，包括商业用途。                                 | 允许广泛使用、修改、再分发，包括商业用途。      |
| **适用场景**         | 适用于需要明确专利授权和变更记录的项目。                                   | 适用于寻求简洁许可条款的项目。               |

Apache 相比 MIT 的多出来的限制就是：
1. 专利授权：如果是Apache，如果有专利，使用者会被自动授权专利的使用权，可以用没关系；
2. 禁止使用贡献者的商标；
3. 做了哪些更改要标明