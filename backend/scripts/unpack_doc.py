#!/usr/bin/env python
"""解压官方文档语料 ``backend/doc/ob_wiki.zip`` → ``backend/doc/ob_wiki/``。

为什么需要它：

1. **CI / 新环境**：``backend/doc/ob_wiki/`` 是解压产物（45 MB、5146 篇）已 gitignore，
   仓库里只跟踪 zip。没有语料，检索索引建不出来，检索评测与文档工具的用例全都跑不了。
2. **编码**：这个 zip 未置 UTF-8 标志位（flag_bits=0x0），zipfile 会按 CP437 解码文件名，
   直接解压会得到 ``OceanBase µ£ÇΣ╜│σ«₧Φ╖╡/`` 这类乱码目录。这里用
   ``name.encode('cp437').decode('utf-8')`` 复原。
3. **可复现**：文件 mtime 一律取自 zip 记录（不是解压时刻），因此索引指纹
   （文件数 + 字节数 + 最大 mtime）在不同机器上一致 —— 否则每次 CI 都会重建索引，
   缓存也就没意义了。

用法::

    python scripts/unpack_doc.py              # 缺什么解什么（已有且完整则跳过）
    python scripts/unpack_doc.py --force      # 先清空目标目录再解压
"""
from __future__ import annotations

import argparse
import calendar
import datetime
import os
import shutil
import sys
import zipfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = _BACKEND / "doc" / "ob_wiki.zip"
DEFAULT_DEST = _BACKEND / "doc" / "ob_wiki"

# zip 里混进的打包垃圾：不参与语料，也不该出现在 grep/检索结果里
_SKIP_PARTS = {"__MACOSX"}
_SKIP_NAMES = {".DS_Store"}

# 时间戳下限：zip 的 date_time 不合法（如 1980-01-01）时用不到，仅作兜底
_EPOCH = 315532800  # 1980-01-01T00:00:00Z


def iter_members(zip_path: Path):
    """产出 (zip_info, 解码后的相对路径)；打包垃圾与目录条目被过滤掉。"""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.flag_bits & 0x800 == 0:
                # 未置 UTF-8 标志：先按 CP437 还原字节再按 UTF-8 解码（见模块 docstring）
                try:
                    name = name.encode("cp437").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if not parts or any(p in _SKIP_PARTS for p in parts):
                continue
            if parts[-1] in _SKIP_NAMES or parts[-1].startswith("._"):
                continue
            if info.is_dir():
                continue
            yield info, Path(*parts)


def unpack(zip_path: Path, dest: Path, *, force: bool = False) -> int:
    """解压并返回写出的文件数。

    压缩包里的成员都带 ``ob_wiki/`` 前缀，这里统一剥掉：调用方给的 dest 就是语料根，
    否则会解出 ``doc/ob_wiki/ob_wiki/…`` 这种套娃目录。
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到语料压缩包: {zip_path}")
    members = list(iter_members(zip_path))
    if members:
        prefix = members[0][1].parts[0]
        if all(rel.parts[0] == prefix for _, rel in members):
            members = [(info, Path(*rel.parts[1:])) for info, rel in members if len(rel.parts) > 1]
    if force and dest.exists():
        shutil.rmtree(dest)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info, relative in members:
            target = dest / relative
            if target.exists() and target.stat().st_size == info.file_size and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            # mtime 取自 zip 记录：索引指纹才是「语料版本」的函数，而不是解压时刻
            try:
                epoch = calendar.timegm(datetime.datetime(*info.date_time[:6]).timetuple())
                stamp = max(epoch, _EPOCH)
                os.utime(target, (stamp, stamp))
            except (ValueError, OverflowError):
                pass
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="解压 ob_wiki 官方文档语料（含文件名编码修复）")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help=f"语料压缩包（默认 {DEFAULT_ZIP}）")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help=f"解压目录（默认 {DEFAULT_DEST}）")
    parser.add_argument("--force", action="store_true", help="先删空目标目录再完整解压")
    args = parser.parse_args(argv)

    zip_path, dest = Path(args.zip), Path(args.dest)
    written = unpack(zip_path, dest, force=args.force)
    total = sum(1 for _ in dest.rglob("*.md"))
    print(f"解压完成: 新写 {written} 个文件, 现在 {dest} 下共 {total} 个 .md")
    if total == 0:
        print("语料为空，后续检索用例无法运行", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())