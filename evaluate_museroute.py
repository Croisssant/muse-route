#!/usr/bin/env python3
"""
统计 MuseRoute 评测结果

用法:
    python evaluate_museroute.py --model <model_name>

输出:
    - 终端打印一行统计结果及缺失文件信息
    - 在 ./results/<model_name>/overall_results.txt 保存统计结果（两行）
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 定义组别及其对应的子路径
GROUPS = {
    "simple_easy": ["easy_semantic/simple", "easy_spatial/simple"],
    "simple_medium": ["medium/simple"],
    "simple_hard": ["hard/simple"],
    "complex_easy": ["easy_semantic/complex", "easy_spatial/complex"],
    "complex_medium": ["medium/complex"],
    "complex_hard": ["hard/complex"],
}

# 指标名称顺序
METRICS = ["svr", "scsr", "scar"]

# def get_ratio(dict_data):
#     values = list(dict_data.values())
#     total = len(values)
#     if total == 0:
#         return 0, 0
#     else:
#         true_count = sum(1 for v in values if v is True or v is None)
#         # ratio = true_count / total
#     return true_count, total

def get_binary(dict_data):
    values = list(dict_data.values())
    total = len(values)
    if total == 0:
        return 0, 1
    else:
        for v in values:
            if v is False:
                return 0, 1
    return 1, 1


# def is_svr_correct(svr_data: Any) -> bool:
#     """判断 SVR 是否正确：svr_data 必须是 None 或者一个字典，且字典内的所有值都为 True 或 None。"""
#     if svr_data is None:
#         return False
#     if not isinstance(svr_data, dict):
#         return False
#     for v in svr_data.values():
#         if v is not True and v is not None:
#             return False
#     return True


# def is_scsr_correct(scsr_data: Any) -> bool:
#     """判断 SCSR 是否正确：scsr_data 必须是 None 或者一个字典，且 start_end_location 为 True 或 None。"""
#     if scsr_data is None:
#         return False
#     if not isinstance(scsr_data, dict):
#         return False
#     val = scsr_data.get("start_end_location")
#     return val is True or val is None


# def is_scar_correct(scar_data: Any) -> bool:
#     """判断 SCAR 是否正确：scar_data 必须是 None 或者一个字典，且 attribute_validations 为 True 或 None。"""
#     if scar_data is None:
#         return True
#     if not isinstance(scar_data, dict):
#         return False
#     val = scar_data.get("attribute_validations")
#     return val is True or val is None


def process_json_file(file_path: Path) -> Tuple[bool, bool, bool]:
    """读取一个 final_results.json 文件，返回三个指标是否正确。文件不存在或解析出错返回 (False,False,False)。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, False, False

    # svr_ok = is_svr_correct(data.get("svr"))
    # scsr_ok = is_scsr_correct(data.get("scsr"))
    # scar_ok = is_scar_correct(data.get("scar"))
    # return svr_ok, scsr_ok, scar_ok

    # svr_cor, svr_tot = get_ratio(data.get("svr"))
    # scsr_cor, scsr_tot = get_ratio(data.get("scsr"))
    # scar_cor, scar_tot = get_ratio(data.get("scar"))
    # return svr_cor, svr_tot, scsr_cor, scsr_tot, scar_cor, scar_tot

    svr_cor, svr_tot = get_binary(data.get("svr"))
    scsr_cor, scsr_tot = get_binary(data.get("scsr"))
    scar_cor, scar_tot = get_binary(data.get("scar"))
    return svr_cor, 1, scsr_cor, 1, scar_cor, 1


def collect_stats(model_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], List[str]]:
    """
    遍历模型目录，收集统计信息。
    Returns:
        stats: 嵌套字典，stats[group][metric] = {"correct": int, "total": int}
        missing: 缺失的 final_results.json 路径列表（相对于模型目录）
    """
    stats = {}
    for group in GROUPS:
        stats[group] = {metric: {"correct": 0, "total": 0} for metric in METRICS}

    missing = []

    for group_name, subpaths in GROUPS.items():
        for subpath in subpaths:
            full_dir = model_dir / subpath
            if not full_dir.exists() or not full_dir.is_dir():
                continue

            for layout_dir in full_dir.iterdir():
                if not layout_dir.is_dir():
                    continue
                json_file = layout_dir / "final_results.json"
                if not json_file.exists():
                    missing.append(str(json_file.relative_to(model_dir)))
                    continue

                svr_cor, svr_tot, scsr_cor, scsr_tot, scar_cor, scar_tot = process_json_file(json_file)
                stats[group_name]["svr"]["total"] += svr_tot
                stats[group_name]["scsr"]["total"] += scsr_tot
                stats[group_name]["scar"]["total"] += scar_tot

                stats[group_name]["svr"]["correct"] += svr_cor
                stats[group_name]["scsr"]["correct"] += scsr_cor
                stats[group_name]["scar"]["correct"] += scar_cor

    return stats, missing


def format_rate(correct: int, total: int) -> str:
    """将正确数/总数格式化为三位小数字符串，分母为0时返回 "0.000" """
    if total == 0:
        return "0.00"
    print(f"{correct:5d} {total:5d} {correct * 100 / total:5.2f}")
    return f"{correct * 100 / total:.2f}"


def generate_header_line() -> str:
    """生成第一行的列名说明字符串，格式：Model & Simple_Easy_SVR & ..."""
    parts = ["Model"]
    group_order = [
        "simple_easy",
        "simple_medium",
        "simple_hard",
        "complex_easy",
        "complex_medium",
        "complex_hard",
    ]
    for group in group_order:
        # 将组名转换为更友好的标题（可选）
        for metric in METRICS:
            col_name = f"{group.upper()}_{metric.upper()}"
            parts.append(col_name)
    return " & ".join(parts)


def main():
    print(f"{'Corr':5} {'Total':5} {'Value':5}")
    parser = argparse.ArgumentParser(description="统计 MuseRoute 模型评测结果")
    parser.add_argument("--model", required=True, help="模型名称，对应 results/<model_name> 目录")
    args = parser.parse_args()

    base_dir = Path("./results")
    model_dir = base_dir / args.model

    if not model_dir.exists() or not model_dir.is_dir():
        print(f"错误：模型目录不存在或不是文件夹: {model_dir}")
        return

    stats, missing = collect_stats(model_dir)

    # 生成结果行（第二行）
    output_parts = [args.model]
    group_order = [
        "simple_easy",
        "simple_medium",
        "simple_hard",
        "complex_easy",
        "complex_medium",
        "complex_hard",
    ]
    for group in group_order:
        for metric in METRICS:
            correct = stats[group][metric]["correct"]
            total = stats[group][metric]["total"]
            rate_str = format_rate(correct, total)
            output_parts.append(rate_str)

    result_line = " & ".join(output_parts) + " \\\\"
    header_line = generate_header_line()

    # 写入文件
    output_file = model_dir / "overall_results.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header_line + "\n")
            f.write(result_line + "\n")
        print(f"汇总结果已保存至: {output_file}")
    except OSError as e:
        print(f"警告：无法写入汇总文件 {output_file}: {e}")

    # 终端输出
    print(result_line)

    # 输出缺失文件信息
    if missing:
        print(f"\n缺失文件总数: {len(missing)}")
        print("缺失路径:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()