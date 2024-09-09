import re
from collections import OrderedDict, defaultdict
from typing import Dict, List, OrderedDict as OrderedDictType, Tuple

import pandas as pd

MLIRSectionsContainer = Dict[str, Dict[str, List[str]]]

def parse_mlir(mlir_code: str) -> MLIRSectionsContainer:
    lines = mlir_code.split("\n")
    sections = {}

    def _match_loc(line: str) -> None:
        _match = re.search(r"#loc(\d+) = loc\((.*)\)", line)
        if not _match:
            return
        loc_num = _match.group(1)
        file_info = _match.group(2)
        sections[loc_num] = {
            "file_info": (file_info.replace('"', "")),
            "code": [],
        }

    def _match_section(line: str) -> None:
        _match = re.search(r".*loc\(#loc(\d+)\)", line)
        if not _match:
            return

        loc_num = _match.group(1)

        if loc_num not in sections:
            return

        sections[loc_num]["code"].append(line.strip())

    for line in lines:
        _match_loc(line)

    # second pass to reference the code to the corresponding section
    for line in lines:
        _match_section(line)

    return sections


def collect_mlir_chunks(mlir_dump_file: str) -> OrderedDictType[str, Tuple[int, int]]:
    with open(mlir_dump_file, "r") as f:
        all_lines = f.readlines()

    mlir_chunks = []
    current_stage = None
    current_start_index = 0
    current_end_index = 0

    for index in range(len(all_lines)):
        line = all_lines[index]
        match = re.findall(r"IR Dump Before (\w+)", line)
        if not match and "IR Dump Before" in line:
            print(line)
            break
        if match:
            stage = match[0]
            if not current_stage or current_stage != stage:
                if current_stage:
                    mlir_chunks.append(
                        (
                            f"{current_stage}-{current_start_index}",
                            (current_start_index, current_end_index),
                        )
                    )
                current_stage = stage
                current_start_index = index + 1
                current_end_index = index + 1
        else:
            while re.search(r"#loc(\d+) = loc\((.*)\)", line):
                current_end_index = index + 1
                index += 1
                line = all_lines[index]

    return OrderedDict(mlir_chunks)


def parse_mlir_chunks(mlir_dump_file: str) -> List[Dict[str, str]]:
    mlir_chunks = collect_mlir_chunks(mlir_dump_file)
    with open(mlir_dump_file, "r") as f:
        all_lines = f.readlines()

    result_list = []
    for stage, (s, e) in mlir_chunks.items():
        joined_chunk = "".join(all_lines[s:e])
        result = defaultdict(list)
        for k, v in parse_mlir(joined_chunk).items():
            if v["file_info"] == "unknown":
                continue
            file_name, line_number, loc = v["file_info"].split(":")
            with open(file_name, "r") as f:
                code = f.readlines()[int(line_number) - 1]

            for instr in v["code"]:
                result[f"{code.strip()}"].append(instr)

        for code, instrs in result.items():
            result_list.append(
                {
                    "stage": stage,
                    "python code": code,
                    "instructions": "\n".join(instrs),
                }
            )

    return result_list

def parse_mlir_chunks_to_df(mlir_dump_file: str) -> pd.DataFrame:
    return pd.DataFrame(parse_mlir_chunks(mlir_dump_file))