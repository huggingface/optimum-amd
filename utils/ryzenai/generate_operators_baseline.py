# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os


def parse_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        result = {"all": 0, "dpu": 0, "cpu": 0, "matmulinteger": 0}
        for entry in data["deviceStat"]:
            result[entry["name"].lower()] = entry["nodeNum"]
        return result


def process_folder(input_folder):
    result_dict = {}
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            json_path = os.path.join(folder_path, "vitisai_ep_report.json")
            if os.path.exists(json_path):
                model_name = folder_name.lower()
                result_dict[model_name] = parse_json(json_path)
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Process JSON files in folders.")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_json", help="Path to the output JSON file")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_json = args.output_json

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    result_dict = process_folder(input_folder)
    sorted_result = dict(sorted(result_dict.items()))

    with open(output_json, "w") as output_file:
        json.dump(sorted_result, output_file, indent=2)

    print(f"Processed successfully. Result saved to '{output_json}'.")


if __name__ == "__main__":
    main()
