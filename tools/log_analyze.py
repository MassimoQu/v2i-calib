import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / 'outputs' / 'logs'
LEGACY_ROOT = ROOT / 'legacy'


def filter_records(input_log_path, output_log_path, re_threshold, te_threshold):
    with open(input_log_path, 'r') as infile, open(output_log_path, 'w') as outfile:
        for line in infile:
            # 使用正则表达式提取 RE 和 TE 的值
            re_match = re.search(r'RE: (\d+\.\d+)', line)
            te_match = re.search(r'TE: (\d+\.\d+)', line)
            
            if re_match and te_match:
                re_value = float(re_match.group(1))
                te_value = float(te_match.group(1))
                
                # 检查 RE 和 TE 是否都小于给定的阈值
                if re_value < re_threshold and te_value < te_threshold:
                    outfile.write(line)  # 将符合条件的记录写入新的日志文件

def analyze_log(file_path, success_rate_threshold, using_stability=False, filter_threshold=5, sp_success_filter=False):
    re_values = []
    te_values = []
    time_values = []
    success_rate = {x: 0 for x in success_rate_threshold}

    selected_pairs_list = None

    if sp_success_filter:
        sel_path = LOG_ROOT / f"selected_pairs_co_greater_{2}.json"
        with sel_path.open('r') as f:
            filter_threshold = success_rate_threshold[0]
            selected_pairs_list = json.load(f)
            selected_pairs_list1 = [tuple(pair) for pair in selected_pairs_list]  # 转换为元组列表
        with (LEGACY_ROOT / 'selected_pairs_olddatarange.json').open('r') as f:
            filter_threshold = success_rate_threshold[0]
            selected_pairs_list = json.load(f)
            selected_pairs_list2 = [tuple(pair) for pair in selected_pairs_list]  # 转换为元组列表
        
        selected_pairs_list = list(set(selected_pairs_list1) & set(selected_pairs_list2))  # 合并两个列表并转换为集合以去重

    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式找到 RE 和 TE 的值
            re_match = re.search(r'RE: (\d+\.\d+)', line)
            te_match = re.search(r'TE: (\d+\.\d+)', line)
            time_match = re.search(r'time: (\d+\.\d+)', line)
            stability = re.search(r'stability: (\d+\.\d+)', line)

            if re_match and te_match and time_match:
                re_value = float(re_match.group(1))
                te_value = float(te_match.group(1))
                time_value = float(time_match.group(1))
                stability_value = float(stability.group(1)) if stability else -1
                
                if sp_success_filter and selected_pairs_list is not None:
                    # 使用正则表达式提取 inf_id 和 veh_id
                    inf_id_match = re.search(r'inf_id: (\d+)', line)
                    veh_id_match = re.search(r'veh_id: (\d+)', line)
                    
                    if inf_id_match and veh_id_match:
                        inf_id = inf_id_match.group(1)
                        veh_id = veh_id_match.group(1)
                        if (inf_id, veh_id) not in selected_pairs_list:
                            continue
                if (using_stability and stability_value < 10) or re_value > filter_threshold or te_value > filter_threshold:
                    continue
                re_values.append(re_value)
                te_values.append(te_value)
                time_values.append(time_value)
                
                for x in success_rate_threshold:
                    # 计算成功率@x
                    if te_value < x:
                        success_rate[x] += 1

    # 计算平均值
    avg_re = sum(re_values) / len(re_values) if re_values else 0
    avg_te = sum(te_values) / len(te_values) if te_values else 0
    avg_time = sum(time_values) / len(time_values) if time_values else 0

    for x in success_rate_threshold:
        success_rate[x] = success_rate[x] / len(selected_pairs_list) if selected_pairs_list else len(re_values)

    return avg_re, avg_te, success_rate, avg_time, len(selected_pairs_list) if selected_pairs_list else len(re_values)


# prefix = 'DAIR-V2X_10'
# prefix = 'DAIR-V2X_demo'
def analyze_with_same_prefix_data(prefix = 'DAIR-V2X_10'):
    '''
    分析所有具有相同前缀的日志文件，首先筛选 RE 和 TE 小于阈值的记录，然后计算平均值和成功率
    '''
    # 获取所有日志文件的路径
    log_directory = LOG_ROOT / 'old_data'
    
    log_files = [f for f in os.listdir(log_directory) if f.startswith(prefix) and f.endswith('.log')]

    # 初始化数据列表
    data = []

    # 分析每个日志文件
    for log_file in log_files:
        log_path = os.path.join(log_directory, log_file)
        success_rate_threshold = [1, 2, 3]
        average_re, average_te, success_rate, average_time, absolute_num = analyze_log(log_path, success_rate_threshold)
        
        # 解析特征名称（方法名称）
        # feature_name = log_file.split('_')[-3:-1]  # 假设特征名称位于第六个位置
        feature_name = log_file
        data_dict = {'Feature': feature_name}
        # 存储结果
        for x in success_rate_threshold:
            average_re, average_te, success_rate, average_time, absolute_num = analyze_log(log_path, [x], sp_success_filter=True)
            data_dict[f'mRRE@{x}'] = average_re
            data_dict[f'mRTE@{x}'] = average_te
            data_dict[f'Success Rate @{x}'] = success_rate[x] * 100
        data_dict['Average Time'] = average_time
        data_dict['Absolute Num'] = absolute_num
        data.append(data_dict)

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 打印或保存 DataFrame
    # print(df)
    df.to_csv('analysis_results.csv', index=False)

def analyze_with_same_folder_data(log_directory = LOG_ROOT):
    '''
    分析所有具有相同前缀的日志文件，首先筛选 RE 和 TE 小于阈值的记录，然后计算平均值和成功率
    '''
    
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]

    # 初始化数据列表
    data = []

    # 分析每个日志文件
    for log_file in log_files:
        log_path = os.path.join(log_directory, log_file)
        success_rate_threshold = [1, 2]
        average_re, average_te, success_rate, average_time, absolute_num = analyze_log(log_path, success_rate_threshold, filter_threshold=10.0)
        
        # 解析特征名称（方法名称）
        # feature_name = log_file.split('_')[-3:-1]  # 假设特征名称位于第六个位置
        feature_name = log_file
        
        # 存储结果
        data.append({
            'Feature': feature_name,
            'Average RE': average_re,
            'Average TE': average_te,
            'Success Rate @1': success_rate[1] * 100,
            'Success Rate @2': success_rate[2] * 100,
            'Average Time': average_time,
            'Absolute Num': absolute_num
        })

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 打印或保存 DataFrame
    # print(df)
    df.to_csv('analysis_results.csv', index=False)


def analyze_with_same_data_range(group_name='easy', filter_threshold=10.0, success_rate_threshold=[1, 2]):
    '''
    Reads all log files in a specified directory, extracts common data parts, ensuring consistency in data ranges for analysis.
    '''
    log_directory = LOG_ROOT / 'group' / group_name
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]
    filtered_pairs = defaultdict(list)

    num = len(log_files)
    num_dict = {}

    # print(num)
    # Read and filter data based on RE and TE
    for log_file in log_files:
        log_path = os.path.join(log_directory, log_file)

        with open(log_path, 'r') as file:
            for line in file:
                data = extract_data_from_line(line)
                if data == None:
                    continue
                if data['RE'] <= filter_threshold and data['TE'] <= filter_threshold:
                    pair_key = (data['inf_id'], data['veh_id'])
                    filtered_pairs[pair_key].append(data)


    # Calculate averages and success rates
    results = {}
    for log_file in log_files:
        results[log_file] = {}
        results[log_file]['RE'] = 0
        results[log_file]['TE'] = 0
        results[log_file]['time'] = 0
        results[log_file]['Success_rate'] = {}
        for threshold in success_rate_threshold:
            results[log_file]['Success_rate'][threshold] = 0
        
    valid_pairs_num = 0

    for _, entries in filtered_pairs.items():
        num_dict[len(entries)] = num_dict.get(len(entries), 0) + 1
        if len(entries) < num:
            continue
        valid_pairs_num += 1
        for j in range(num):
            results[log_files[j]]['RE'] += entries[j]['RE']
            results[log_files[j]]['TE'] += entries[j]['TE']
            results[log_files[j]]['time'] += entries[j]['time']
            
            for threshold in success_rate_threshold:
                if entries[j]['TE'] < threshold and entries[j]['RE'] < threshold:
                    results[log_files[j]]['Success_rate'][threshold] += 1
        
    for j in range(num):
        if valid_pairs_num > 0:
            results[log_files[j]]['RE'] /= valid_pairs_num
            results[log_files[j]]['TE'] /= valid_pairs_num
            results[log_files[j]]['time'] /= valid_pairs_num
            for threshold in success_rate_threshold:
                results[log_files[j]]['Success_rate'][threshold] /= valid_pairs_num

    return results, valid_pairs_num, num_dict

def extract_data_from_line(line):
    '''
    Extracts data from a single line of log file.
    '''
    # pattern = r'inf_id: (\d+),?\s*veh_id: (\d+),?\s*RE: ([\d.]+),?\s*TE: ([\d.]+),?\s*time: ([\d.]+)'
    pattern = r'inf_id: (\d+), veh_id: (\d+),?\s*RE: ([\d.]+),\s*TE: ([\d.]+)(?:,.*?\s*time: ([\d.]+))?'
    match = re.search(pattern, line)
    if match:
        return {
            'inf_id': match.group(1),
            'veh_id': match.group(2),
            'RE': float(match.group(3)),
            'TE': float(match.group(4)),
            'time': float(match.group(5))
        }
    else:
        return None

def load_selected_pairs_from_json(path=LOG_ROOT / "selected_pairs.json"):
    print(f"Loading pairs from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        loaded_pairs = json.load(f)
    print(f"Loaded {len(loaded_pairs)} pairs.")
    # 注意：JSON没有“元组”类型，所以读取后内部的元组会变成列表。
    # 例如 [('a', 'b')] 会被读取为 [['a', 'b']]
    # 如果必须是元组，可以进行转换：
    loaded_pairs_as_tuples = [tuple(pair) for pair in loaded_pairs]
    return loaded_pairs_as_tuples


def analyze_valid_line_list(valid_line_list, success_rate_threshold, using_stability=False, filter_threshold=10):
    re_values = []
    te_values = []
    time_values = []
    success_rate = {x: 0 for x in success_rate_threshold}

    for line in valid_line_list:
        # 使用正则表达式找到 RE 和 TE 的值
        re_match = re.search(r'RE: (\d+\.\d+)', line)
        te_match = re.search(r'TE: (\d+\.\d+)', line)
        time_match = re.search(r'time: (\d+\.\d+)', line)
        stability = re.search(r'stability: (\d+\.\d+)', line)

        if re_match and te_match:
            re_value = float(re_match.group(1))
            te_value = float(te_match.group(1))
            time_value = float(time_match.group(1))
            stability_value = float(stability.group(1)) if stability else 11
            if (using_stability and stability_value < 10) or re_value > filter_threshold or te_value > filter_threshold:
                continue
            re_values.append(re_value)
            te_values.append(te_value)
            time_values.append(time_value)
            
            for x in success_rate_threshold:
                # 计算成功率@x
                if te_value < x:
                    success_rate[x] += 1

    # 计算平均值
    avg_re = sum(re_values) / len(re_values) if re_values else 0
    avg_te = sum(te_values) / len(te_values) if te_values else 0
    avg_time = sum(time_values) / len(time_values) if time_values else 0

    for x in success_rate_threshold:
        success_rate[x] = success_rate[x] / len(re_values) if re_values else 0

    return avg_re, avg_te, success_rate, avg_time, len(re_values)



def analyze_filter_pairs_with_same_prefix_data(prefix = 'DAIR-V2X_10'):
    '''
    分析所有具有相同前缀的日志文件，首先筛选 RE 和 TE 小于阈值的记录，然后计算平均值和成功率
    '''
    # 获取所有日志文件的路径
    log_directory = LOG_ROOT
    
    log_files = [f for f in os.listdir(log_directory) if f.startswith(prefix)]

    # 初始化数据列表
    data = []

    selected_pairs_list = load_selected_pairs_from_json()

    # 分析每个日志文件
    for log_file in log_files:
        log_path = os.path.join(log_directory, log_file)

        valid_line_list = []

        with open(log_path, 'r') as file:
            for line in file:
                # 使用正则表达式找到 RE 和 TE 的值
                # available = re.search(r"filtered_available_matches_cnt: (?P<filtered_cnt>\d+),\s*", line)
                # inf_id = re.search(r"inf_id: (?P<inf_id>\d{6}),\s*", line)
                # veh_id = re.search(r"veh_id: (?P<veh_id>\d{6}),\s*", line)

                # available_value = int(available.group(1)) if available else 0
                # inf_id_value = data['inf_id']
                # veh_id_value = data['veh_id']
                
                pattern = re.compile(
                    r"inf_id: (?P<inf_id>\d{6}),\s*"
                    r"veh_id: (?P<veh_id>\d{6}),\s*"
                    r"RE: (?P<RE>[\d.]+),\s*"
                    r"TE: (?P<TE>[\d.]+),\s*"
                    r"stability: (?P<stability>[\d.]+),\s*"
                    r"time: (?P<time>[\d.]+)\s*==details==>\s*"
                    r"infra_total_box_cnt : (?P<infra_cnt>\d+),\s*"
                    r"vehicle_total_box_cnt: (?P<veh_cnt>\d+),\s*"
                    r"filtered_available_matches_cnt: (?P<filtered_cnt>\d+),\s*"
                    r"result_matched_cnt: (?P<matched_cnt>\d+),\s*"
                    r"wrong_result_matches_cnt: (?P<wrong_cnt>\d+)"
                )

                match = pattern.search(line)

                available_value = int(match.group(9)) if match else 0
                inf_id_value = match.group(1) if match else None
                veh_id_value = match.group(2) if match else None


                if (inf_id_value, veh_id_value) in selected_pairs_list and (available_value > 1):
                    valid_line_list.append(line)



        success_rate_threshold = [1, 2]
        average_re, average_te, success_rate, average_time, absolute_num = analyze_valid_line_list(valid_line_list, success_rate_threshold)
        
        # 解析特征名称（方法名称）
        # feature_name = log_file.split('_')[-3:-1]  # 假设特征名称位于第六个位置
        feature_name = log_file
        
        # 存储结果
        data.append({
            'Feature': feature_name,
            'Average RE': average_re,
            'Average TE': average_te,
            'Success Rate @1': success_rate[1] * 100,
            'Success Rate @2': success_rate[2] * 100,
            'Average Time': average_time,
            'Absolute Num': absolute_num
        })

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 打印或保存 DataFrame
    # print(df)
    df.to_csv('analysis_results.csv', index=False)


def parse_log_and_save_pairs(log_filename: str, output_filename: str = "selected_pairs_olddatarange.json"):
    """
    Parses a log file to extract (inf_id, veh_id) pairs and saves them to a JSON file.

    Args:
        log_filename (str): The path to the input log file.
        output_filename (str): The path for the output JSON file.
    """
    # 📝 A regular expression to find the inf_id and veh_id, which are both 6-digit numbers.
    # r"inf_id: (\d{6})" captures the first 6-digit number after "inf_id: "
    # r"veh_id: (\d{6})" captures the second 6-digit number after "veh_id: "
    pattern = re.compile(r"inf_id: (\d{6}),\s*veh_id: (\d{6})")

    selected_pairs = []

    print(f"Reading from log file: {log_filename}")
    try:
        # Open the log file for reading, ensuring it handles different text encodings
        with open(log_filename, 'r', encoding='utf-8') as f:
            # Process the file line by line to save memory
            for line in f:
                match = pattern.search(line)
                # If a match is found on the current line
                if match:
                    # group(1) is the first captured group (inf_id)
                    # group(2) is the second captured group (veh_id)
                    inf_id = match.group(1)
                    veh_id = match.group(2)
                    # Add the pair as a tuple to our list
                    selected_pairs.append((inf_id, veh_id))

    except FileNotFoundError:
        print(f"❌ Error: The file '{log_filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # --- Save the collected data ---
    print(f"Found {len(selected_pairs)} pairs. Saving to {output_filename}...")
    
    # Open the output file for writing
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Use json.dump to write the list of pairs to the file
        # indent=4 makes the JSON file human-readable
        json.dump(selected_pairs, f, indent=4)

    print(f"✅ Save complete. Check the file '{output_filename}'.")


if __name__ == '__main__':
    
    # 路径和阈值
    # log_path = f'Log/DAIR-V2X_15_[\'category\', \'core\']_[\'centerpoint_distance\', \'vertex_distance\']_thresholdRetained_weightedSVD_2024-08-16-22-52-18.log'
    # log_path = f'Log/dair-v2x_demo_quatro_fpfh_2024-08-12-04-26-55.log'
    # log_path = f'Log/dair-v2x_demo_fgr_fpfh_2024-08-12-04-25-40.log'
    # log_path = f'Log/DAIR-V2X_demo_GNC_TLS_fpfh_2024-08-12-06-45-18.log'
    # log_path = f'Log/dair-v2x_demo_teaser_fpfh_2024-08-12-04-25-00.log'

    # prefix = 'Log/DAIR-V2X_15_[\'category\', \'core\']_[\'centerpoint_distance\', \'vertex_distance\']_'

    # threshold = [1, 2]
    # # 执行分析
    # average_re, average_te, success_rate, average_time = analyze_log(log_path, threshold)

    # print(f"Log: {log_path}")
    # print(f"Average RE: {average_re}")
    # print(f"Average TE: {average_te}")
    # print(f"Average Time: {average_time}")
    # for x in threshold:
    #     print(f"Success Rate @{x}: {success_rate[x] * 100}%")
    
    analyze_with_same_prefix_data(prefix = '') #================

    # analyze_filter_pairs_with_same_prefix_data()

    # analyze_with_same_folder_data()

    # 2. 分析具有相同数据范围的日志文件
    # groups = ['easy', 'hard', 'selected']
    # groups = ['benchmark1', 'benchmark2', 'hard']

    # Initialize a list to store all the results to later convert into a DataFrame
    # all_results = []

    # for group in groups:
    #     results, num, num_dict = analyze_with_same_data_range(group_name=group)
    #     print(f'Group: {group}, Num: {num}, Num_dict: {num_dict}')
    #     for log_file, data in results.items():
    #         # Append each result as a dictionary to the list
    #         all_results.append({
    #             'Group': group,
    #             'Log File': log_file,
    #             'Average RE': data['RE'],
    #             'Average TE': data['TE'],
    #             'Average Time': data['time'],
    #             'Success Rate@1': f"{data['Success_rate'][1] * 100}%",
    #             'Success Rate@2': f"{data['Success_rate'][2] * 100}%"
    #         })

    # # Convert list of dictionaries into a DataFrame
    # df = pd.DataFrame(all_results)

    # # Print the DataFrame
    # print(df.to_string(index=False))



    # parse_log_and_save_pairs(log_filename="Log/old_data/DAIR-V2X_demo_QUATRO_fpfh__selected_data_info_2024-08-29-00-10-15.log")
