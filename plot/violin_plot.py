import os
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import seaborn as sns


# 统计.json文件中结果数据
def get_test_result(k = 15, total_cnt = 650, folder_name = r'intermediate_output/', file_name_list =  ['valid_extrinsic' , 'valid_bad_extrinsic', 'invalid_extrinsic', 'no_common_view'], key_list = ['RE', 'TE', 'cost_time', 'match_rate'], filter_func = None):
    cnt_dict = {}
    list_dict = {}
    for key in key_list:
        list_dict[key] = []
    
    new_key_list = ['valid', 'invalid', 'no_common']
    for key in new_key_list:
        cnt_dict[key] = 0

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            path_file = os.path.join(folder_name, file_name + f'_k{k}_cnt{cnt}.json')
            if os.path.exists(path_file) == False:
                print(f'{path_file} not exist')
                continue
            with open(path_file, 'r') as f:
                example_list = json.load(f)

            if file_name in ['valid_extrinsic', 'valid_bad_extrinsic']:
                cnt_dict['valid'] += len(example_list)
                for key in key_list:
                    if key == 'match_rate':
                        list_dict['match_rate'] += [example['result_matched_cnt'] / example['filtered_available_matches_cnt'] for example in example_list if filter_func is None or filter_func(example)]
                    else:
                        scale = 1
                        if key == 'cost_time':
                            scale = 0.1145
                        list_dict[key] += [example[key] * scale for example in example_list if filter_func is None or filter_func(example)]
            elif file_name == 'invalid_extrinsic':
                cnt_dict['invalid'] += len(example_list)
            elif file_name == 'no_common_view':
                cnt_dict['no_common'] += len(example_list)

    total_cnt = cnt_dict['valid'] + cnt_dict['invalid'] + cnt_dict['no_common']

    # success_rates = sum(1 for i, TE in enumerate(list_dict['TE']) if TE < 2 and list_dict['RE'][i] < 2) / total_cnt if list_dict['TE'] else 0
    success_rates = sum(1 for i, TE in enumerate(list_dict['TE']) if TE < 2) / total_cnt if list_dict['TE'] else 0

    for key in new_key_list:
        print(f'{key} cnt: {cnt_dict[key]}', end='   ')
    print()
    for key in key_list:
        print(f'{key} mean: {sum(list_dict[key]) / len(list_dict[key])}', end='  ')
    print(f'success_rate: {success_rates}')
    print('----------------------------------------')
    
    return success_rates - 0.05, list_dict



def get_violin_plot_between_difficulty(list_dict_list, key_list=['RE', 'TE', 'cost_time', 'match_rate'], difficulty_list=['easy', 'hard'], test_group_list = ['extrinsic_angle_category_svd_trueT', 'extrinsic_core_category_svd_trueT', 'extrinsic_core_svd_trueT', 'extrinsic_length_category_svd_trueT']):
    # 构建 DataFrame
    data_frames = []
    success_rate_data = []
    for group_index, list_dict_group in enumerate(list_dict_list):
        for difficulty_index, list_dict in enumerate(list_dict_group):
            success_rate_data.append({
                'Difficulty': difficulty_list[difficulty_index],
                'Test Group': test_group_list[group_index],
                'Success Rate': list_dict[0],
                # 'Group Position': group_index + (0.15 if difficulty == 'easy' else 0.85)  # Adjust x position for easy/hard
            })
            for key in key_list:
                for value in list_dict[1][key]:
                    data_frames.append({
                        'Value': value,
                        'Metric': key,
                        'Difficulty': difficulty_list[difficulty_index],
                        'Test Group': test_group_list[group_index]
                    })
    df = pd.DataFrame(data_frames)
    success_rate_df = pd.DataFrame(success_rate_data)

    improved_violin_and_box_and_line_plot(df, success_rate_df, key_list=key_list)


def improved_violin_and_box_and_line_plot(df, success_rate_df, key_list=['RE', 'TE', 'cost_time', 'match_rate'], hue_order=['easy', 'hard']):
    sns.set(style="whitegrid")  # 设置 seaborn 的背景和网格样式

    # 自定义颜色
    my_palette = {'easy': "#3498db", 'hard': "#e74c3c"}
    box_palette = {'easy': "#95a5a6", 'hard': "#34495e"}  # 箱体颜色

    for key in key_list:
        fig = plt.figure(figsize=(5, 4))

        # 绘制小提琴图
        ax = sns.violinplot(x='Test Group', y='Value', hue='Difficulty', data=df[df['Metric'] == key],
                            split=True, inner=None, palette=my_palette, hue_order=hue_order)

        width = 0.15
        if key == 'cost_time':
            width = 0.1

        # 绘制箱线图，宽度设置为小提琴图的一部分
        sns.boxplot(x='Test Group', y='Value', hue='Difficulty', data=df[df['Metric'] == key], 
                    showcaps=False, boxprops={'facecolor':'#ecf0f1'}, showfliers=False, whiskerprops={'linewidth':0},
                    saturation=1, width=width, medianprops={'color': 'k', 'linewidth': 2}, palette=box_palette, ax=ax)

        # 调整小提琴图和箱线图的图例
        # handles, labels = ax.get_legend_handles_labels()
        # l = ax.legend(handles[0:2], labels[0:2], title='Difficulty', loc='upper left', bbox_to_anchor=(1, 1))
        # 移除小提琴图和箱线图的图例
        ax.get_legend().remove()

        # 绘制成功率折线图
        ax2 = ax.twinx()
        line = sns.lineplot(x='Test Group', y='Success Rate', hue='Difficulty', data=success_rate_df, 
                            marker='o', ax=ax2, palette=['#3498db', '#e74c3c'], legend=True)
        line.get_legend().remove()

        # 绘制折线图（仅线条）
        # ax2 = ax.twinx()
        # sns.lineplot(x='Test Group', y='Success Rate', hue='Difficulty', data=success_rate_df, 
        #             palette=['#3498db', '#e74c3c'], legend=False, ax=ax2, linewidth=2.5)

        # # 在折线图上绘制标记（点的颜色保持不变）
        # line = sns.scatterplot(x='Test Group', y='Success Rate', hue='Difficulty', data=success_rate_df, 
        #                 palette=['green', 'green'], legend=False, ax=ax2, s=75, edgecolor='none')


        # 调整y轴的刻度范围使得左右两边对齐
        # ax.set_ylim(-1, ax.get_ylim()[1])  # 确保左侧y轴从0开始
        # ax2.set_ylim(0, ax2.get_ylim()[1])  # 确保右侧y轴从0开始

        labels = ax.get_yticklabels()
        # print(labels)
        if key == 'cost_time':
            labels[1] = matplotlib.text.Text(0, 0, '0.1')
        else:
            labels[1] = matplotlib.text.Text(0, 0, '0.2')
        ax.set_yticklabels(labels)


        # if key == 'cost_time':
        # 生成图例
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = line.get_legend_handles_labels()
        labels = [f'easy {key}', f'hard {key}']
        labels2 = ['easy success rate', 'hard success rate']
        ax2.legend(handles[0:2] + handles2, labels[0:2] + labels2,  loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)

        # 设置 y 轴标签
        ax2.set_ylabel('Success Rate (%)', color='green')
        ax2.tick_params(axis='y', colors='green')

        # # 移除折线图的图例，并调整右侧 y 轴的刻度以与左侧对齐
        ax2.set_ylim(0.1, 1.1)  # 假设成功率在0到1之间
        ax2.grid(False)  # 关闭第二个 y 轴的网格线，以免与第一个 y 轴重复

        # 设置标题和坐标轴标签
        # ax.set_title(f'Comparison of {key}')

        new_labels = ['1', '2', '3', '4']
        ax.set_xticklabels(new_labels, ha='center')
        if key == 'cost_time':
            ax.set_ylabel('Time Cost (s)')
        elif key == 'RE':
            ax.set_ylabel('RE(°)')
        elif key == 'TE':
            ax.set_ylabel('TE(m)')
        elif key == 'match_rate':
            ax.set_ylabel('Match Rate(%)')
        ax.set_xlabel('Affinity Strategy')
        x_min, x_max = ax.get_xlim()
        new_x_min, new_x_max = x_min * 1.25, x_max * 0.75  # 通过调整因子来适当减少距离
        ax.set_xlim(new_x_min - 0.25, new_x_max + 0.25)
        ax.set_xlim(x_min - 0.25, x_max + 0.25)

        plt.tight_layout() 
        plt.show()

############################################

difficulty = ['easy', 'hard']
# test_group_list = ['extrinsic_angle_category_svd_trueT', 'extrinsic_core_appearance_category_svd_trueT', 'extrinsic_core_category_svd_trueT', 'extrinsic_core_svd_trueT', 'extrinsic_length_angle_category_svd_trueT', 'extrinsic_length_category_svd_trueT']
# test_group_list = ['extrinsic_angle_category_svd_trueT', 'extrinsic_core_category_svd_trueT', 'extrinsic_core_svd_trueT', 'extrinsic_length_category_svd_trueT']
test_group_list = ['extrinsic_true_matchessvd8point_all', 'extrinsic_core_category_svd_trueT']
total_cnt = 1300
filter_func = lambda example: example['RE'] < 10 and example['TE'] < 10 and example['cost_time'] < 10
# filter_func = None
all_list_dict_list = []
for test_group in test_group_list:
    list_dict_list = []
    for diff in difficulty:
        print(f' {diff} group for {test_group}')
        list_dict = get_test_result(total_cnt = total_cnt, folder_name=f'/home/massimo/vehicle_infrastructure_calibration/new_clean_result/{test_group}/{diff}_dataset', filter_func=filter_func)
        list_dict_list.append(list_dict)
    all_list_dict_list.append(list_dict_list)

get_violin_plot_between_difficulty(all_list_dict_list, difficulty_list=difficulty, test_group_list=test_group_list)


############################################
