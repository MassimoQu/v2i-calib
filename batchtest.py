import itertools
import subprocess
from multiprocessing import Pool

# 参数取值列表
# matches_filter_strategies = ['thresholdRetained', 'topRetained', 'trueRetained', 'allRetained']
# matches2extrinsic_strategies = ['weightedSVD', 'evenSVD']
# svd_strategies = ['svd_with_match', 'svd_without_match']
# filter_nums = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# matches_filter_strategies = ['thresholdRetained', 'trueRetained']
# matches2extrinsic_strategies = ['weightedSVD', 'evenSVD']
# svd_strategies = ['svd_with_match', 'svd_without_match']
# filter_nums = [15, 20]

matches_filter_strategies = ['thresholdRetained', 'trueRetained']
matches2extrinsic_strategies = ['weightedSVD']
svd_strategies = ['svd_with_match']
filter_nums = [10, 15, 20, 25]
parallel_flag = 1
corresponding_parallel_flag = [0, 1]

# 生成所有参数组合
all_combinations = itertools.product(
    matches_filter_strategies,
    matches2extrinsic_strategies,
    svd_strategies,
    filter_nums,
    corresponding_parallel_flag
)

def run_command(combination):
    mf, me, svd, fn, cp = combination
    cmd = [
        'conda', 'run', '-n', 'v2icalib', 
        'python', 'test.py',
        '--test_type', 'batch',
        '--matches_filter_strategy', mf,
        '--matches2extrinsic_strategies', me,
        '--svd_strategy', svd,
        '--filter_num', str(fn),
        '--parallel_flag', str(parallel_flag),
        '--correspoding_parallel_flag', str(cp)
    ]
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True  # 自动检查返回码
        )
        print(f"✅ Success: {' '.join(cmd)}")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {' '.join(cmd)}")
        print(f"Exit Code: {e.returncode}")
        print(f"Error Output:\n{e.stderr}")

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.map(run_command, all_combinations)
