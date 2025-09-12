####################################
################## 通用功能函数
####################################
### 导入依赖库
import os
import json
import jax
import shutil
import zipfile
import random
import math
import pandas as pd
import numpy as np
import logging

# 获取日志记录器
logger = logging.getLogger('BindCraft')

# 定义数据表格的列标签
# 这些标签用于记录蛋白质设计过程中的各种统计数据和质量指标
def generate_dataframe_labels():
    # 轨迹标签
    trajectory_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes',
                        'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues',
                        'n_InterfaceHbonds', 'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
                        'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD', 'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    # MPNN设计标签
    core_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                    'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                    'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%', 
                    'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD', 'Binder_pLDDT', 'Binder_pTM', 'Binder_pAE', 'Binder_RMSD']

    design_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'MPNN_score', 'MPNN_seq_recovery']

    for label in core_labels:
        design_labels += ['Average_' + label] + [f'{i}_{label}' for i in range(1, 6)]

    design_labels += ['DesignTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    final_labels = ['Rank'] + design_labels

    return trajectory_labels, design_labels, final_labels

# 创建项目的基础目录结构
def generate_directories(design_path):
    design_path_names = ["Accepted", "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle", "Trajectory",
                        "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing", "Trajectory/LowConfidence", "Trajectory/Animation",
                        "Trajectory/Pickle", "MPNN", "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed", "Rejected"]
    design_paths = {}

    # 创建目录并设置design_paths[文件夹名称]变量
    for name in design_path_names:
        path = os.path.join(design_path, name)
        os.makedirs(path, exist_ok=True)
        design_paths[name] = path

    return design_paths

# 生成CSV文件以跟踪未通过筛选的设计
def generate_filter_pass_csv(failure_csv, filter_json):
    if not os.path.exists(failure_csv):
        with open(filter_json, 'r') as file:
            data = json.load(file)
        
        # 创建修改后键的列表
        names = ['Trajectory_logits_pLDDT', 'Trajectory_softmax_pLDDT', 'Trajectory_one-hot_pLDDT', 'Trajectory_final_pLDDT', 'Trajectory_Contacts', 'Trajectory_Clashes', 'Trajectory_WrongHotspot']
        special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
        tracked_filters = set()

        for key in data.keys():
            processed_name = key  # 默认使用完整键

            # 检查键是否以任何特殊前缀开头
            for prefix in special_prefixes:
                if key.startswith(prefix):
                    # 去掉前缀并使用剩余部分
                    processed_name = key.split('_', 1)[1]
                    break

            # 处理'InterfaceAAs'并附加氨基酸
            if 'InterfaceAAs' in processed_name:
                # 生成20个'InterfaceAAs'变体，附加氨基酸
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                for aa in amino_acids:
                    variant_name = f"InterfaceAAs_{aa}"
                    if variant_name not in tracked_filters:
                        names.append(variant_name)
                        tracked_filters.add(variant_name)
            elif processed_name not in tracked_filters:
                # 如果之前未添加过，则添加处理后的名称
                names.append(processed_name)
                tracked_filters.add(processed_name)

        # 创建包含0的数据框
        df = pd.DataFrame(columns=names)
        df.loc[0] = [0] * len(names)

        df.to_csv(failure_csv, index=False)

# 更新轨迹和早期预测的失败率
def update_failures(failure_csv, failure_column_or_dict):
    failure_df = pd.read_csv(failure_csv)
    
    def strip_model_prefix(name):
        # 如果存在模型特定前缀则去除
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name
    
    # 更新来自复合物预测的字典
    if isinstance(failure_column_or_dict, dict):
        # 使用失败字典进行更新
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        # 从轨迹生成更新单个列
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1
    
    failure_df.to_csv(failure_csv, index=False)

# 检查生成的轨迹数量
def check_n_trajectories(design_paths, advanced_settings):
    n_trajectories = [f for f in os.listdir(design_paths["Trajectory/Relaxed"]) if f.endswith('.pdb') and not f.startswith('.')]

    if advanced_settings["max_trajectories"] is not False and len(n_trajectories) >= advanced_settings["max_trajectories"]:
        logger.info(f"达到目标轨迹数量 {len(n_trajectories)}，停止执行...")
        return True
    else:
        return False

# 检查是否达到所需的接受目标数量，对它们进行排序，并分析序列和结构特性
def check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels):
    accepted_binders = [f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb') and not f.startswith('.')]

    if len(accepted_binders) >= target_settings["number_of_final_designs"]:
        logger.info(f"达到目标设计数量 {len(accepted_binders)}！重新排序...")

        # 清空排序文件夹，以防在此期间添加了新设计，这样我们可以重新排序所有设计
        for f in os.listdir(design_paths["Accepted/Ranked"]):
            os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

        # 加载设计结合剂的数据框
        design_df = pd.read_csv(mpnn_csv)
        design_df = design_df.sort_values('Average_i_pTM', ascending=False)
        
        # 创建最终CSV数据框以复制匹配行，用列标签初始化
        final_df = pd.DataFrame(columns=final_labels)

        # 检查设计的排序并将它们复制到文件夹中，使用新的排序ID
        rank = 1
        for _, row in design_df.iterrows():
            for binder in accepted_binders:
                target_settings["binder_name"], model = binder.rsplit('_model', 1)
                if target_settings["binder_name"] == row['Design']:
                    # 排序并复制到排序文件夹
                    row_data = {'Rank': rank, **{label: row[label] for label in design_labels}}
                    final_df = pd.concat([final_df, pd.DataFrame([row_data])], ignore_index=True)
                    old_path = os.path.join(design_paths["Accepted"], binder)
                    new_path = os.path.join(design_paths["Accepted/Ranked"], f"{rank}_{target_settings['binder_name']}_model{model.rsplit('.', 1)[0]}.pdb")
                    shutil.copyfile(old_path, new_path)

                    rank += 1
                    break

        # 将final_df保存到final_csv
        final_df.to_csv(final_csv, index=False)

        # 压缩大文件夹以节省空间
        if advanced_settings["zip_animations"]:
            zip_and_empty_folder(design_paths["Trajectory/Animation"], '.html')

        if advanced_settings["zip_plots"]:
            zip_and_empty_folder(design_paths["Trajectory/Plots"], '.png')

        return True

    else:
        return False

# 加载所需的螺旋性值
def load_helicity(advanced_settings):
    if advanced_settings["random_helicity"] is True:
        # 将采样对螺旋性的随机偏向
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # 使用预设的螺旋性偏向
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # 对螺旋性无偏向
        helicity_value = 0
    return helicity_value

# 报告支持JAX的设备
def check_jax_gpu():
    devices = jax.devices()

    has_gpu = any(device.platform == 'gpu' for device in devices)

    if not has_gpu:
        logger.error("未找到GPU设备，程序终止。")
        exit()
    else:
        logger.info("可用的GPU:")
        for i, device in enumerate(devices):
            logger.info(f"{device.device_kind}{i + 1}: {device.platform}")

# 检查所有传递的输入文件
def perform_input_check(args):
    # 获取当前脚本的目录
    binder_script_path = os.path.dirname(os.path.abspath(__file__))

    # 确保提供了设置文件
    if not args.settings:
        logger.error("错误: --settings 参数是必需的。")
        exit()

    # 如果未提供，设置默认的filters.json路径
    if not args.filters:
        args.filters = os.path.join(binder_script_path, 'settings_filters', 'default_filters.json')

    # 如果未提供，设置随机的高级json设置文件
    if not args.advanced:
        args.advanced = os.path.join(binder_script_path, 'settings_advanced', 'default_4stage_multimer.json')

    return args.settings, args.filters, args.advanced

# 检查特定的高级设置
def perform_advanced_settings_check(advanced_settings, bindcraft_folder):
    # 设置模型权重和可执行文件的路径
    if bindcraft_folder == "colab":
        advanced_settings["af_params_dir"] = '/content/bindcraft/params/'
        advanced_settings["dssp_path"] = '/content/bindcraft/functions/dssp'
        advanced_settings["dalphaball_path"] = '/content/bindcraft/functions/DAlphaBall.gcc'
    else:
        # 如果尚未设置，则单独设置路径
        if not advanced_settings["af_params_dir"]:
            advanced_settings["af_params_dir"] = bindcraft_folder
        if not advanced_settings["dssp_path"]:
            advanced_settings["dssp_path"] = os.path.join(bindcraft_folder, 'functions', 'dssp')
        if not advanced_settings["dalphaball_path"]:
            advanced_settings["dalphaball_path"] = os.path.join(bindcraft_folder, 'functions', 'DAlphaBall.gcc')

    # 检查omit_AAs设置的格式
        omit_aas = advanced_settings["omit_AAs"]
    if advanced_settings["omit_AAs"] in [None, False, '']:
        advanced_settings["omit_AAs"] = None
    elif isinstance(advanced_settings["omit_AAs"], str):
        advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

    return advanced_settings

# 从JSON文件加载设置
def load_json_settings(settings_json, filters_json, advanced_json):
    # 从json文件加载设置
    with open(settings_json, 'r') as file:
        target_settings = json.load(file)

    with open(advanced_json, 'r') as file:
        advanced_settings = json.load(file)

    with open(filters_json, 'r') as file:
        filters = json.load(file)

    return target_settings, advanced_settings, filters

# AF2模型设置，确保使用带有模板选项的非重叠模型进行设计和重新预测
def load_af2_models(af_multimer_setting):
    if af_multimer_setting:
        design_models = [0,1,2,3,4]
        prediction_models = [0,1]
        multimer_validation = False
    else:
        design_models = [0,1]
        prediction_models = [0,1,2,3,4]
        multimer_validation = True

    return design_models, prediction_models, multimer_validation

# 创建CSV用于数据插入
def create_dataframe(csv_file, columns):
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)

# 将统计数据行插入CSV
def insert_data(csv_file, data_array):
    df = pd.DataFrame([data_array])
    df.to_csv(csv_file, mode='a', header=False, index=False)

# 保存生成的序列
def save_fasta(design_name, sequence, design_paths):
    fasta_path = os.path.join(design_paths["MPNN/Sequences"], design_name+".fasta")
    with open(fasta_path,"w") as fasta:
        line = f'>{design_name}\n{sequence}'
        fasta.write(line+"\n")

# 清除PDB中不必要的rosetta信息
def clean_pdb(pdb_file):
    # 读取PDB文件并过滤相关行
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK'))]

    # 将清理后的行写回原始PDB文件
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)

def zip_and_empty_folder(folder_path, extension):
    folder_basename = os.path.basename(folder_path)
    zip_filename = os.path.join(os.path.dirname(folder_path), folder_basename + '.zip')

    # 以'a'模式打开压缩文件，如果存在则追加，否则创建新文件
    with zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                # 创建绝对路径
                file_path = os.path.join(folder_path, file)
                # 将文件添加到压缩文件中，如果已存在则替换
                zipf.write(file_path, arcname=file)
                # 添加到压缩文件后删除该文件
                os.remove(file_path)
    logger.info(f"文件夹 '{folder_path}' 中的文件已被压缩并删除。")

# 计算统计数据的平均值
def calculate_averages(statistics, handle_aa=False):
    # 初始化字典来保存每个统计数据的总和
    sums = {}
    # 初始化字典来保存每个氨基酸计数的总和
    aa_sums = {}

    # 遍历模型编号
    for model_num in range(1, 6):  # 假设模型编号为1到5
        # 检查模型的数据是否存在
        if model_num in statistics:
            # 获取模型的统计数据
            model_stats = statistics[model_num]
            # 对于每个统计数据，将其值加到总和中
            for stat, value in model_stats.items():
                # 如果这是我们第一次看到这个统计数据，将其总和初始化为0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # 如果统计数据是mpnn_interface_AA且我们应该单独处理它，则这样做
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # 如果这是我们第一次看到这个氨基酸，将其总和初始化为0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # 现在我们有了总和，可以计算平均值
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # 如果我们正在处理氨基酸计数，计算它们的平均值
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages

# 基于特征阈值过滤设计
def check_filters(mpnn_data, design_labels, filters):
    # 检查mpnn_data与标签的对应关系
    mpnn_dict = {label: value for label, value in zip(design_labels, mpnn_data)}

    unmet_conditions = []

    # 检查过滤器与阈值的对比
    for label, conditions in filters.items():
        # 界面氨基酸计数的特殊条件
        if label == 'Average_InterfaceAAs' or label == '1_InterfaceAAs' or label == '2_InterfaceAAs' or label == '3_InterfaceAAs' or label == '4_InterfaceAAs' or label == '5_InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if mpnn_dict.get(label) is None:
                    continue
                value = mpnn_dict.get(label).get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            # 如果没有阈值，则跳过
            value = mpnn_dict.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # 如果所有过滤器都通过则返回True
    if len(unmet_conditions) == 0:
        return True
    # 如果某些过滤器未满足，返回它们
    else:
        return unmet_conditions
