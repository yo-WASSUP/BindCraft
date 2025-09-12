####################################
###################### BindCraft 蛋白质结合剂设计运行脚本
####################################
### 导入依赖库
from functions import *
from logging_config import setup_logging

# 检查是否有JAX兼容的GPU可用，否则退出程序
# JAX是用于机器学习的数值计算库，需要GPU加速
check_jax_gpu()

######################################
### 解析输入路径参数
# 创建命令行参数解析器，用于接收用户输入的配置文件路径
parser = argparse.ArgumentParser(description='运行BindCraft蛋白质结合剂设计的脚本')

# 必需参数：基础设置文件路径，包含目标蛋白质信息和设计参数
parser.add_argument('--settings', '-s', type=str, required=True,
                    help='基础设置JSON文件的路径。必需参数。')
# 可选参数：过滤器设置文件，用于筛选设计结果
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='用于过滤设计结果的过滤器JSON文件路径。如未提供，将使用默认设置。')
# 可选参数：高级设置文件，包含详细的设计算法参数
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='包含附加设计设置的高级JSON文件路径。如未提供，将使用默认设置。')

args = parser.parse_args()

# 检查输入设置文件的有效性和完整性
settings_path, filters_path, advanced_path = perform_input_check(args)

### 从JSON文件加载设置参数
# target_settings: 目标蛋白质相关设置（PDB文件路径、结合位点等）
# advanced_settings: 高级算法参数（迭代次数、权重等）
# filters: 结果筛选条件（质量阈值等）
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

# 提取文件名（不包含扩展名）用于记录和命名
settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

### 加载AlphaFold2模型设置
# design_models: 用于设计的模型索引
# prediction_models: 用于预测验证的模型索引  
# multimer_validation: 是否使用多聚体模型进行验证
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

### 检查和完善高级设置参数
# 获取BindCraft安装目录，用于定位工具和参数文件
bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

### 生成输出目录结构
# 创建保存设计结果的各种子目录（轨迹、MPNN优化、接受的设计等）
design_paths = generate_directories(target_settings["design_path"])

### 生成数据表格标签
# 定义CSV文件的列标题，用于记录设计过程中的各种统计数据
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

# 定义各种统计数据CSV文件的路径
trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')  # 轨迹统计
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')      # MPNN设计统计
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')    # 最终设计统计
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')         # 失败原因统计

# 创建空的CSV文件，包含列标题
create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(final_csv, final_labels)
generate_filter_pass_csv(failure_csv, args.filters)

# 初始化日志系统
logger = setup_logging(target_settings["design_path"])

####################################
####################################
####################################
### 初始化PyRosetta分子建模软件
# PyRosetta是用于蛋白质结构预测、设计和分析的软件包
# 设置各种参数：忽略未识别残基、忽略零占用率、静音模式、使用DAlphaBall计算孔洞等
logger.info("正在初始化PyRosetta分子建模软件...")
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

logger.info("="*80)
logger.info("BindCraft 蛋白质结合剂设计系统启动")
logger.info("="*80)
logger.info(f"🎯 目标蛋白质: {settings_file}")
logger.info(f"⚙️  设计协议: {advanced_file}")
logger.info(f"🔍 筛选标准: {filters_file}")
logger.info(f"📁 输出目录: {target_settings['design_path']}")
logger.info(f"🧬 设计长度范围: {target_settings['lengths'][0]}-{target_settings['lengths'][1]} 残基")
logger.info(f"🎯 目标设计数量: {target_settings['number_of_final_designs']}")
logger.info("="*80)

####################################
# 初始化计数器和计时器
script_start_time = time.time()  # 记录脚本开始时间
trajectory_n = 1                 # 轨迹编号计数器
accepted_designs = 0             # 通过筛选的设计数量计数器

### 开始设计主循环
# 这个循环会持续运行直到获得足够数量的合格设计或达到最大轨迹数
while True:
    ### 检查是否已达到目标结合剂数量
    # 检查已接受的设计数量是否达到用户设定的目标
    final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

    if final_designs_reached:
        # 如果达到目标数量，停止设计循环
        break

    ### 检查是否达到最大允许的轨迹数量
    # 防止程序无限运行，设置轨迹生成的上限
    max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

    if max_trajectories_reached:
        break

    ### 初始化单个设计轨迹
    # 记录生成单个设计轨迹的开始时间
    trajectory_start_time = time.time()

    # 生成随机种子以产生不同的设计变体
    # 随机种子确保每次运行产生不同的结果
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

    # 从用户定义的长度范围中随机选择结合剂长度
    # 不同长度的结合剂可能有不同的结合特性
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(samples)

    # 加载期望的螺旋性值以采样不同的二级结构内容
    # 螺旋性影响蛋白质的形状和结合特性
    helicity_value = load_helicity(advanced_settings)

    # 生成设计名称并检查是否已运行过相同的轨迹
    # 命名格式：目标名称_长度_种子号
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
    trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

    if not trajectory_exists:
        logger.info(f"🚀 开始轨迹 {trajectory_n}: {design_name}")
        logger.debug(f"轨迹参数 - 长度: {length}, 种子: {seed}, 螺旋性: {helicity_value}")

        ### 开始结合剂幻化（Hallucination）过程
        # 这是核心的AI设计步骤，使用AlphaFold2反向传播来"幻化"出能结合目标蛋白质的新蛋白质
        # 输入参数包括：设计名称、目标PDB文件、目标链、热点残基、长度、随机种子、螺旋性等
        logger.debug("正在执行结合剂幻化过程...")
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths, failure_csv)
        
        # 提取轨迹的质量指标（pLDDT：置信度，pTM：模板匹配，i_pTM：界面模板匹配，pAE：预测误差等）
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

        # 将指标四舍五入到小数点后两位以便记录
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

        # 计算轨迹生成所用时间
        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        
        logger.info(f"✅ 初始轨迹生成完成，耗时: {trajectory_time_text}")
        logger.debug(f"轨迹质量指标 - pLDDT: {trajectory_metrics.get('plddt', 'N/A')}")

        # 如果轨迹没有终止信号（即成功生成），则继续后续处理
        if trajectory.aux["log"]["terminate"] == "":
            # 使用PyRosetta对结合剂进行结构松弛优化以计算统计数据
            # 松弛过程会优化原子位置，消除不合理的几何结构
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
            pr_relax(trajectory_pdb, trajectory_relaxed)

            # 定义结合剂链标识符（通常为"B"链）
            # 这是为了防止ColabDesign多链解析发生变化时的占位符
            binder_chain = "B"

            # 计算松弛前后的原子冲突数量
            # 冲突是指原子间距离过近的不合理结构
            num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

            # 分析初始轨迹结合剂和界面的二级结构含量
            # 二级结构包括：α-螺旋（alpha）、β-折叠（beta）、无规卷曲（loops）
            trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

            # 分析松弛后AF2轨迹的界面评分
            # 界面评分包括结合能、形状互补性、氢键等多个指标
            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)

            # 获取初始结合剂的氨基酸序列
            trajectory_sequence = trajectory.get_seq(get_best=True)[0]

            # 分析序列的合理性（检查半胱氨酸、UV吸收残基等）
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)

            # 计算目标结构相对于输入PDB的RMSD（均方根偏差）
            # RMSD衡量两个结构的相似程度，数值越小越相似
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

            # 将轨迹统计数据保存到CSV文件
            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues, 
                                trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                                trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                                trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                                trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                                trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                                trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd, 
                                trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
            insert_data(trajectory_csv, trajectory_data)

            if not trajectory_interface_residues:
                logger.warning(f"⚠️  未发现界面残基 - {design_name}，跳过MPNN优化")
                logger.debug(f"轨迹 {design_name} 的界面分析未找到结合界面残基")
                continue
            
            if advanced_settings["enable_mpnn"]:
                # 初始化MPNN优化阶段
                logger.info(f"🧬 开始MPNN序列优化阶段 - {design_name}")
                mpnn_n = 1
                accepted_mpnn = 0
                mpnn_dict = {}
                design_start_time = time.time()

                ### 使用MPNN重新设计初始结合剂
                mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
                existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)

                # 创建符合氨基酸组成要求的MPNN序列集合
                restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()

                mpnn_sequences = sorted({
                    mpnn_trajectories['seq'][n][-length:]: {
                        'seq': mpnn_trajectories['seq'][n][-length:],
                        'score': mpnn_trajectories['score'][n],
                        'seqid': mpnn_trajectories['seqid'][n]
                    } for n in range(advanced_settings["num_seqs"])
                    if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                    and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
                }.values(), key=lambda x: x['score'])

                del existing_mpnn_sequences
  
                # 检查氨基酸排除和重复检查后是否还有序列剩余，如果有则继续预测
                if mpnn_sequences:
                    logger.info(f"📊 MPNN生成了 {len(mpnn_sequences)} 个候选序列")
                    logger.debug(f"氨基酸限制: {advanced_settings['omit_AAs'] if advanced_settings['force_reject_AA'] else '无'}")
                    
                    # 如果轨迹含有β-折叠结构，增加验证循环数以优化预测
                    if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                        logger.debug(f"检测到高β-折叠含量 ({trajectory_beta}%)，增加验证循环数")
                        advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                    ### 编译预测模型以加快MPNN序列预测速度
                    clear_mem()
                    # 编译复合物预测模型
                    complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"], data_dir=advanced_settings["af_params_dir"], 
                                                                use_multimer=multimer_validation, use_initial_guess=advanced_settings["predict_initial_guess"], use_initial_atom_pos=advanced_settings["predict_bigbang"])
                    if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
                        complex_prediction_model.prep_inputs(pdb_filename=trajectory_pdb, chain='A', binder_chain='B', binder_len=length, use_binder_template=True, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
                    else:
                        complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"], binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"])

                    # 编译结合剂单体预测模型
                    binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False, 
                                                                use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"], 
                                                                data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
                    binder_prediction_model.prep_inputs(length=length)

                    # 遍历设计的序列        
                    for mpnn_sequence in mpnn_sequences:
                        mpnn_time = time.time()

                        # 生成MPNN设计名称编号
                        mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                        mpnn_score = round(mpnn_sequence['score'],2)
                        mpnn_seqid = round(mpnn_sequence['seqid'],2)

                        # 将设计添加到字典中
                        mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid}

                        # 保存FASTA序列文件
                        if advanced_settings["save_mpnn_fasta"] is True:
                            save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)
                        
                        ### 使用掩码模板预测MPNN重新设计的结合剂复合物
                        mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                                        mpnn_sequence['seq'], mpnn_design_name,
                                                                                        target_settings["starting_pdb"], target_settings["chains"],
                                                                                        length, trajectory_pdb, prediction_models, advanced_settings,
                                                                                        filters, design_paths, failure_csv)

                        # 如果AF2筛选未通过则跳过评分
                        if not pass_af2_filters:
                            logger.debug(f"❌ AF2基础筛选未通过 - {mpnn_design_name}，跳过界面评分")
                            mpnn_n += 1
                            continue

                        # 分别计算每个模型的统计数据
                        for model_num in prediction_models:
                            mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                            mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_design_pdb):
                                # 计算松弛前后的冲突数量
                                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                                # 分析松弛后AF2轨迹的界面评分
                                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)

                                # 计算初始轨迹结合剂的二级结构含量
                                mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)
                                
                                # 计算未对齐RMSD以确定结合剂是否位于设计的结合位点
                                rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain)

                                # 计算目标相对于输入PDB的RMSD
                                target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                                # 将附加统计数据添加到mpnn_complex_statistics字典中
                                mpnn_complex_statistics[model_num+1].update({
                                    'i_pLDDT': mpnn_i_plddt,
                                    'ss_pLDDT': mpnn_ss_plddt,
                                    'Unrelaxed_Clashes': num_clashes_mpnn,
                                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                                    'PackStat': mpnn_interface_scores['interface_packstat'],
                                    'dG': mpnn_interface_scores['interface_dG'],
                                    'dSASA': mpnn_interface_scores['interface_dSASA'], 
                                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                    'InterfaceAAs': mpnn_interface_AA,
                                    'Interface_Helix%': mpnn_alpha_interface,
                                    'Interface_BetaSheet%': mpnn_beta_interface,
                                    'Interface_Loop%': mpnn_loops_interface,
                                    'Binder_Helix%': mpnn_alpha,
                                    'Binder_BetaSheet%': mpnn_beta,
                                    'Binder_Loop%': mpnn_loops,
                                    'Hotspot_RMSD': rmsd_site,
                                    'Target_RMSD': target_rmsd
                                })

                                # 通过删除未松弛的预测MPNN复合物PDB来节省空间
                                if advanced_settings["remove_unrelaxed_complex"]:
                                    os.remove(mpnn_design_pdb)

                        # 计算复合物平均值
                        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
                        
                        ### 在单序列模式下单独预测结合剂
                        binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                                trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths)

                        # 提取结合剂相对于原始轨迹的RMSD
                        for model_num in prediction_models:
                            mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_binder_pdb):
                                rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A")

                            # 添加到统计数据中
                            binder_statistics[model_num+1].update({
                                    'Binder_RMSD': rmsd_binder
                                })

                            # 通过删除结合剂单体模型来节省空间
                            if advanced_settings["remove_binder_monomer"]:
                                os.remove(mpnn_binder_pdb)

                        # 计算结合剂平均值
                        binder_averages = calculate_averages(binder_statistics)

                        # 分析序列以确保没有半胱氨酸并包含吸收UV的残基用于检测
                        seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

                        # 测量生成设计所用的时间
                        mpnn_end_time = time.time() - mpnn_time
                        elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"


                        # 将MPNN设计的统计数据插入CSV，如果对应模型不存在则返回None
                        model_numbers = range(1, 6)
                        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                            'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                            'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

                        # 用非统计数据初始化mpnn_data
                        mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid]

                        # 添加mpnn_complex的统计数据
                        for label in statistics_labels:
                            mpnn_data.append(mpnn_complex_averages.get(label, None))
                            for model in model_numbers:
                                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

                        # 添加结合剂的统计数据
                        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # 这些是单独结合剂的标签
                            mpnn_data.append(binder_averages.get(label, None))
                            for model in model_numbers:
                                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                        # 添加其余的非统计数据
                        mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

                        # 将数据插入CSV
                        insert_data(mpnn_csv, mpnn_data)

                        # 通过pLDDT找到最佳模型编号
                        plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}

                        # 找到具有最高值的键
                        highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

                        # 输出键的编号部分
                        best_model_number = highest_plddt_key - 10
                        best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

                        # 根据筛选阈值运行设计数据
                        filter_conditions = check_filters(mpnn_data, design_labels, filters)
                        if filter_conditions == True:
                            logger.info(f"✅ 设计通过所有筛选 - {mpnn_design_name}")
                            logger.debug(f"设计质量 - 平均pLDDT: {mpnn_complex_averages.get('pLDDT', 'N/A')}, "
                                       f"平均i_pTM: {mpnn_complex_averages.get('i_pTM', 'N/A')}")
                            accepted_mpnn += 1
                            accepted_designs += 1
                            
                            # 将设计复制到接受文件夹
                            shutil.copy(best_model_pdb, design_paths["Accepted"])

                            # 将数据插入最终CSV
                            final_data = [''] + mpnn_data
                            insert_data(final_csv, final_data)

                            # 从接受的轨迹复制动画
                            if advanced_settings["save_design_animations"]:
                                accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
                                if not os.path.exists(accepted_animation):
                                    shutil.copy(os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html"), accepted_animation)

                            # 复制接受轨迹的图表
                            plot_files = os.listdir(design_paths["Trajectory/Plots"])
                            plots_to_copy = [f for f in plot_files if f.startswith(design_name) and f.endswith('.png')]
                            for accepted_plot in plots_to_copy:
                                source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
                                target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
                                if not os.path.exists(target_plot):
                                    shutil.copy(source_plot, target_plot)

                        else:
                            logger.debug(f"❌ 筛选条件未满足 - {mpnn_design_name}")
                            logger.debug(f"未通过的筛选条件: {', '.join(filter_conditions) if isinstance(filter_conditions, list) else '多项条件'}")
                            failure_df = pd.read_csv(failure_csv)
                            special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                            incremented_columns = set()

                            for column in filter_conditions:
                                base_column = column
                                for prefix in special_prefixes:
                                    if column.startswith(prefix):
                                        base_column = column.split('_', 1)[1]

                                if base_column not in incremented_columns:
                                    failure_df[base_column] = failure_df[base_column] + 1
                                    incremented_columns.add(base_column)

                            failure_df.to_csv(failure_csv, index=False)
                            shutil.copy(best_model_pdb, design_paths["Rejected"])
                        
                        # 增加MPNN设计编号
                        mpnn_n += 1

                        # 如果同一轨迹有足够的MPNN序列通过筛选则停止
                        if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                            break

                    if accepted_mpnn >= 1:
                        logger.info(f"🎉 轨迹 {design_name} 产生了 {accepted_mpnn} 个通过筛选的MPNN设计")
                    else:
                        logger.info(f"⚠️  轨迹 {design_name} 未产生通过筛选的MPNN设计")

                else:
                    logger.warning(f"⚠️  检测到重复的MPNN设计序列，跳过当前轨迹优化 - {design_name}")

                # 通过删除未松弛的设计轨迹PDB来节省空间
                if advanced_settings["remove_unrelaxed_trajectory"]:
                    os.remove(trajectory_pdb)
                    logger.debug(f"已删除未松弛的轨迹PDB文件以节省空间")

                # 测量为一个轨迹生成设计所需的时间
                design_time = time.time() - design_start_time
                design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                logger.info(f"⏱️  轨迹 {design_name} 设计和验证完成，耗时: {design_time_text}")

            # 分析轨迹的拒绝率，看是否需要重新调整设计权重
            if trajectory_n >= advanced_settings["start_monitoring"] and advanced_settings["enable_rejection_check"]:
                acceptance = accepted_designs / trajectory_n
                if not acceptance >= advanced_settings["acceptance_rate"]:
                    logger.error(f"🚨 成功设计的比例 ({acceptance:.2%}) 低于定义的接受率 ({advanced_settings['acceptance_rate']:.2%})!")
                    logger.error("建议调整设计参数！脚本执行停止...")
                    break

        # 增加轨迹编号
        trajectory_n += 1
        
        # 定期输出进度信息
        if trajectory_n % 10 == 0:
            current_acceptance = accepted_designs / trajectory_n if trajectory_n > 0 else 0
            logger.info(f"📊 进度报告 - 已完成轨迹: {trajectory_n}, 接受的设计: {accepted_designs}, 接受率: {current_acceptance:.2%}")
        
        gc.collect()

### 脚本完成
elapsed_time = time.time() - script_start_time
elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"

logger.info("="*80)
logger.info("🎯 BindCraft 设计任务完成!")
logger.info("="*80)
logger.info(f"📊 总轨迹数量: {trajectory_n}")
logger.info(f"✅ 接受的设计: {accepted_designs}")
logger.info(f"📈 总体接受率: {(accepted_designs / trajectory_n * 100):.2f}%" if trajectory_n > 0 else "N/A")
logger.info(f"⏱️  总执行时间: {elapsed_text}")
logger.info(f"📁 结果保存在: {target_settings['design_path']}")
logger.info("="*80)
