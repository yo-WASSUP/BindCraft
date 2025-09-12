####################################
############## ColabDesign功能函数
####################################
### 导入依赖库
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import logging
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .pyrosetta_utils import pr_relax, align_pdbs
from .generic_utils import update_failures

# 获取日志记录器
logger = logging.getLogger('BindCraft')

# 幻化（生成）一个结合剂
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # 为新轨迹清理GPU内存
    clear_mem()

    # 初始化结合剂幻化模型
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # 检查热点残基的合理性
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])

    ### 根据指定设置更新权重
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })

    # 重新定义分子内接触(con)和分子间接触(i_con)的定义
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
        

    ### 附加损失函数
    if advanced_settings["use_rg_loss"]:
        # 回转半径损失
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # 界面pTM损失
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # 末端距离损失
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # 添加螺旋性损失
    add_helix_loss(af_model, helicity_value)

    # 根据蛋白质长度计算要进行的突变数量
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### 根据选择开始设计算法
    if advanced_settings["design_algorithm"] == '2stage':
        # 使用梯度下降获取PSSM配置文件，然后使用PSSM偏向随机突变采样以减少损失
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        # 使用logits、softmax和独热编码的3阶段设计
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"], 
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        # 通过使用减少损失的随机突变进行设计
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        # 通过使用减少损失的随机突变进行设计（MCMC方法）
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        # 初始logits预筛选轨迹
        logger.info("阶段1: 测试Logits")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

        # 根据最低'loss'值确定最佳迭代的pLDDT
        initial_plddt = get_best_plddt(af_model, length)
        
        # 如果最佳迭代具有足够高的置信度则继续
        if initial_plddt > 0.65:
            logger.info(f"初始轨迹pLDDT良好，继续: {initial_plddt}")
            if advanced_settings["optimise_beta"]:
                # 临时保存模型以评估二级结构
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                # 如果检测到β-折叠轨迹则选择优化
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    logger.info("检测到β-折叠轨迹，优化设置")

            # 剩余多少logit迭代
            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                logger.info("阶段1: 额外Logits优化")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                logger.info(f"优化后logit轨迹pLDDT: {logit_plddt}")
            else:
                logit_plddt = initial_plddt

            # 执行softmax轨迹设计
            if advanced_settings["temporary_iterations"] > 0:
                logger.info("阶段2: Softmax优化")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # 执行独热编码
            if softmax_plddt > 0.65:
                logger.info(f"Softmax轨迹pLDDT良好，继续: {softmax_plddt}")
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    logger.info("阶段3: 独热编码优化")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > 0.65:
                    # 执行贪婪突变优化
                    logger.info(f"独热编码轨迹pLDDT良好，继续: {onehot_plddt}")
                    if advanced_settings["greedy_iterations"] > 0:
                        logger.info("阶段4: PSSM半贪婪优化")
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    logger.warning(f"独热编码轨迹pLDDT太低无法继续: {onehot_plddt}")

            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                logger.warning(f"Softmax轨迹pLDDT太低无法继续: {softmax_plddt}")

        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            logger.warning(f"初始轨迹pLDDT太低无法继续: {initial_plddt}")

    else:
        logger.error("错误: 未选择有效的设计模型")
        exit()
        return

    ### 保存轨迹PDB文件
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # 通过检查置信度、冲突和接触来检查轨迹是否值得优化
    # 检查冲突
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 or ca_clashes > 0:
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        logger.warning("检测到严重冲突，跳过分析和MPNN优化")
    else:
        # 检查是否为低质量预测
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            logger.warning("轨迹起始置信度低，跳过分析和MPNN优化")
        else:
            # 是否有足够的接触可以考虑？
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # 如果少于3个接触则蛋白质漂浮在上方且不是结合剂
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                logger.warning("界面接触太少，跳过分析和MPNN优化")
            else:
                # 太好了，轨迹没问题！我们可以继续
                af_model.aux["log"]["terminate"] = ""
                logger.info(f"轨迹成功，最终pLDDT: {final_plddt}")

    # 移动低质量预测文件:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    ### 获取采样序列用于绘图
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### 保存幻化轨迹动画
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# 使用掩码模板目标运行结合剂预测
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, failure_csv, seed=None):
    prediction_stats = {}

    # 清理序列
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # 重置筛选条件
    pass_af2_filters = True
    filter_failures = {}

    # 开始每个AF2模型的预测，由于掩码模板默认使用2个
    for model_num in prediction_models:
        # 检查以确保预测尚不存在
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # 预测模型
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # 包含plddt, ptm, i_ptm, pae, i_pae

            # 提取模型的统计数据
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'i_pTM': round(prediction_metrics['i_ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2), 
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats

            # 筛选条件列表和相应的键
            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]

            # 执行初始AF2值筛选以确定是否跳过松弛和界面评分
            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

            if not pass_af2_filters:
                break

    # 用失败计数更新CSV文件
    if filter_failures:
        update_failures(failure_csv, filter_failures)

    # AF2筛选通过，继续松弛
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            pr_relax(complex_pdb, mpnn_relaxed)
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)

    return prediction_stats, pass_af2_filters

# 单独运行结合剂预测
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None):
    binder_stats = {}

    # 为预测准备序列
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)

    # 分别预测每个模型
    for model_num in prediction_models:
        # 检查以确保预测尚不存在
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # 预测模型
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # 包含plddt, ptm, pae

            # 将结合剂模型对齐到轨迹结合剂
            align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")

            # 提取模型的统计数据
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats

    return binder_stats

# 运行MPNN为结合剂生成序列
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    # 清理GPU内存
    clear_mem()

    # 初始化MPNN模型
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # 检查是否保持轨迹生成的界面或是否用MPNN重新设计
    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        logger.info(f"固定界面残基: {trajectory_interface_residues}")
    else:
        fixed_positions = 'A'

    # 为MPNN准备输入
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # 并行采样MPNN序列
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences

# 获取最佳模型的pLDDT
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# 为ColabDesign定义回转半径损失
def add_rg_loss(self, weight=0.1):
    '''添加回转半径损失'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# 为ColabDesign定义界面pTM损失
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# 添加螺旋性损失
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # 定义距离图
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# 添加N端和C端距离损失
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''添加惩罚N端和C端之间距离的损失'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # 仅考虑最后_binder_len个残基

        # 提取N端（第一个CA原子）和C端（最后一个CA原子）
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # 计算N端和C端之间的距离
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # 使用ELU激活函数计算与阈值距离的偏差
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # 确保损失永远不低于0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # 将损失函数附加到模型回调中
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# 绘制设计轨迹损失图
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # 为每个指标创建新图形
            plt.figure()

            loss = af_model.get_loss(metric)
            # 为迭代创建x轴
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # 添加标签和图例
            plt.xlabel('迭代次数')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # 保存图表
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # 关闭图形
            plt.close()
