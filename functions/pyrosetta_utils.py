####################################
################ PyRosetta功能函数
####################################
### 导入依赖库
import os
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from .generic_utils import clean_pdb
from .biopython_utils import hotspot_residues

# Rosetta界面评分
def score_interface(pdb_file, binder_chain="B"):
    # 加载姿态
    pose = pr.pose_from_pdb(pdb_file)

    # 分析界面统计数据
    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # 用所有氨基酸初始化字典
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # 初始化列表以存储界面处的PDB残基ID
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    # 遍历界面残基
    for pdb_res_num, aa_type in interface_residues_set.items():
        # 增加此氨基酸类型的计数
        interface_AA[aa_type] += 1

        # 将结合剂链和PDB残基编号附加到列表中
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # 计算界面残基数量
    interface_nres = len(interface_residues_pdb_ids)

    # 将列表转换为逗号分隔的字符串
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    # 计算结合剂界面疏水残基的百分比
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # 检索统计数据
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # 形状互补性
    interface_interface_hbonds = interfacescore.interface_hbonds # 界面氢键数量
    interface_dG = iam.get_interface_dG() # 界面dG
    interface_dSASA = iam.get_interface_delta_sasa() # 界面dSASA（界面表面积）
    interface_packstat = iam.get_interface_packstat() # 界面包装统计评分
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # dG/dSASA比率（界面区域大小的标准化能量）
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # 每个界面大小的氢键百分比
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # 未饱和氢键的百分比
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # 计算结合剂能量评分
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # 计算结合剂SASA分数
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # 计算表面疏水性
    binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in zip(range(1, pose.num_chains()+1), pose.split_by_chain())}[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0 
    
    # 计算表面的非极性和芳香族残基
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # 将非极性和芳香族残基计为疏水性
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count

    # 输出界面评分数组和界面处的氨基酸计数
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_fraction': interface_binder_fraction,
    'interface_hydrophobicity': interface_hydrophobicity,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # 四舍五入到两位小数
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

# 对齐PDB以具有相同的方向
def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    # 初始化姿态
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    align = AlignChainMover()
    align.pose(reference_pose)

    # 如果链ID包含逗号，将其分割并只取第一个值
    reference_chain_id = reference_chain_id.split(',')[0]
    align_chain_id = align_chain_id.split(',')[0]

    # 获取姿态中与链ID对应的链编号
    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(reference_chain_id, reference_pose)
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(align_chain_id, align_pose)

    align.source_chain(align_chain)
    align.target_chain(reference_chain)
    align.apply(align_pose)

    # 覆盖对齐的PDB
    align_pose.dump_pdb(align_pdb)
    clean_pdb(align_pdb)

# 计算未对齐的RMSD
def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # 为参考链和对齐链定义链选择器
    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    # 应用选择器获取残基子集
    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    # 将子集转换为残基索引向量
    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)

    # 创建空的子姿态
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    # 填充子姿态
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # 使用RMSDMetric计算RMSD
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)

# 松弛设计的结构
def pr_relax(pdb_file, relaxed_pdb_path):
    if not os.path.exists(relaxed_pdb_path):
        # 生成姿态
        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        ### 生成移动映射
        mmf = MoveMap()
        mmf.set_chi(True) # 启用侧链移动
        mmf.set_bb(True) # 启用主链移动，可以禁用以提高30%的速度，但平均使指标看起来更差
        mmf.set_jump(False) # 禁用整个链的移动

        # 运行FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf) # 设置MoveMap
        fastrelax.max_iter(200) # 默认迭代次数是2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # 将松弛结构对齐到原始轨迹
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # 从start_pose复制B因子到pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # 获取残基中第一个重原子的B因子
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # 输出松弛和对齐的PDB
        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)