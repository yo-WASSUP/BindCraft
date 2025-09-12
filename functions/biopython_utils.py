####################################
################ BioPython功能函数
####################################
### 导入依赖库
import os
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

# 分析设计序列的组成
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # 检查蛋白质在松弛后是否包含冲突
    if num_clashes > 0:
        note_array.append('松弛结构包含冲突。')

    # 检查序列是否包含不允许的氨基酸
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('包含: '+restricted_AA+'!')

    # 分析蛋白质
    analysis = ProteinAnalysis(sequence)

    # 计算每1%溶液的还原消光系数
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # 检查吸收是否足够高
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'吸收值为 {extinction_coefficient_reduced_1}，考虑在设计中添加色氨酸。')

    # 将注释合并为单个字符串
    notes = ' '.join(note_array)

    return notes

# 临时函数，计算输入PDB和轨迹目标的RMSD
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # 解析PDB文件
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # 从trajectory_pdb提取链A
    chain_trajectory = structure_trajectory[0]['A']
    
    # 从starting_pdb提取指定的链
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # 从trajectory_pdb的链A中提取残基
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # 确保两个结构具有相同数量的残基
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # 从两组残基中收集CA原子
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # 使用结构对齐计算RMSD
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# 检测变形轨迹的C-α冲突
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # 详细的原子信息用于调试和处理

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # 跳过氢原子
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # 排除同一残基内的冲突
        if chain_i == chain_j and res_i == res_j:
            continue

        # 对所有原子排除同一链中直接相邻的残基
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # 如果计算侧链冲突，只考虑不同链之间的冲突
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

# 三字母到单字母氨基酸代码映射表
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# 识别结合剂界面的相互作用残基
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # 解析PDB文件
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # 获取指定的链
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # 获取目标链的原子和坐标
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # 为两条链构建KD树
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # 准备收集相互作用的残基
    interacting_residues = {}

    # 查询树中距离截止值内的原子对
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # 处理每个结合剂原子的相互作用
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # 使用手动字典将三字母代码转换为单字母代码
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

# 计算设计的二级结构百分比
def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # 解析结构
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # 只考虑结构中的第一个模型

    # 为模型计算DSSP
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # 准备计数残基
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # 一次性获取链和相互作用残基
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # 获取二级结构
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # 计算二级结构标准化pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # 计算界面pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # 计算百分比
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

# 计算百分比的辅助函数
def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage