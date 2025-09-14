# BindCraft
![alt text](https://github.com/martinpacesa/BindCraft/blob/main/pipeline.png?raw=true)

使用AlphaFold2反向传播、MPNN和PyRosetta的简单结合子设计管道。选择您的目标，让脚本完成其余工作，当您有足够的设计可以订购时就完成了！

[BindCraft预印本链接](https://www.biorxiv.org/content/10.1101/2024.09.30.615802)

**注意：在发布问题之前，请阅读完整的wiki <a href="https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft">这里</a>。wiki中已涵盖的问题将被关闭而不予回答。**

## 安装
首先您需要克隆此存储库。将**[install_folder]**替换为您想要安装的路径。

`git clone https://github.com/martinpacesa/BindCraft [install_folder]`

然后使用*cd*导航到您的安装文件夹并运行安装代码。BindCraft需要兼容CUDA的Nvidia显卡才能运行。在*cuda*设置中，请指定与您的显卡兼容的CUDA版本，例如'11.8'。如果不确定，请留空，但安装可能会选择错误的版本，这会导致错误。在*pkg_manager*中指定您使用的是'mamba'还是'conda'，如果留空则默认使用'conda'。

注意：此安装脚本将安装PyRosetta，商业用途需要许可证。代码需要约2 Mb的存储空间，而AlphaFold2权重占用约5.3 Gb。

`bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'`

## Google Colab
<a href="https://colab.research.google.com/github/martinpacesa/BindCraft/blob/main/notebooks/BindCraft.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <br />
我们准备了一个方便的Google Colab笔记本来测试bindcraft代码功能。但是，由于管道需要大量GPU内存来运行较大的目标+结合子复合物，我们强烈建议使用本地安装和至少32 Gb的GPU内存来运行。

**始终尝试将输入目标PDB修剪到尽可能小的尺寸！这将显著加快结合子生成速度并最小化GPU内存需求。**

**准备好运行至少几百个轨迹才能看到一些被接受的结合子，对于困难的目标可能需要几千个。**

## 本地运行脚本和设置说明
要在本地运行脚本，首先您需要在*settings_target*文件夹中配置您的目标.json文件。json文件中有以下设置：

```
design_path         -> 保存设计和统计数据的路径
binder_name         -> 为设计的结合子文件添加什么前缀
starting_pdb        -> 目标蛋白PDB的路径
chains              -> 要针对蛋白中的哪些链，其余将被忽略
target_hotspot_residues   -> 要针对结合子设计的哪些位置，例如`1,2-10`或链特定`A1-10,B1-20`或整个链`A`，如果要让AF2选择结合位点则设置为null；最好选择多个目标残基或小补丁以减少结合子的搜索空间
lengths             -> 要设计的结合子长度范围
number_of_final_designs   -> 要通过所有过滤器的设计数量目标，如果达到这个数量脚本将停止
```
然后运行结合子设计脚本：

`sbatch ./bindcraft.slurm --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'`

*settings*标志应指向您上面设置的目标.json。*filters*标志指向指定设计过滤器的json（默认为./filters/default_filters.json）。*advanced*标志指向您的高级设置（默认为./advanced_settings/default_4stage_multimer.json）。如果您省略过滤器和高级设置标志，它将自动指向默认值。

或者，如果您的机器不支持SLURM，您可以通过在conda中激活环境并运行python代码直接运行：

```
conda activate BindCraft
cd /path/to/bindcraft/folder/
python -u ./bindcraft.py --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'
```

**我们建议生成至少100个通过所有过滤器的最终设计，然后订购前5-20个进行实验表征。**如果需要高亲和力结合子，最好筛选更多，因为用于排名的ipTM指标不是亲和力的良好预测因子，但已被证明是结合的良好二元预测因子。

以下是各个过滤器和高级设置的说明。

## 高级设置
以下是控制设计过程的高级设置：

```
omit_AAs                        -> 从设计中排除哪些氨基酸（注意：如果位置没有其他选择，它们仍可能出现）
force_reject_AA                 -> 如果设计包含omit_AAs中指定的任何氨基酸，是否强制拒绝设计
design_algorithm                -> 轨迹使用的设计算法，当前实现的算法如下
use_multimer_design             -> 是否使用AF2-ptm或AF2-multimer进行结合子设计；另一个模型将用于验证
num_recycles_design             -> AF2设计的循环次数
num_recycles_validation         -> AF2用于结构预测和验证的循环次数
sample_models = True            -> 是否从AF2模型随机采样参数，建议避免过拟合
rm_template_seq_design          -> 移除目标模板序列进行设计（增加目标灵活性）
rm_template_seq_predict         -> 移除目标模板序列进行重新预测（增加目标灵活性）
rm_template_sc_design           -> 从目标模板移除侧链进行设计
rm_template_sc_predict          -> 从目标模板移除侧链进行重新预测
predict_initial_guess           -> 通过提供结合子原子位置作为预测起点引入偏差。如果MPNN优化后设计失败，建议使用。
predict_bigbang                 -> 将原子位置偏差引入结构模块进行原子初始化。如果目标和设计很大（超过600个氨基酸），建议使用。

# 设计迭代
soft_iterations                 -> 软迭代次数（所有位置考虑所有氨基酸）
temporary_iterations            -> 临时迭代次数（softmax，所有位置考虑最可能的氨基酸）
hard_iterations                 -> 硬迭代次数（one hot编码，所有位置考虑单个氨基酸）
greedy_iterations               -> 从PSSM采样减少损失的随机突变迭代次数
greedy_percentage               -> 每次贪婪迭代期间要突变的蛋白质长度百分比

# 设计权重，较高值对优化参数给予更多权重。
weights_plddt                   -> 设计权重 - 设计链的pLDDT
weights_pae_intra               -> 设计权重 - 设计链内的PAE
weights_pae_inter               -> 设计权重 - 链间PAE
weights_con_intra               -> 设计权重 - 最大化设计链内的接触数
weights_con_inter               -> 设计权重 - 最大化链间接触数
intra_contact_distance          -> 结合子内接触的Cbeta-Cbeta截止距离
inter_contact_distance          -> 结合子和目标间接触的Cbeta-Cbeta截止距离
intra_contact_number            -> 每个接触残基在链内应进行的接触数，不包括直接邻居
inter_contact_number            -> 每个接触残基在链间应进行的接触数
weights_helicity                -> 设计权重 - 设计的螺旋倾向，默认0，负值偏向β折叠
random_helicity                 -> 是否为轨迹随机采样螺旋权重，从-1到1

# 额外损失
use_i_ptm_loss                  -> 使用i_ptm损失优化界面pTM分数？
weights_iptm                    -> 设计权重 - 链间i_ptm
use_rg_loss                     -> 使用回转半径损失？
weights_rg                      -> 设计权重 - 结合子回转半径权重
use_termini_distance_loss       -> 尝试最小化结合子N端和C端之间的距离？对嫁接有帮助
weights_termini_loss            -> 设计权重 - 结合子N端和C端距离最小化权重

# MPNN设置
mpnn_fix_interface              -> 是否修复起始轨迹中设计的界面
num_seqs                        -> 每个结合子要采样和预测的MPNN生成序列数
max_mpnn_sequences              -> 如果多个通过过滤器，每个轨迹要保存的最大MPNN序列数
sampling_temp = 0.1             -> 氨基酸采样温度，T=0.0表示取argmax，T>>1.0表示随机采样。

# MPNN设置 - 高级
backbone_noise                  -> 采样期间的骨架噪声，0.00-0.02是好的值
model_path                      -> MPNN模型权重的路径
mpnn_weights                    -> 是否使用"original"mpnn权重或"soluble"权重
save_mpnn_fasta                 -> 是否将MPNN序列保存为fasta文件，通常不需要，因为序列也在CSV文件中

# AF2设计设置 - 高级
num_recycles_design             -> AF2设计的循环次数
num_recycles_validation         -> AF2用于结构预测和验证的循环次数
optimise_beta                   -> 如果检测到β折叠轨迹，优化预测？
optimise_beta_extra_soft        -> 如果检测到β折叠，添加多少额外软迭代
optimise_beta_extra_temp        -> 如果检测到β折叠，添加多少额外临时迭代
optimise_beta_recycles_design   -> 如果检测到β折叠，设计期间进行多少循环
optimise_beta_recycles_valid    -> 如果检测到β折叠，重新预测期间进行多少循环

# 优化脚本
remove_unrelaxed_trajectory     -> 移除未松弛设计轨迹的PDB文件，保留松弛的PDB
remove_unrelaxed_complex        -> 移除未松弛预测MPNN优化复合物的PDB文件，保留松弛的PDB
remove_binder_monomer           -> 评分后移除预测结合子单体的PDB文件以节省空间
zip_animations                  -> 最后，压缩Animations轨迹文件夹以节省空间
zip_plots                       -> 最后，压缩Plots轨迹文件夹以节省空间
save_trajectory_pickle          -> 保存生成轨迹的pickle文件，注意，占用大量存储空间！
max_trajectories                -> 生成的最大轨迹数，用于基准测试
acceptance_rate                 -> 轨迹中应产生通过过滤器的设计的比例，如果成功设计的比例低于此比例，脚本将停止，您应该调整设计权重
start_monitoring                -> 在多少个轨迹后开始监控acceptance_rate，不要设置太低，可能过早终止

# 调试设置
enable_mpnn = True              -> 是否启用MPNN设计
enable_rejection_check          -> 启用拒绝率检查
```

## 过滤器
以下是您的设计将被过滤的特征，如果您不想使用某些，只需将阈值设置为*null*。*higher*选项表示是否应保留高于阈值（true）或低于（false）的值。以N_开头的特征对应于每个AlphaFold模型的统计，平均值是所有预测模型的平均值。
```
MPNN_score            -> MPNN序列分数，通常不推荐，因为它取决于蛋白质
MPNN_seq_recovery       -> MPNN序列恢复原始轨迹
pLDDT             -> AF2复合物预测的pLDDT置信度分数，标准化为0-1
pTM               -> AF2复合物预测的pTM置信度分数，标准化为0-1
i_pTM             -> AF2复合物预测的界面pTM置信度分数，标准化为0-1
pAE               -> AF2复合物预测的预测对齐误差，与AF2相比标准化为n/31到0-1
i_pAE             -> AF2复合物预测的预测界面对齐误差，与AF2相比标准化为n/31到0-1
i_pLDDT             -> AF2复合物预测的界面pLDDT置信度分数，标准化为0-1
ss_pLDDT            -> AF2复合物预测的二级结构pLDDT置信度分数，标准化为0-1
Unrelaxed_Clashes       -> 松弛前界面冲突数
Relaxed_Clashes         -> 松弛后界面冲突数
Binder_Energy_Score       -> 结合子单独的Rosetta能量分数
Surface_Hydrophobicity      -> 结合子表面疏水性分数
ShapeComplementarity      -> 界面形状互补性
PackStat            -> 界面packstat rosetta分数
dG                -> 界面rosetta dG能量
dSASA             -> 界面delta SASA（大小）
dG/dSASA            -> 界面能量除以界面大小
Interface_SASA_%        -> 界面覆盖的结合子表面分数
Interface_Hydrophobicity        -> 结合子界面界面疏水性分数
n_InterfaceResidues       -> 界面残基数
n_InterfaceHbonds       -> 界面氢键数
InterfaceHbondsPercentage   -> 氢键数与界面大小相比
n_InterfaceUnsatHbonds      -> 界面未满足埋藏氢键数
InterfaceUnsatHbondsPercentage  -> 未满足埋藏氢键数与界面大小相比
Interface_Helix%        -> 界面α螺旋比例
Interface_BetaSheet%      -> 界面β折叠比例
Interface_Loop%         -> 界面环比例
Binder_Helix%         -> 结合子结构α螺旋比例
Binder_BetaSheet%       -> 结合子结构β折叠比例
Binder_Loop%          -> 结合子结构环比例
InterfaceAAs          -> 界面每种氨基酸的数量
HotspotRMSD           -> 结合子与原始轨迹相比的未对齐RMSD，换句话说，重新预测复合物中的结合子距离原始结合位点有多远
Target_RMSD           -> 在设计的结合子背景下预测的目标与输入PDB相比的RMSD
Binder_pLDDT          -> 单独预测的结合子pLDDT置信度分数
Binder_pTM            -> 单独预测的结合子pTM置信度分数
Binder_pAE            -> 单独预测的结合子预测对齐误差
Binder_RMSD           -> 单独预测的结合子与原始轨迹相比的RMSD
```

## 已实现的设计算法
<ul>
 <li>2stage - 使用logits->pssm_semigreedy设计（更快）</li>
 <li>3stage - 使用logits->softmax(logits)->one-hot设计（标准）</li>
 <li>4stage - 使用logits->softmax(logits)->one-hot->pssm_semigreedy设计（默认，广泛）</li>
 <li>greedy - 使用减少损失的随机突变设计（内存密集度较低，较慢，效率较低）</li>
 <li>mcmc - 使用减少损失的随机突变设计，类似于Wicky等人（内存密集度较低，较慢，效率较低）</li>
</ul>

## 已知限制
<ul>
 <li>设置可能不适用于所有目标！可能需要调整迭代次数、设计权重和/或过滤器。目标位点选择也很重要，但如果不指定热点，AF2在检测良好结合位点方面非常出色。</li>
 <li>AF2在预测/设计亲水界面方面比疏水界面差。</li>
 <li>有时轨迹可能最终变形或"压扁"。这对AF2多聚体设计来说是正常的，因为它对序列输入非常敏感，这无法在不重新训练模型的情况下避免。但是这些轨迹会被快速检测并丢弃。</li>
</ul>

## 致谢
感谢Lennart Nickel、Yehlin Cho、Casper Goverde和Sergey Ovchinnikov在编码和讨论想法方面的帮助。此存储库使用以下代码：
<ul>
 <li>Sergey Ovchinnikov的ColabDesign (https://github.com/sokrypton/ColabDesign)</li>
 <li>Justas Dauparas的ProteinMPNN (https://github.com/dauparas/ProteinMPNN)</li>
 <li>PyRosetta (https://github.com/RosettaCommons/PyRosetta.notebooks)</li>
</ul>

