"""Script to train RL agent with RSL-RL."""

import difflib
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import mjlab
import tyro
import warp as wp
import yaml
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPOSITORY_ROOT))

if __name__ == "__main__":
  # Keep TrainConfig identity stable when console modules import scripts.train.
  sys.modules.setdefault("scripts.train", sys.modules[__name__])


@dataclass(frozen=True)
class _KeepTaskDefault:
  def __repr__(self) -> str:
    return "KEEP_TASK_DEFAULT"


KEEP_TASK_DEFAULT = _KeepTaskDefault()


def configure_warp_cuda_compilation() -> None:
  """Compile Warp CUDA kernels as CUBINs instead of driver-JITed PTX."""
  wp.config.cuda_output = "cubin"


class TuningError(ValueError):
  pass


@dataclass(frozen=True)
class TuningEntry:
  value: object
  definition: str
  unit_or_range: str
  effect: str
  dependencies: str
  requires_retraining: bool
  contract_effect: str


def _tuning_entry(
  definition: str,
  unit_or_range: str,
  effect: str,
  dependencies: str,
  *,
  value: object = KEEP_TASK_DEFAULT,
  requires_retraining: bool = True,
  contract_effect: str = "不改变观测、动作或检查点张量维度。",
) -> TuningEntry:
  return TuningEntry(
    value=value,
    definition=definition,
    unit_or_range=unit_or_range,
    effect=effect,
    dependencies=dependencies,
    requires_retraining=requires_retraining,
    contract_effect=contract_effect,
  )


# 快速调参方式：只在“精确路径”条目中增加 value=具体值。
# 保持 value=KEEP_TASK_DEFAULT 时完全沿用注册任务的 baseline；命令行值仍有最高优先级。
# 带 * 的通配规则只负责批量提供中文说明，禁止填写具体值，避免一次误改整类参数。
# 示例：在 agent.algorithm.learning_rate 条目末尾增加 value=5.0e-4。
RL_TUNING: dict[str, dict[str, TuningEntry]] = {
  "training_runtime": {
    # 动作参考文件路径；仅动作跟踪任务使用，速度跟踪任务通常保持为空。
    "motion_file": _tuning_entry(
      "动作参考文件路径。", "文件路径或 None", "决定动作跟踪任务读取哪段参考运动。", "仅 MotionCommand 任务依赖；G1 楼梯速度跟踪不依赖。"
    ),
    # 是否在训练期间录制视频；会增加渲染和磁盘开销。
    "video": _tuning_entry(
      "训练视频开关。", "布尔值", "开启后定期保存训练画面，可能降低训练吞吐。", "依赖渲染后端和可写日志目录。", requires_retraining=False
    ),
    # 每段训练视频包含的环境步数。
    "video_length": _tuning_entry(
      "单段训练视频长度。", "正整数，单位：环境步", "只改变每段视频持续时间和文件大小。", "仅在 video=True 时生效。", requires_retraining=False
    ),
    # 相邻训练视频的触发步数间隔。
    "video_interval": _tuning_entry(
      "训练视频录制间隔。", "正整数，单位：环境步", "值越小录制越频繁、训练开销越高。", "仅在 video=True 时生效。", requires_retraining=False
    ),
    # 是否启用仿真 NaN 诊断保护。
    "enable_nan_guard": _tuning_entry(
      "训练入口 NaN 保护开关。", "布尔值", "开启后异常数值会触发额外诊断数据。", "与 env.sim.nan_guard.* 共同决定输出。", requires_retraining=False
    ),
    # 多 GPU torchrunx 的日志目录。
    "torchrunx_log_dir": _tuning_entry(
      "多进程启动日志目录。", "目录路径、空字符串或 None", "只改变多 GPU 子进程日志位置。", "仅多 GPU torchrunx 模式使用。", requires_retraining=False
    ),
    # 选择 CPU、单 GPU 或多 GPU 训练设备。
    "gpu_ids": _tuning_entry(
      "训练使用的 GPU 编号。", "整数列表、all 或 None", "决定训练设备和并行进程数，不改变优化目标。", "需与 CUDA_VISIBLE_DEVICES 和可用显卡一致。", requires_retraining=False
    ),
    # 预留：只打印最终生效配置，不开始训练。
    "print_effective_config": _tuning_entry(
      "打印最终生效配置并退出。", "布尔值", "用于训练前核对默认值、字典值和 CLI 覆盖。", "Task 4 接入命令行流程。", requires_retraining=False
    ),
    # 预留：把最终生效配置导出为 YAML。
    "dump_effective_config": _tuning_entry(
      "最终生效配置导出路径。", "YAML 文件路径或 None", "便于保存和比较每次调参结果。", "Task 4 接入命令行流程。", requires_retraining=False
    ),
    # 预留：显式允许改变模型张量契约的高风险配置。
    "allow_contract_changes": _tuning_entry(
      "模型契约变更确认开关。", "布尔值", "开启后才允许修改网络或观测结构等不兼容参数。", "禁止用改变契约后的配置直接续训旧检查点。", requires_retraining=False, contract_effect="这是契约保护开关，本身不改变张量维度。"
    ),
  },
  "runner_sampling": {
    # 随机种子同时传给策略与环境，影响可复现实验轨迹。
    "agent.seed": _tuning_entry(
      "训练随机种子。", "整数", "改变初始化、采样和随机化序列。", "比较实验时应固定其余配置并记录种子。"
    ),
    # 每个并行环境一次 PPO rollout 采集的步数。
    "agent.num_steps_per_env": _tuning_entry(
      "每环境 rollout 步数。", "正整数，单位：环境步", "增大会提高单次更新样本量和显存占用。", "与 num_envs、mini-batch 数和控制周期共同决定批量大小。"
    ),
    # PPO 最大策略更新轮数。
    "agent.max_iterations": _tuning_entry(
      "最大训练迭代数。", "正整数", "决定训练时长上限和总采样量。", "应结合收敛曲线、保存间隔和课程进度判断。"
    ),
    # 相邻检查点的训练迭代间隔。
    "agent.save_interval": _tuning_entry(
      "检查点保存间隔。", "正整数，单位：迭代", "值越小保存越密集并增加磁盘占用。", "需保证日志目录空间充足。", requires_retraining=False
    ),
    # 动作输出裁剪范围。
    "agent.clip_actions": _tuning_entry(
      "策略动作裁剪边界。", "正数或 None", "限制送入环境的归一化动作幅值。", "需与动作项 scale、执行器限位及旧检查点分布一致。"
    ),
    # 是否从已有检查点继续训练。
    "agent.resume": _tuning_entry(
      "续训开关。", "布尔值", "开启后加载指定检查点及优化状态。", "必须与 load_run、load_checkpoint 和模型契约匹配。", requires_retraining=False
    ),
    # 续训时选择的运行目录模式。
    "agent.load_run": _tuning_entry(
      "续训运行目录选择器。", "名称、正则表达式或 None", "决定从哪个历史 run 查找检查点。", "仅 resume=True 时使用。", requires_retraining=False
    ),
    # 续训时选择的检查点模式。
    "agent.load_checkpoint": _tuning_entry(
      "续训检查点选择器。", "文件名、正则表达式或 None", "决定加载哪个 model 文件。", "仅 resume=True 时使用，检查点契约必须兼容。", requires_retraining=False
    ),
    # 训练日志后端。
    "agent.logger": _tuning_entry(
      "训练日志后端。", "支持的 logger 名称", "改变指标写入方式，不改变策略目标。", "外部 logger 可能需要账号和网络配置。", requires_retraining=False
    ),
    # 日志根目录中的实验名。
    "agent.experiment_name": _tuning_entry(
      "实验名称。", "非空字符串", "决定日志与检查点所属目录。", "续训查询会依赖此名称。", requires_retraining=False
    ),
    # 单次运行附加名称。
    "agent.run_name": _tuning_entry(
      "运行名称后缀。", "字符串", "帮助区分同一实验下的不同调参运行。", "会加入时间戳日志目录名。", requires_retraining=False
    ),
    # Weights & Biases 项目名。
    "agent.wandb_project": _tuning_entry(
      "W&B 项目名称。", "字符串", "只改变远程实验归档位置。", "仅对应 logger 启用时生效。", requires_retraining=False
    ),
    # Weights & Biases 标签集合。
    "agent.wandb_tags": _tuning_entry(
      "W&B 运行标签。", "字符串列表", "便于筛选和比较实验。", "仅对应 logger 启用时生效。", requires_retraining=False
    ),
    # 是否上传最终模型。
    "agent.upload_model": _tuning_entry(
      "训练后模型上传开关。", "布尔值", "只影响产物发布，不改变学习过程。", "需要已配置的远程日志服务。", requires_retraining=False
    ),
    # Runner 实现类名。
    "agent.class_name": _tuning_entry(
      "训练 Runner 类名。", "已注册类名", "改变 rollout 与更新调度实现。", "必须与 agent 配置字段和 rsl_rl 版本兼容。", contract_effect="可能改变训练运行契约，但不直接改变策略输入输出维度。"
    ),
  },
  "actor_critic": {
    # Actor 与 Critic 的多层感知机宽度。
    "agent.*.hidden_dims": _tuning_entry(
      "网络隐藏层宽度序列。", "正整数元组", "控制模型容量、显存和推理耗时。", "Actor 与 Critic 可分别设置；修改后不能直接复用旧权重。", contract_effect="会改变网络参数形状和检查点契约。"
    ),
    # Actor 与 Critic 的隐藏层激活函数。
    "agent.*.activation": _tuning_entry(
      "网络激活函数名称。", "框架支持的激活函数", "改变网络非线性和优化特性。", "必须被 rsl_rl 模型构造器支持。", contract_effect="会改变模型计算定义，旧检查点仅可按显式兼容策略处理。"
    ),
    # Actor 与 Critic 的模型实现类。
    "agent.*.class_name": _tuning_entry(
      "Actor 或 Critic 模型类名。", "已注册类名", "切换网络实现。", "类必须接受当前观测组与模型配置。", contract_effect="可能改变模型结构、参数名和检查点契约。"
    ),
    # Actor 与 Critic 的观测归一化开关。
    "agent.*.obs_normalization": _tuning_entry(
      "模型侧观测归一化开关。", "布尔值", "影响输入尺度和归一化统计。", "训练、play 与部署必须保持一致。", contract_effect="会改变模型前处理状态和检查点契约。"
    ),
    # 可选 CNN 编码器配置，通常在纯 MLP 速度策略中为空。
    "agent.*.cnn_cfg": _tuning_entry(
      "CNN 编码器配置。", "配置字典或 None", "为图像或栅格观测增加编码器。", "必须与对应观测形状和模型类兼容。", contract_effect="会改变观测编码路径、参数形状和检查点契约。"
    ),
    # Actor 或 Critic 的整体分布配置；None 表示该网络不构造动作分布。
    "agent.*.distribution_cfg": _tuning_entry(
      "动作分布整体配置。", "配置对象或 None", "决定网络是否及如何表达随机动作分布。", "通常只由 Actor 使用，需与 PPO 采样实现匹配。", contract_effect="会改变动作分布状态和检查点契约。"
    ),
    # Actor 动作分布的具体子参数，如初始标准差和标准差形式。
    "agent.*.distribution_cfg.*": _tuning_entry(
      "动作分布子参数。", "由字段定义的标量或类名", "影响初始探索噪声及其参数化方式。", "需与 distribution class、动作维度和旧检查点一致。", contract_effect="可能改变动作分布参数形状或语义。"
    ),
    # Actor 与 Critic 使用的命名观测组。
    "agent.obs_groups.*": _tuning_entry(
      "模型输入观测组列表。", "观测组名称元组", "决定模型接收哪些已注册观测组。", "名称必须存在且最终维度与模型兼容。", contract_effect="会改变模型输入契约和检查点兼容性。"
    ),
  },
  "ppo_algorithm": {
    # PPO 算法实现类。
    "agent.algorithm.class_name": _tuning_entry(
      "PPO 算法类名。", "已注册算法类名", "切换策略更新实现。", "必须与 Runner、模型和配置字段兼容。", contract_effect="可能改变优化状态和检查点契约。"
    ),
    # 单批 rollout 数据重复优化的轮数。
    "agent.algorithm.num_learning_epochs": _tuning_entry(
      "每次更新的学习轮数。", "正整数", "增大可提高样本复用率，也会增加过拟合和 KL 跳变风险。", "与 mini-batch 数、学习率和 desired_kl 联动。"
    ),
    # 每个 epoch 将 rollout 数据切分的份数。
    "agent.algorithm.num_mini_batches": _tuning_entry(
      "每个 epoch 的 mini-batch 数。", "正整数", "增大可降低单批显存但增加优化步数。", "总样本数应能被该值合理切分。"
    ),
    # 优化器基础学习率。
    "agent.algorithm.learning_rate": _tuning_entry(
      "PPO 学习率。", "正有限浮点数", "控制每次梯度更新幅度，过高可能发散，过低则收敛慢。", "与 schedule、epoch、mini-batch、desired_kl 联动。"
    ),
    # 学习率调度策略。
    "agent.algorithm.schedule": _tuning_entry(
      "学习率调度模式。", "框架支持的调度名称", "决定训练期间是否按 KL 等信号调整学习率。", "需与 desired_kl 和算法实现兼容。"
    ),
    # 奖励折扣系数。
    "agent.algorithm.gamma": _tuning_entry(
      "回报折扣因子。", "0 到 1", "越接近 1 越重视长期回报。", "与 episode 长度、控制频率和奖励尺度联动。"
    ),
    # GAE 优势估计系数。
    "agent.algorithm.lam": _tuning_entry(
      "GAE lambda。", "0 到 1", "在优势估计偏差和方差之间权衡。", "与 gamma、rollout 长度共同决定时间信用分配。"
    ),
    # 策略熵奖励系数。
    "agent.algorithm.entropy_coef": _tuning_entry(
      "熵正则系数。", "非负浮点数", "增大可维持探索，但可能妨碍动作收敛。", "与动作分布标准差和奖励总尺度联动。"
    ),
    # 自适应调度使用的目标 KL。
    "agent.algorithm.desired_kl": _tuning_entry(
      "期望 KL 散度。", "正浮点数或 None", "限制新旧策略变化速度。", "主要在 adaptive schedule 下生效。"
    ),
    # 梯度范数裁剪阈值。
    "agent.algorithm.max_grad_norm": _tuning_entry(
      "最大梯度范数。", "正浮点数", "抑制异常大梯度，过小会限制学习。", "与学习率和损失尺度联动。"
    ),
    # Critic 值函数损失权重。
    "agent.algorithm.value_loss_coef": _tuning_entry(
      "值函数损失系数。", "非负浮点数", "控制 Critic 更新相对 Actor 的强度。", "与奖励尺度、值裁剪和学习率联动。"
    ),
    # 是否使用裁剪值函数损失。
    "agent.algorithm.use_clipped_value_loss": _tuning_entry(
      "值函数裁剪开关。", "布尔值", "限制单次 Critic 预测变化。", "裁剪范围复用 PPO clip 参数。"
    ),
    # PPO 策略比率裁剪宽度。
    "agent.algorithm.clip_param": _tuning_entry(
      "PPO 裁剪系数。", "正浮点数", "越小策略更新越保守。", "与学习率、epoch 和目标 KL 联动。"
    ),
    # 是否按 mini-batch 单独归一化优势。
    "agent.algorithm.normalize_advantage_per_mini_batch": _tuning_entry(
      "mini-batch 优势归一化开关。", "布尔值", "改变不同批次的优势尺度和梯度统计。", "与 mini-batch 数和批量大小联动。"
    ),
    # 优化器类型。
    "agent.algorithm.optimizer": _tuning_entry(
      "优化器名称。", "框架支持的优化器", "改变参数更新规则和优化状态。", "需与学习率调度及检查点恢复兼容。", contract_effect="可能改变优化器状态的检查点契约。"
    ),
    # Actor/Critic 是否共享 CNN 编码器。
    "agent.algorithm.share_cnn_encoders": _tuning_entry(
      "CNN 编码器共享开关。", "布尔值", "影响图像特征参数共享和显存。", "仅存在 CNN 编码器时生效。", contract_effect="会改变网络参数共享关系和检查点契约。"
    ),
  },
  "commands": {
    # 速度命令保持后重新采样的时间区间。
    "env.commands.*.resampling_time_range": _tuning_entry(
      "命令重采样时间范围。", "有序二元组，单位：秒", "范围越长，同一目标速度保持越久。", "需结合 episode 长度和速度课程设置。"
    ),
    # 一部分环境采样为原地站立命令的比例。
    "env.commands.*.rel_standing_envs": _tuning_entry(
      "站立环境比例。", "0 到 1", "提高可加强零速稳定训练，但减少移动样本。", "与 heading 环境比例和命令范围共同分配样本。"
    ),
    # 一部分环境启用朝向目标控制的比例。
    "env.commands.*.rel_heading_envs": _tuning_entry(
      "朝向控制环境比例。", "0 到 1", "提高可增加由 heading 转换 yaw 速度的训练样本。", "仅 heading_command=True 时有效。"
    ),
    # 是否将 heading 误差转换为 yaw 速度命令。
    "env.commands.*.heading_command": _tuning_entry(
      "朝向命令开关。", "布尔值", "开启后部分环境按目标朝向生成旋转速度。", "与 rel_heading_envs、heading 范围及控制增益联动。"
    ),
    # 朝向误差到 yaw 速度的比例增益。
    "env.commands.*.heading_control_stiffness": _tuning_entry(
      "朝向控制增益。", "非负浮点数", "越大转向响应越快，过高会造成摆动。", "仅 heading_command=True 时生效，并受 yaw 范围限制。"
    ),
    # episode 初始化时赋予目标速度的概率。
    "env.commands.*.init_velocity_prob": _tuning_entry(
      "初始速度注入概率。", "0 到 1", "改变 reset 后机器人初速度与命令一致的频率。", "会影响起步难度和状态分布。"
    ),
    # 前后线速度采样范围。
    "env.commands.*.ranges.lin_vel_x": _tuning_entry(
      "前后速度命令范围。", "有序二元组，单位：米/秒", "决定训练覆盖的前进和后退速度。", "应与课程阶段、地形难度和执行器能力匹配。"
    ),
    # 左右线速度采样范围。
    "env.commands.*.ranges.lin_vel_y": _tuning_entry(
      "横向速度命令范围。", "有序二元组，单位：米/秒", "决定训练覆盖的左移和右移速度。", "过宽会增加横向平衡难度。"
    ),
    # 绕竖直轴角速度采样范围。
    "env.commands.*.ranges.ang_vel_z": _tuning_entry(
      "偏航角速度命令范围。", "有序二元组，单位：弧度/秒", "决定训练覆盖的左转和右转速率。", "与 heading 控制及足部摩擦能力匹配。"
    ),
    # 目标朝向角采样范围。
    "env.commands.*.ranges.heading": _tuning_entry(
      "目标朝向范围。", "有序二元组，单位：弧度", "决定 heading 模式覆盖的目标方向。", "仅 heading_command=True 时使用。"
    ),
    # 是否显示命令可视化标记。
    "env.commands.*.debug_vis": _tuning_entry(
      "命令调试可视化开关。", "布尔值", "只影响查看器中的目标箭头，不改变训练信号。", "需要启用渲染。", requires_retraining=False
    ),
    # 命令箭头显示比例。
    "env.commands.*.viz.scale": _tuning_entry(
      "命令可视化比例。", "正浮点数", "只改变调试箭头大小。", "仅 debug_vis=True 时可见。", requires_retraining=False
    ),
    # 命令箭头相对机器人高度偏移。
    "env.commands.*.viz.z_offset": _tuning_entry(
      "命令可视化高度偏移。", "浮点数，单位：米", "只改变调试箭头显示位置。", "仅 debug_vis=True 时可见。", requires_retraining=False
    ),
  },
  "rewards": {
    # 每个 reward term 的系数；这是迭代行为时最常调整的入口。
    "env.rewards.*.weight": _tuning_entry(
      "奖励项权重。", "有限浮点数", "绝对值决定该行为目标相对强度，正负决定奖励或惩罚。", "一次只调整一个行为方向，并结合未加权 term 与总回报观察。"
    ),
    # 高斯型跟踪奖励的误差宽度。
    "env.rewards.*.params.std": _tuning_entry(
      "跟踪误差标准差。", "正浮点数", "越小对误差越敏感，奖励峰越窄。", "与对应 reward 权重和命令范围联动。"
    ),
    # 按站立、行走、跑动状态配置的整组关节姿态容差。
    "env.rewards.*.params.std_standing": _tuning_entry(
      "站立姿态关节容差表。", "关节正则表达式到正数的字典", "控制站立时各关节偏离默认姿态的容忍度。", "键必须匹配关节，需与 walking/running 阈值一致。"
    ),
    # 行走状态的整组关节姿态容差。
    "env.rewards.*.params.std_walking": _tuning_entry(
      "行走姿态关节容差表。", "关节正则表达式到正数的字典", "控制行走时各关节动作自由度。", "需与站立、跑动容差及速度阈值一致。"
    ),
    # 跑动状态的整组关节姿态容差。
    "env.rewards.*.params.std_running": _tuning_entry(
      "跑动姿态关节容差表。", "关节正则表达式到正数的字典", "控制高速运动时各关节动作自由度。", "需与 walking/running 阈值和命令上限一致。"
    ),
    # 决定从站立姿态容差切换到行走容差的速度阈值。
    "env.rewards.*.params.walking_threshold": _tuning_entry(
      "行走状态阈值。", "非负速度阈值", "改变姿态奖励选择行走模板的时机。", "与命令速度单位和 running_threshold 保持有序。"
    ),
    # 决定从行走姿态容差切换到跑动容差的速度阈值。
    "env.rewards.*.params.running_threshold": _tuning_entry(
      "跑动状态阈值。", "非负速度阈值", "改变姿态奖励选择跑动模板的时机。", "应大于 walking_threshold 并落在命令范围内。"
    ),
    # 仅在移动命令超过此值时激活的通用门限。
    "env.rewards.*.params.command_threshold": _tuning_entry(
      "奖励激活命令阈值。", "非负浮点数", "避免站立命令下仍强制摆腿或滑移等移动行为。", "必须按对应 command 的速度范数语义设置。"
    ),
    # 足部或接触判定使用的力门限。
    "env.rewards.*.params.force_threshold": _tuning_entry(
      "接触力判定阈值。", "非负数，单位：牛", "改变碰撞或接触被计入奖励的灵敏度。", "需结合接触传感器量程、机器人质量和仿真步长。"
    ),
    # 周期步态奖励的一个完整周期长度。
    "env.rewards.*.params.period": _tuning_entry(
      "步态奖励周期。", "正数，单位：秒", "决定左右腿节律的时间尺度。", "应与 phase 观测 period 和目标行走速度保持一致。"
    ),
    # 周期步态奖励的相位偏移。
    "env.rewards.*.params.offset": _tuning_entry(
      "步态相位偏移。", "标量或固定长度元组", "改变左右足期望接触窗口的相对相位。", "需与腿数量、period 和 gait 实现匹配。"
    ),
    # 足部离地高度的奖励目标。
    "env.rewards.*.params.target_height": _tuning_entry(
      "目标足部离地高度。", "非负数，单位：米", "提高会鼓励更高摆腿，但增加能耗和平衡难度。", "需结合地形台阶、命令激活阈值和足端测量定义。"
    ),
    # 其他奖励数值参数的覆盖模板。
    "env.rewards.*.params.*": _tuning_entry(
      "奖励函数数值参数。", "由当前字段类型和单位定义", "调整该 reward 内部的容差、阈值或目标。", "修改前必须结合对应 reward 函数语义，且一次只验证一个行为问题。"
    ),
  },
  "observations": {
    # 观测组是否把各 term 拼接为一个张量。
    "env.observations.*.concatenate_terms": _tuning_entry(
      "观测项拼接开关。", "布尔值", "决定模型接收拼接张量还是结构化观测。", "必须与 Actor/Critic 模型实现和部署导出一致。", contract_effect="会改变观测数据结构和模型输入契约。"
    ),
    # 观测组拼接使用的维度。
    "env.observations.*.concatenate_dim": _tuning_entry(
      "观测拼接维度。", "整数", "改变各 term 在张量中的拼接轴。", "需与观测张量形状和模型实现一致。", contract_effect="可能改变模型输入张量形状和检查点契约。"
    ),
    # 是否在该观测组启用已配置的 corruption/noise。
    "env.observations.*.enable_corruption": _tuning_entry(
      "观测扰动总开关。", "布尔值", "关闭可用于纯净 play，开启可提高传感器噪声鲁棒性。", "只有配置了 noise 或 delay 的 term 才受影响。"
    ),
    # 观测组保存的历史帧数。
    "env.observations.*.history_length": _tuning_entry(
      "观测组历史长度。", "非负整数，单位：帧", "增加历史信息也增加输入规模与显存。", "需与 term 历史、flatten 方式和策略网络一致。", contract_effect="会改变观测维度和检查点输入契约。"
    ),
    # 是否把观测组的历史维展开到特征维。
    "env.observations.*.flatten_history_dim": _tuning_entry(
      "观测组历史展平开关。", "布尔值", "改变历史帧在最终张量中的排列。", "必须与模型输入处理和部署一致。", contract_effect="会改变观测张量布局和检查点契约。"
    ),
    # 观测组遇到 NaN 时的处理策略。
    "env.observations.*.nan_policy": _tuning_entry(
      "观测 NaN 处理策略。", "支持的策略名称", "决定异常观测是报错、替换还是继续。", "应与 nan_check_per_term 和仿真 NaN guard 联用。", requires_retraining=False
    ),
    # 是否逐 term 检查 NaN。
    "env.observations.*.nan_check_per_term": _tuning_entry(
      "逐观测项 NaN 检查开关。", "布尔值", "开启可定位异常 term，但增加运行开销。", "主要用于调试和验证。", requires_retraining=False
    ),
    # 单个观测项的裁剪范围。
    "env.observations.*.terms.*.clip": _tuning_entry(
      "观测项裁剪范围。", "有序二元组或 None", "限制异常值进入网络，过窄会丢失有效信息。", "需按该 term 单位、scale 和真实传感器范围设置。"
    ),
    # 单个观测项的缩放系数或逐通道缩放表。
    "env.observations.*.terms.*.scale": _tuning_entry(
      "观测项缩放。", "标量、映射或 None", "改变输入数值尺度而不改变物理量。", "训练、play 和部署必须一致；映射键要与通道匹配。"
    ),
    # 单个观测项保留的历史帧数。
    "env.observations.*.terms.*.history_length": _tuning_entry(
      "观测项历史长度。", "非负整数，单位：帧", "增加单项时间信息并可能扩大输入维度。", "与组级 history、flatten 设置和策略结构联动。", contract_effect="可能改变观测维度和模型输入契约。"
    ),
    # 是否把该观测项的历史维展平。
    "env.observations.*.terms.*.flatten_history_dim": _tuning_entry(
      "观测项历史展平开关。", "布尔值", "改变该 term 历史张量布局。", "与组级拼接方式和模型输入一致。", contract_effect="可能改变观测张量布局和检查点契约。"
    ),
    # 观测延迟的最小离散滞后。
    "env.observations.*.terms.*.delay_min_lag": _tuning_entry(
      "最小观测延迟。", "非负整数，单位：仿真步", "模拟传感器至少延迟多少步。", "必须不大于 delay_max_lag，并结合控制频率换算。"
    ),
    # 观测延迟的最大离散滞后。
    "env.observations.*.terms.*.delay_max_lag": _tuning_entry(
      "最大观测延迟。", "非负整数，单位：仿真步", "提高可增强延迟鲁棒性但增加控制难度。", "必须不小于 delay_min_lag。"
    ),
    # 延迟是否为每个并行环境独立采样。
    "env.observations.*.terms.*.delay_per_env": _tuning_entry(
      "逐环境延迟开关。", "布尔值", "开启后各环境具有不同延迟，增加随机化多样性。", "与延迟范围和更新周期联动。"
    ),
    # 各环境延迟更新是否使用不同相位。
    "env.observations.*.terms.*.delay_per_env_phase": _tuning_entry(
      "逐环境延迟更新相位开关。", "布尔值", "错开延迟重采样时刻，减少同步变化。", "仅延迟随机化启用时有意义。"
    ),
    # 延迟值重新采样的周期。
    "env.observations.*.terms.*.delay_update_period": _tuning_entry(
      "观测延迟更新周期。", "正整数，单位：步", "越小延迟变化越频繁。", "需结合 delay_hold_prob 和控制周期设置。"
    ),
    # 延迟更新时保持旧延迟的概率。
    "env.observations.*.terms.*.delay_hold_prob": _tuning_entry(
      "延迟保持概率。", "0 到 1", "越高延迟状态持续越久。", "与更新周期和逐环境相位设置联动。"
    ),
    # 均匀噪声的下界。
    "env.observations.*.terms.*.noise.n_min": _tuning_entry(
      "观测噪声下界。", "有限数值或逐通道范围", "决定加入该 term 的最小随机扰动。", "应与 n_max 有序并按缩放前单位设置。"
    ),
    # 均匀噪声的上界。
    "env.observations.*.terms.*.noise.n_max": _tuning_entry(
      "观测噪声上界。", "有限数值或逐通道范围", "决定加入该 term 的最大随机扰动。", "应不小于 n_min，并参考真实传感器误差。"
    ),
    # 噪声与原始观测的组合操作。
    "env.observations.*.terms.*.noise.operation": _tuning_entry(
      "观测噪声操作。", "add、scale 等支持值", "决定噪声是相加还是按比例作用。", "必须与 n_min/n_max 的物理含义一致。"
    ),
    # phase 等观测函数的数值参数。
    "env.observations.*.terms.*.params.period": _tuning_entry(
      "观测项周期参数。", "正数，单位：秒", "改变相位观测的时间尺度。", "应与步态 reward period 保持一致。", contract_effect="数值维度不变，但会改变相位语义，训练和部署必须一致。"
    ),
    # 传感器本身的直接运行参数，如量程、历史长度或归约方式。
    "env.scene.sensors.*.*": _tuning_entry(
      "场景传感器参数。", "由当前字段类型与传感器定义决定", "改变射线、接触等传感器的采样和输出行为。", "选择器字段由只读规则排除；结构字段改变时需重新训练。", contract_effect="部分字段可能改变传感器输出结构和模型观测契约。"
    ),
    # 射线传感器的扫描图案参数。
    "env.scene.sensors.*.pattern.*": _tuning_entry(
      "射线扫描图案参数。", "方向、尺寸或分辨率", "改变地形扫描覆盖范围和射线数量。", "分辨率或尺寸变化可能改变 height scan 维度。", contract_effect="可能改变扫描点数量、观测维度和检查点契约。"
    ),
    # 传感器调试图形参数。
    "env.scene.sensors.*.viz.*": _tuning_entry(
      "传感器可视化参数。", "颜色、尺寸、长度或布尔值", "只改变调试显示，不改变传感器数值。", "需要启用相应 debug_vis 和查看器。", requires_retraining=False
    ),
  },
  "terrain_curriculum": {
    # 并行仿真环境数量，是训练吞吐与显存占用的主要开关。
    "env.scene.num_envs": _tuning_entry(
      "并行环境数量。", "正整数", "增大可提高每次采样量和 GPU 利用率，也显著增加显存。", "与 rollout 步数、mini-batch 数和设备容量联动。"
    ),
    # 各并行环境在世界坐标中的间距。
    "env.scene.env_spacing": _tuning_entry(
      "环境间距。", "正数，单位：米", "过小可能使不同环境的几何或传感器互相干扰。", "应大于单个地形块与机器人活动范围。"
    ),
    # 场景空间范围提示。
    "env.scene.extent": _tuning_entry(
      "场景范围。", "正数，单位：米", "影响场景资源和可视化的空间尺度。", "需覆盖环境布局和地形尺寸。"
    ),
    # 地形实体类型，如生成地形或平面。
    "env.scene.terrain.terrain_type": _tuning_entry(
      "地形类型。", "框架支持的地形类型", "决定训练使用平面、生成地形或其他地形源。", "必须与 terrain_generator 和场景构造兼容。"
    ),
    # 地形实体内部的环境数量。
    "env.scene.terrain.num_envs": _tuning_entry(
      "地形承载环境数量。", "正整数", "决定生成地形为多少环境分配位置。", "通常应与 env.scene.num_envs 一致。"
    ),
    # 地形实体内部的环境间距。
    "env.scene.terrain.env_spacing": _tuning_entry(
      "地形环境间距。", "正数，单位：米", "决定地形上的环境原点间隔。", "通常与 scene.env_spacing 和单块地形尺寸一致。"
    ),
    # reset 时允许抽取的最高初始地形等级。
    "env.scene.terrain.max_init_terrain_level": _tuning_entry(
      "最大初始地形等级。", "非负整数或 None", "提高会让机器人更早从困难地形开始。", "需小于地形行数并结合课程进度。"
    ),
    # 是否启用按表现升降地形等级的课程。
    "env.scene.terrain.terrain_generator.curriculum": _tuning_entry(
      "地形课程开关。", "布尔值", "开启后地形难度可随训练表现变化。", "需配置课程 term、难度范围和足够的地形行数。"
    ),
    # 地形难度采样范围。
    "env.scene.terrain.terrain_generator.difficulty_range": _tuning_entry(
      "地形难度范围。", "0 到 1 内的有序二元组", "决定生成器从简单到困难参数的覆盖范围。", "与 curriculum、行数及各子地形范围共同作用。"
    ),
    # 地形网格行数，通常对应难度离散等级。
    "env.scene.terrain.terrain_generator.num_rows": _tuning_entry(
      "地形行数。", "正整数", "增大可提供更细的难度等级，但增加生成和内存开销。", "max_init_terrain_level 必须落在有效行内。"
    ),
    # 地形网格列数，通常对应类型采样槽位。
    "env.scene.terrain.terrain_generator.num_cols": _tuning_entry(
      "地形列数。", "正整数", "增大可提高各子地形分布的空间采样量。", "与 num_envs、proportion 和地形尺寸联动。"
    ),
    # 单个地形块的长宽。
    "env.scene.terrain.terrain_generator.size": _tuning_entry(
      "单块地形尺寸。", "正二元组，单位：米", "决定每个环境可用的连续运动空间。", "需覆盖 episode 内可能位移并与边界宽度匹配。"
    ),
    # 生成地形外侧安全边界宽度。
    "env.scene.terrain.terrain_generator.border_width": _tuning_entry(
      "地形边界宽度。", "非负数，单位：米", "扩大可减少机器人越界后立即离开地形。", "会增加总地形面积和内存。"
    ),
    # 生成地形边界高度。
    "env.scene.terrain.terrain_generator.border_height": _tuning_entry(
      "地形边界高度。", "浮点数，单位：米", "决定地形块外圈基准高度。", "需与子地形高度和碰撞定义一致。"
    ),
    # 地形生成随机种子。
    "env.scene.terrain.terrain_generator.seed": _tuning_entry(
      "地形随机种子。", "整数或 None", "改变具体台阶、坡面和粗糙地形样本。", "复现实验时固定该值及训练 seed。"
    ),
    # 地形颜色映射方式。
    "env.scene.terrain.terrain_generator.color_scheme": _tuning_entry(
      "地形着色方案。", "支持的枚举值", "主要改变可视化颜色，也可能按高度着色。", "不应作为行为训练变量。", requires_retraining=False
    ),
    # 是否为生成地形附加灯光。
    "env.scene.terrain.terrain_generator.add_lights": _tuning_entry(
      "地形灯光生成开关。", "布尔值", "只影响渲染场景灯光。", "仅可视化或图像观测任务相关。", requires_retraining=False
    ),
    # 各子地形在采样集合中的相对比例。
    "env.scene.terrain.terrain_generator.sub_terrains.*.proportion": _tuning_entry(
      "子地形采样比例。", "0 到 1 的非负数", "提高某类比例会增加其训练样本。", "所有子地形比例需共同形成有效概率分布。"
    ),
    # 楼梯子地形的台阶高度范围。
    "env.scene.terrain.terrain_generator.sub_terrains.*.step_height_range": _tuning_entry(
      "台阶高度范围。", "正且有序的二元组，单位：米", "决定上楼或下楼训练覆盖的台阶高度。", "需与步态能力、地形难度和足部目标行为匹配。"
    ),
    # 楼梯子地形的踏面宽度。
    "env.scene.terrain.terrain_generator.sub_terrains.*.step_width": _tuning_entry(
      "台阶踏面宽度。", "正数，单位：米", "越窄对落脚位置和稳定性要求越高。", "应大于可实现的足长与定位误差。"
    ),
    # 其他子地形生成参数，如坡度、噪声、平台和网格尺度。
    "env.scene.terrain.terrain_generator.sub_terrains.*.*": _tuning_entry(
      "子地形生成参数。", "由当前字段类型和单位决定", "改变对应地形的几何形状、分辨率或难度。", "需结合 proportion、difficulty_range 和 MuJoCo 接触稳定性验证。"
    ),
    # 速度课程每个阶段开始生效的训练步。
    "env.curriculum.*.params.velocity_stages.*.step": _tuning_entry(
      "速度课程阶段起点。", "非负整数，单位：训练步或迭代定义", "决定何时扩大命令速度范围。", "各阶段 step 必须递增并与总训练轮数匹配。"
    ),
    # 速度课程阶段内的线速度或角速度范围。
    "env.curriculum.*.params.velocity_stages.*.*": _tuning_entry(
      "速度课程阶段范围。", "有序二元组，单位随命令字段", "决定该阶段允许采样的速度区间。", "必须落在最终命令能力边界内并与阶段顺序一致。"
    ),
  },
  "domain_randomization": {
    # 事件触发模式，如启动、reset 或固定间隔。
    "env.events.*.mode": _tuning_entry(
      "随机化事件模式。", "框架支持的事件模式", "决定随机化在何时执行。", "必须与 interval、reset 流程及事件函数兼容。"
    ),
    # 周期事件的触发时间范围。
    "env.events.*.interval_range_s": _tuning_entry(
      "事件时间间隔范围。", "正且有序的二元组或 None，单位：秒", "范围越小随机扰动越频繁。", "仅 interval 模式使用，并需结合 episode 长度。"
    ),
    # 事件计时是否在所有环境间共享。
    "env.events.*.is_global_time": _tuning_entry(
      "事件全局计时开关。", "布尔值", "开启会使触发依据全局时间，关闭可按环境独立计时。", "与事件 mode 和 reset 方式联动。"
    ),
    # reset 后事件再次触发前要求的最少步数。
    "env.events.*.min_step_count_between_reset": _tuning_entry(
      "事件最小重置间隔。", "非负整数，单位：环境步", "避免扰动在刚 reset 后立即重复发生。", "与 interval 和控制周期共同决定实际等待时间。"
    ),
    # reset 位姿、速度、摩擦、编码器偏差和 COM 等事件参数。
    "env.events.*.params.*": _tuning_entry(
      "域随机化事件参数。", "由当前字段类型与事件函数定义", "改变随机化范围、运算方式或共享策略。", "必须按对应物理单位设置；asset_cfg 选择器由只读规则排除。"
    ),
    # 嵌套事件范围的单轴或列表元素。
    "env.events.*.params.*.*": _tuning_entry(
      "域随机化分量参数。", "有限数值或有序范围", "改变某一位置、速度或质心轴的随机化幅度。", "需与机器人能力、接触稳定性和其他轴范围联合验证。"
    ),
  },
  "actions_terminations": {
    # 动作项输出裁剪范围。
    "env.actions.*.clip": _tuning_entry(
      "动作项裁剪范围。", "标量、范围映射或 None", "限制策略动作在转换前的幅度。", "需与 runner clip_actions、scale 和关节限位共同设置。"
    ),
    # 从归一化策略动作到执行器目标的缩放。
    "env.actions.*.scale": _tuning_entry(
      "动作缩放。", "标量或关节正则表达式映射", "决定单位策略输出对应的关节目标幅度。", "训练、play 和部署必须一致，映射键应覆盖预期关节。", contract_effect="缩放形状或通道映射变化可能改变动作契约。"
    ),
    # 动作转换使用的固定偏移。
    "env.actions.*.offset": _tuning_entry(
      "动作偏移。", "标量、映射或 None", "平移策略动作对应的执行器目标中心。", "与 use_default_offset 和机器人默认关节姿态联动。"
    ),
    # 是否使用资产默认关节位置作为动作偏移。
    "env.actions.*.use_default_offset": _tuning_entry(
      "默认姿态偏移动作开关。", "布尔值", "开启后零动作对应机器人默认关节姿态。", "需与 offset、资产初始姿态和部署控制器一致。"
    ),
    # 终止项是否按超时而不是失败终止处理。
    "env.terminations.*.time_out": _tuning_entry(
      "终止项超时标记。", "布尔值", "影响 bootstrap 与 episode 统计语义。", "仅真正的时间上限应标记为超时。"
    ),
    # 倾倒角度等终止阈值。
    "env.terminations.*.params.*": _tuning_entry(
      "终止条件参数。", "由终止函数定义的有限数值", "改变 episode 被判定失败或结束的边界。", "需结合机器人正常运动范围和奖励统计设置。"
    ),
  },
  "simulation": {
    # 每次策略动作保持的 MuJoCo 仿真步数。
    "env.decimation": _tuning_entry(
      "控制降采样倍数。", "正整数", "与 timestep 相乘得到策略控制周期。", "改变后需重新评估控制频率、奖励 dt 缩放和部署周期。"
    ),
    # 单个 episode 的最长物理时间。
    "env.episode_length_s": _tuning_entry(
      "回合时长。", "正数，单位：秒", "决定超时前可采集的最长轨迹。", "与控制周期、gamma、课程和命令重采样时间联动。"
    ),
    # 是否把时间上限视为真正终止。
    "env.is_finite_horizon": _tuning_entry(
      "有限时域开关。", "布尔值", "改变超时后的价值 bootstrap 语义。", "应与 time_out termination 和算法回报计算一致。"
    ),
    # 是否按环境时间步长缩放 reward。
    "env.scale_rewards_by_dt": _tuning_entry(
      "奖励按时间缩放开关。", "布尔值", "开启可使不同控制频率下每秒奖励尺度更一致。", "改变 decimation 或 timestep 时必须一并检查。"
    ),
    # 环境随机种子，训练入口通常会同步为 agent.seed。
    "env.seed": _tuning_entry(
      "环境随机种子。", "整数", "改变 reset、地形和域随机化序列。", "run_train 会按训练进程同步 agent.seed。"
    ),
    # 仿真层直接容量和并行设置。
    "env.sim.*": _tuning_entry(
      "仿真容量或并行参数。", "正整数、布尔值或当前字段类型", "影响接触缓存容量、并行求解或运行稳定性。", "过小可能丢失接触，过大增加内存；嵌套 mujoco/nan_guard 使用更具体规则。"
    ),
    # MuJoCo 时间步、重力、积分器、求解器和容差等选项。
    "env.sim.mujoco.*": _tuning_entry(
      "MuJoCo 求解参数。", "由 MuJoCo 字段定义的数值或枚举", "改变动力学精度、接触收敛、速度和物理行为。", "timestep 与 decimation 决定控制周期；求解设置需做稳定性验证。"
    ),
    # NaN 保护的开关、缓存、输出路径和导出环境数。
    "env.sim.nan_guard.*": _tuning_entry(
      "仿真 NaN 保护参数。", "布尔值、正整数或目录路径", "只改变异常检测和诊断数据容量。", "与训练入口 enable_nan_guard 共同使用。", requires_retraining=False
    ),
    # 机器人执行器的刚度、阻尼、惯量、摩擦和力矩上限。
    "env.scene.entities.*.articulation.actuators.*.*": _tuning_entry(
      "执行器物理参数。", "非负数或关节正则映射", "改变关节控制响应、耗散和最大输出能力。", "需与真实 G1 参数、动作缩放和仿真稳定性一致。"
    ),
    # 软关节位置限位缩放。
    "env.scene.entities.*.articulation.soft_joint_pos_limit_factor": _tuning_entry(
      "软关节限位系数。", "0 到 1", "越小策略可用关节范围越保守。", "需与机器人硬限位、动作 scale 和终止条件一致。"
    ),
    # 碰撞几何的摩擦、求解、接触维度和优先级。
    "env.scene.entities.*.collisions.*.*": _tuning_entry(
      "碰撞物理参数。", "MuJoCo 接触字段对应的数值或映射", "改变足地摩擦与碰撞约束求解行为。", "几何选择器由只读规则排除，数值需与地形材料共同验证。"
    ),
    # 资产默认初始姿态和速度。
    "env.scene.entities.*.init_state.*": _tuning_entry(
      "实体初始状态。", "位置、旋转、速度或关节映射", "改变构建或 reset 时的默认状态。", "域随机化应优先通过 reset event 叠加，且姿态必须物理可行。"
    ),
  },
  "viewer_video": {
    # 查看器相机、显示环境、分辨率、阴影和反射等直接参数。
    "env.viewer.*": _tuning_entry(
      "MuJoCo 查看器参数。", "由字段定义的相机数值、布尔值或分辨率", "只改变交互查看和视频画面，不改变策略训练目标。", "entity/body 选择器由只读规则排除。", requires_retraining=False
    ),
  },
}


RL_READ_ONLY: dict[str, str] = {
  # 场景构建回调决定模型拓扑，只能由任务配置代码提供。
  "env.scene.spec_fn": "场景构建回调属于结构定义，不能作为数值调参项。",
  # 命令绑定的场景实体是任务接线，不属于数值调参。
  "env.commands.*.entity_name": "命令实体选择器属于任务通信接线，只读。",
  # 课程绑定的命令名称必须与任务注册项一致。
  "env.curriculum.*.params.command_name": "课程命令名称属于注册表选择器，只读。",
  # 空奖励参数容器只用于函数调用协议，不能整体替换。
  "env.rewards.*.params": "空奖励参数容器属于调用结构，只读。",
  # 奖励所读取的命令名称属于任务接线。
  "env.rewards.*.params.command_name": "奖励命令选择器属于任务接线，只读。",
  # 奖励所读取的传感器名称属于任务接线。
  "env.rewards.*.params.sensor_name": "奖励传感器选择器属于任务接线，只读。",
  # 奖励资产选择器决定实体、关节和几何拓扑。
  "env.rewards.*.params.asset_cfg.*": "奖励资产选择器属于实体拓扑，只读。",
  # 未配置的观测噪声对象不能通过标量入口凭空构造。
  "env.observations.*.terms.*.noise": "空观测噪声对象需要配置类构造，只读。",
  # 空观测参数容器只用于函数调用协议。
  "env.observations.*.terms.*.params": "空观测参数容器属于调用结构，只读。",
  # 观测读取的命令名称属于任务接线。
  "env.observations.*.terms.*.params.command_name": "观测命令选择器属于任务接线，只读。",
  # 观测读取的传感器名称属于任务接线。
  "env.observations.*.terms.*.params.sensor_name": "观测传感器选择器属于任务接线，只读。",
  # 观测资产选择器决定实体和关节拓扑。
  "env.observations.*.terms.*.params.asset_cfg.*": "观测资产选择器属于实体拓扑，只读。",
  # 随机化事件的资产选择器决定作用对象。
  "env.events.*.params.asset_cfg.*": "事件资产选择器属于实体拓扑，只读。",
  # 空指标参数容器不提供数值调节项。
  "env.metrics.*.params": "空指标参数容器属于统计调用结构，只读。",
  # 空终止参数容器不提供数值调节项。
  "env.terminations.*.params": "空终止参数容器属于终止函数调用结构，只读。",
  # 动作实体名称决定输出路由。
  "env.actions.*.entity_name": "动作实体选择器属于输出通信接线，只读。",
  # 动作执行器名称决定动作维度和关节映射。
  "env.actions.*.actuator_names": "动作执行器选择器会改变动作拓扑，只读。",
  # 动作顺序开关决定通道排列，不允许作为普通数值项切换。
  "env.actions.*.preserve_order": "动作通道顺序属于模型契约，只读。",
  # 动作传动类型决定执行器映射结构。
  "env.actions.*.transmission_type": "动作传动类型属于执行结构，只读。",
  # 资产附属相机、灯光、材质和纹理容器属于场景拓扑。
  "env.scene.entities.*.cameras": "实体相机容器属于场景拓扑，只读。",
  "env.scene.entities.*.lights": "实体灯光容器属于场景拓扑，只读。",
  "env.scene.entities.*.materials": "实体材质容器属于场景拓扑，只读。",
  "env.scene.entities.*.textures": "实体纹理容器属于场景拓扑，只读。",
  # 执行器排序决定动作通道映射。
  "env.scene.entities.*.sort_actuators": "执行器排序属于动作通道契约，只读。",
  # 执行器目标表达式决定关节集合。
  "env.scene.entities.*.articulation.actuators.*.target_names_expr": "执行器关节选择表达式属于动作拓扑，只读。",
  # 执行器传动类型决定控制映射实现。
  "env.scene.entities.*.articulation.actuators.*.transmission_type": "执行器传动类型属于控制结构，只读。",
  # 碰撞几何表达式和排他开关会改变几何拓扑。
  "env.scene.entities.*.collisions.*.geom_names_expr": "碰撞几何选择器属于场景拓扑，只读。",
  "env.scene.entities.*.collisions.*.disable_other_geoms": "碰撞几何排他开关会改变场景拓扑，只读。",
  # 传感器名称、坐标框架和几何组是任务接线。
  "env.scene.sensors.*.name": "传感器注册名称属于任务接线，只读。",
  "env.scene.sensors.*.frame.*": "传感器坐标框架属于实体接线，只读。",
  "env.scene.sensors.*.include_geom_groups": "传感器几何组选择器属于场景拓扑，只读。",
  # 接触传感器主次对象选择器决定采样实体集合。
  "env.scene.sensors.*.primary.*": "接触传感器主对象选择器属于实体拓扑，只读。",
  "env.scene.sensors.*.secondary.*": "接触传感器次对象选择器属于实体拓扑，只读。",
  # 地形实体的关节、相机、碰撞和初态容器不是训练数值入口。
  "env.scene.terrain.articulation": "地形关节容器属于场景拓扑，只读。",
  "env.scene.terrain.cameras": "地形相机容器属于场景拓扑，只读。",
  "env.scene.terrain.collisions": "地形碰撞容器属于场景拓扑，只读。",
  "env.scene.terrain.sort_actuators": "地形执行器排序属于场景拓扑，只读。",
  "env.scene.terrain.init_state.*": "地形实体初始状态属于场景结构，只读。",
  # 平坦区域采样配置需要专用配置对象，不能作为标量整体替换。
  "env.scene.terrain.terrain_generator.sub_terrains.*.flat_patch_sampling": "平坦区域采样器属于地形结构，只读。",
  # 灯光、材质和纹理定义只影响场景资产，不属于 RL 数值调参。
  "env.scene.terrain.lights.*.*": "地形灯光定义属于渲染资产，只读。",
  "env.scene.terrain.materials.*.*": "地形材质定义属于渲染资产，只读。",
  "env.scene.terrain.textures.*.*": "地形纹理定义属于渲染资产，只读。",
  # 查看器实体和刚体名称是相机跟随选择器。
  "env.viewer.entity_name": "查看器实体选择器只影响接线，只读。",
  "env.viewer.body_name": "查看器刚体选择器只影响接线，只读。",
}


@dataclass(frozen=True)
class ParameterRecord:
  path: str
  default_value: object
  value_type: str
  editable: bool
  read_only_reason: str | None
  source: str
  category: str | None = None
  rule_pattern: str | None = None
  tuning_entry: TuningEntry | None = None
  tuning_value: object = KEEP_TASK_DEFAULT
  profile_value: object = KEEP_TASK_DEFAULT
  cli_value: object = KEEP_TASK_DEFAULT
  final_value: object = KEEP_TASK_DEFAULT


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  print_effective_config: bool = False
  dump_effective_config: str | None = None
  allow_contract_changes: bool = False

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


INSPECTION_CONTROL_PATHS = frozenset(
  {
    "print_effective_config",
    "dump_effective_config",
    "allow_contract_changes",
  }
)


@dataclass(frozen=True)
class PreparedTrainConfig:
  task_id: str
  baseline_cfg: TrainConfig
  tuned_cfg: TrainConfig
  cfg: TrainConfig
  catalog: dict[str, ParameterRecord]
  tuning_diff: dict[str, dict[str, object]]
  profile_snapshot: dict[str, object] | None
  profile_path: str | None
  profile_revision: str | None
  profile_diff: dict[str, dict[str, object]]
  cli_diff: dict[str, dict[str, object]]

  def dump_if_requested(self) -> None:
    output_path = self.cfg.dump_effective_config
    if output_path is None:
      return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
      yaml.safe_dump(
        render_effective_config(self), sort_keys=False, allow_unicode=True
      ),
      encoding="utf-8",
    )


def _qualified_name(value: object) -> str:
  module = getattr(value, "__module__", None)
  qualified = getattr(value, "__qualname__", None)
  if module and qualified:
    return f"{module}.{qualified}"
  value_type = type(value)
  return f"{value_type.__module__}.{value_type.__qualname__}"


def stable_config_value(value: object, _seen: set[int] | None = None) -> object:
  """Convert a configuration value into deterministic YAML-safe primitives."""
  if value is KEEP_TASK_DEFAULT:
    return "KEEP_TASK_DEFAULT"
  if isinstance(value, Enum):
    return stable_config_value(value.value, _seen)
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, type) or callable(value):
    return _qualified_name(value)

  seen = _seen if _seen is not None else set()
  identity = id(value)
  if identity in seen:
    return f"<cycle:{_qualified_name(value)}>"
  seen.add(identity)
  try:
    if is_dataclass(value) and not isinstance(value, type):
      return {
        item.name: stable_config_value(getattr(value, item.name), seen)
        for item in fields(value)
      }
    if isinstance(value, Mapping):
      return {
        str(key): stable_config_value(value[key], seen)
        for key in sorted(value, key=str)
      }
    if isinstance(value, (tuple, list)):
      return [stable_config_value(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
      converted = [stable_config_value(item, seen) for item in value]
      return sorted(converted, key=repr)
    return f"<{_qualified_name(value)}>"
  finally:
    seen.remove(identity)


def _is_atomic_component(value: object) -> bool:
  def is_scalar(item: object) -> bool:
    return item is None or isinstance(item, (bool, int, float, str, Enum, Path))

  def is_flat_value(item: object) -> bool:
    return is_scalar(item) or _is_atomic_component(item)

  if is_scalar(value):
    return True
  if isinstance(value, (tuple, list)):
    return all(is_scalar(item) for item in value)
  if isinstance(value, Mapping):
    if not value:
      return True
    regex_marker = re.compile(r"[.*+?\[\](){}|^$\\]")
    has_pattern_key = any(
      isinstance(key, str) and regex_marker.search(key) for key in value
    )
    return has_pattern_key and all(
      is_flat_value(item) for item in value.values()
    )
  return False


def _is_atomic_value(value: object) -> bool:
  return _is_atomic_component(value)


def _is_private_path(path: Sequence[str | int]) -> bool:
  return any(isinstance(token, str) and token.startswith("_") for token in path)


def _declared_type_is_structural(declared_type: object | None) -> bool:
  if declared_type is None:
    return False
  origin = get_origin(declared_type)
  if origin in (Callable, type):
    return True
  if declared_type in (Callable, type):
    return True
  return any(_declared_type_is_structural(item) for item in get_args(declared_type))


def _format_parameter_path(path: Sequence[str | int]) -> str:
  return ".".join(str(token) for token in path)


def _record_parameter(
  value: object,
  path: Sequence[str | int],
  records: dict[str, ParameterRecord],
  *,
  editable: bool,
  read_only_reason: str | None,
) -> None:
  parameter_path = _format_parameter_path(path)
  records[parameter_path] = ParameterRecord(
    path=parameter_path,
    default_value=value,
    value_type=_qualified_name(type(value)),
    editable=editable,
    read_only_reason=read_only_reason,
    source=_qualified_name(value),
  )


def _discover_value(
  value: object,
  path: Sequence[str | int],
  records: dict[str, ParameterRecord],
  seen: set[int],
  declared_type: object | None = None,
) -> None:
  if _is_private_path(path):
    _record_parameter(
      value,
      path,
      records,
      editable=False,
      read_only_reason="private runtime state",
    )
    return
  if _declared_type_is_structural(declared_type):
    _record_parameter(
      value,
      path,
      records,
      editable=False,
      read_only_reason="callable, class, or factory field",
    )
    return
  if isinstance(value, type) or callable(value):
    _record_parameter(
      value,
      path,
      records,
      editable=False,
      read_only_reason="callable or class",
    )
    return
  if _is_atomic_value(value):
    _record_parameter(value, path, records, editable=True, read_only_reason=None)
    return

  identity = id(value)
  if identity in seen:
    _record_parameter(
      value,
      path,
      records,
      editable=False,
      read_only_reason="cyclic configuration reference",
    )
    return

  if is_dataclass(value) and not isinstance(value, type):
    seen.add(identity)
    try:
      try:
        type_hints = get_type_hints(type(value))
      except (NameError, TypeError):
        type_hints = {}
      for item in fields(value):
        _discover_value(
          getattr(value, item.name),
          (*path, item.name),
          records,
          seen,
          type_hints.get(item.name, item.type),
        )
    finally:
      seen.remove(identity)
    return

  if isinstance(value, Mapping):
    seen.add(identity)
    try:
      for key, item in value.items():
        _discover_value(item, (*path, str(key)), records, seen)
    finally:
      seen.remove(identity)
    return

  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    seen.add(identity)
    try:
      for index, item in enumerate(value):
        _discover_value(item, (*path, index), records, seen)
    finally:
      seen.remove(identity)
    return

  _record_parameter(
    value,
    path,
    records,
    editable=False,
    read_only_reason="opaque runtime object",
  )


def discover_parameters(cfg: TrainConfig) -> dict[str, ParameterRecord]:
  """Discover stable dotted paths for every configuration leaf."""
  records: dict[str, ParameterRecord] = {}
  _discover_value(cfg, (), records, set(), TrainConfig)
  return dict(sorted(records.items()))


def _path_pattern_matches(pattern: str, path: str) -> bool:
  pattern_tokens = pattern.split(".")
  path_tokens = path.split(".")
  return len(pattern_tokens) == len(path_tokens) and all(
    expected == "*" or expected == actual
    for expected, actual in zip(pattern_tokens, path_tokens, strict=True)
  )


def _pattern_specificity(pattern: str) -> int:
  return sum(token != "*" for token in pattern.split("."))


def _choose_pattern(path: str, patterns: Sequence[str]) -> str | None:
  matches = [pattern for pattern in patterns if _path_pattern_matches(pattern, path)]
  if not matches:
    return None
  best_specificity = max(_pattern_specificity(pattern) for pattern in matches)
  best = [
    pattern for pattern in matches if _pattern_specificity(pattern) == best_specificity
  ]
  if len(best) != 1:
    raise TuningError(
      f"Ambiguous tuning rules for {path}: {', '.join(sorted(best))}"
    )
  return best[0]


def match_tuning_entry(path: str) -> tuple[str, str, TuningEntry] | None:
  matches: list[tuple[str, str, TuningEntry]] = []
  for category, entries in RL_TUNING.items():
    for pattern, entry in entries.items():
      if _path_pattern_matches(pattern, path):
        matches.append((category, pattern, entry))
  if not matches:
    return None

  best_specificity = max(_pattern_specificity(pattern) for _, pattern, _ in matches)
  best = [
    match
    for match in matches
    if _pattern_specificity(match[1]) == best_specificity
  ]
  if len(best) != 1:
    patterns = ", ".join(sorted(pattern for _, pattern, _ in best))
    raise TuningError(f"Ambiguous tuning rules for {path}: {patterns}")
  return best[0]


def build_parameter_catalog(
  cfg: TrainConfig,
  *,
  require_documented: bool = False,
) -> dict[str, ParameterRecord]:
  records = discover_parameters(cfg)
  catalog: dict[str, ParameterRecord] = {}
  undocumented: list[str] = []

  for path, record in records.items():
    if not record.editable:
      catalog[path] = record
      continue

    read_only_pattern = _choose_pattern(path, tuple(RL_READ_ONLY))
    if read_only_pattern is not None:
      catalog[path] = replace(
        record,
        editable=False,
        read_only_reason=RL_READ_ONLY[read_only_pattern],
        rule_pattern=read_only_pattern,
      )
      continue

    match = match_tuning_entry(path)
    if match is None:
      catalog[path] = record
      if require_documented:
        undocumented.append(path)
      continue

    category, pattern, entry = match
    catalog[path] = replace(
      record,
      category=category,
      rule_pattern=pattern,
      tuning_entry=replace(entry),
    )

  if undocumented:
    details = "\n".join(f"  - {path}" for path in undocumented)
    raise TuningError(f"Undocumented editable parameters:\n{details}")
  return catalog


def _tuning_error_message(
  path: str,
  old: object,
  new: object,
  expected: str,
  correction: str,
) -> str:
  match = match_tuning_entry(path)
  category = match[0] if match is not None else "unclassified"
  return (
    f"[{category}] {path}: old={stable_config_value(old)!r}, "
    f"requested={stable_config_value(new)!r}; expected {expected}; "
    f"correction: {correction}"
  )


def _replace_tokens(
  current: object,
  tokens: Sequence[str],
  value: object,
) -> object:
  if not tokens:
    return value

  token, *remaining = tokens
  if is_dataclass(current) and not isinstance(current, type):
    field_names = {item.name for item in fields(current)}
    if token not in field_names:
      raise TuningError(f"Cannot replace unknown dataclass field {token!r}.")
    child = getattr(current, token)
    return replace(
      current,
      **{token: _replace_tokens(child, remaining, value)},
    )

  if isinstance(current, Mapping):
    if token not in current:
      raise TuningError(f"Cannot replace unknown mapping key {token!r}.")
    updated = dict(current)
    updated[token] = _replace_tokens(current[token], remaining, value)
    if type(current) is dict:
      return updated
    try:
      return type(current)(updated)
    except TypeError:
      return updated

  if isinstance(current, (list, tuple)):
    if not token.isdecimal():
      raise TuningError(f"Expected a sequence index, got {token!r}.")
    index = int(token)
    if index >= len(current):
      raise TuningError(
        f"Sequence index {index} is outside length {len(current)}."
      )
    updated_items = list(current)
    updated_items[index] = _replace_tokens(
      current[index], remaining, value
    )
    return tuple(updated_items) if isinstance(current, tuple) else updated_items

  raise TuningError(
    f"Cannot descend through {type(current).__name__} at token {token!r}."
  )


def replace_parameter_value(
  cfg: TrainConfig,
  path: str,
  value: object,
) -> TrainConfig:
  if not path:
    raise TuningError("Parameter path cannot be empty.")
  changed = _replace_tokens(cfg, path.split("."), value)
  if not isinstance(changed, TrainConfig):
    raise TuningError(f"Replacing {path} did not produce a TrainConfig.")
  return changed


def _allows_none(path: str) -> bool:
  optional_patterns = (
    "motion_file",
    "torchrunx_log_dir",
    "agent.load_run",
    "agent.load_checkpoint",
    "agent.algorithm.desired_kl",
    "agent.*.cnn_cfg",
    "agent.*.distribution_cfg",
    "env.actions.*.clip",
    "env.actions.*.offset",
    "env.observations.*.terms.*.clip",
    "env.observations.*.terms.*.noise",
    "env.scene.terrain.max_init_terrain_level",
    "env.scene.terrain.terrain_generator.seed",
  )
  return any(_path_pattern_matches(pattern, path) for pattern in optional_patterns)


def _allows_variable_length(path: str) -> bool:
  variable_patterns = (
    "agent.*.hidden_dims",
    "agent.obs_groups.*",
    "agent.wandb_tags",
    "env.scene.sensors.*.fields",
  )
  return any(
    _path_pattern_matches(pattern, path) for pattern in variable_patterns
  )


def _coerce_override_value(path: str, old: object, new: object) -> object:
  if path == "gpu_ids":
    if new == "all" or new is None:
      return new
    if isinstance(new, (list, tuple)) and all(
      isinstance(item, int) and not isinstance(item, bool) for item in new
    ):
      return list(new)
    raise TuningError(
      _tuning_error_message(
        path,
        old,
        new,
        "a list of GPU integers, 'all', or None",
        "use an available GPU id, 'all', or None for CPU",
      )
    )

  if new is None:
    if old is None or _allows_none(path):
      return None
    raise TuningError(
      _tuning_error_message(
        path, old, new, type(old).__name__, "provide a non-None value"
      )
    )

  if old is None:
    if isinstance(new, type) or callable(new):
      raise TuningError(
        _tuning_error_message(
          path,
          old,
          new,
          "a YAML-safe scalar or container",
          "provide a documented scalar value",
        )
      )
    return new

  if isinstance(old, bool):
    if type(new) is not bool:
      raise TuningError(
        _tuning_error_message(path, old, new, "bool", "use true or false")
      )
    return new

  if isinstance(old, Enum):
    if isinstance(new, type(old)):
      return new
    try:
      return type(old)(new)
    except (TypeError, ValueError) as exc:
      raise TuningError(
        _tuning_error_message(
          path, old, new, type(old).__name__, "use one declared enum value"
        )
      ) from exc

  if isinstance(old, int):
    if not isinstance(new, int) or isinstance(new, bool):
      raise TuningError(
        _tuning_error_message(path, old, new, "int", "provide an integer")
      )
    return new

  if isinstance(old, float):
    if not isinstance(new, (int, float)) or isinstance(new, bool):
      raise TuningError(
        _tuning_error_message(
          path, old, new, "float", "provide a finite numeric value"
        )
      )
    return float(new)

  if isinstance(old, str):
    if not isinstance(new, str):
      raise TuningError(
        _tuning_error_message(path, old, new, "str", "provide a string")
      )
    return new

  if isinstance(old, Path):
    if not isinstance(new, (str, Path)):
      raise TuningError(
        _tuning_error_message(path, old, new, "path", "provide a path string")
      )
    return type(old)(new)

  if isinstance(old, Mapping):
    if not isinstance(new, Mapping) or set(new) != set(old):
      raise TuningError(
        _tuning_error_message(
          path,
          old,
          new,
          f"a mapping with keys {sorted(old, key=str)!r}",
          "preserve every existing mapping key",
        )
      )
    return {
      key: _coerce_override_value(f"{path}.{key}", old[key], new[key])
      for key in old
    }

  if isinstance(old, (list, tuple)):
    if not isinstance(new, (list, tuple)):
      raise TuningError(
        _tuning_error_message(
          path, old, new, type(old).__name__, "provide a sequence"
        )
      )
    if not _allows_variable_length(path) and len(new) != len(old):
      raise TuningError(
        _tuning_error_message(
          path,
          old,
          new,
          f"a {type(old).__name__} of length {len(old)}",
          "preserve the existing container length",
        )
      )
    if _allows_variable_length(path) and old:
      items = [
        _coerce_override_value(f"{path}.{index}", old[0], new_item)
        for index, new_item in enumerate(new)
      ]
    elif old:
      items = [
        _coerce_override_value(f"{path}.{index}", old_item, new_item)
        for index, (old_item, new_item) in enumerate(
          zip(old, new, strict=True)
        )
      ]
    else:
      items = list(new)
    return tuple(items) if isinstance(old, tuple) else items

  if not isinstance(new, type(old)):
    raise TuningError(
      _tuning_error_message(
        path, old, new, type(old).__name__, "preserve the current value type"
      )
    )
  return new


def _validate_type(path: str, old: object, new: object) -> None:
  _coerce_override_value(path, old, new)


def _validate_finite(path: str, value: object) -> None:
  if isinstance(value, bool) or value is None:
    return
  if isinstance(value, (int, float)):
    if not math.isfinite(value):
      raise TuningError(
        f"{path}: requested value {value!r} must be finite; "
        "correction: use a finite number"
      )
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      _validate_finite(f"{path}.{key}", item)
    return
  if isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _validate_finite(f"{path}.{index}", item)


def _is_numeric(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_constraints(path: str, value: object) -> None:
  leaf = path.rsplit(".", maxsplit=1)[-1]
  positive_fields = {
    "learning_rate",
    "num_steps_per_env",
    "max_iterations",
    "save_interval",
    "num_learning_epochs",
    "num_mini_batches",
    "decimation",
    "episode_length_s",
    "timestep",
    "num_rows",
    "num_cols",
    "num_envs",
    "history_length",
    "delay_update_period",
    "nconmax",
    "njmax",
    "contact_sensor_maxmatch",
    "buffer_size",
    "max_envs_to_dump",
    "video_length",
    "video_interval",
    "period",
    "clip_param",
    "max_grad_norm",
  }
  if (
    leaf in positive_fields
    and value is not None
    and (not _is_numeric(value) or value <= 0)
  ):
    raise TuningError(
      f"{path}: requested value {value!r} must be positive; "
      "correction: use a value greater than zero"
    )

  probability_fields = {
    "gamma",
    "lam",
    "rel_standing_envs",
    "rel_heading_envs",
    "init_velocity_prob",
    "delay_hold_prob",
    "proportion",
    "soft_joint_pos_limit_factor",
  }
  if leaf in probability_fields and (
    not _is_numeric(value) or not 0 <= value <= 1
  ):
    raise TuningError(
      f"{path}: requested probability {value!r} must be in [0, 1]; "
      "correction: choose a value from zero through one"
    )

  if path.endswith(".hidden_dims") and (
    not isinstance(value, (list, tuple))
    or not value
    or any(not isinstance(item, int) or item <= 0 for item in value)
  ):
    raise TuningError(
      f"{path}: hidden_dims must contain positive integers; "
      "correction: provide at least one positive layer width"
    )

  nonnegative_fields = {
    "entropy_coef",
    "value_loss_coef",
    "command_threshold",
    "force_threshold",
    "target_height",
    "threshold",
    "walking_threshold",
    "running_threshold",
    "delay_min_lag",
    "delay_max_lag",
    "min_step_count_between_reset",
  }
  if (
    leaf in nonnegative_fields
    and value is not None
    and (not _is_numeric(value) or value < 0)
  ):
    raise TuningError(
      f"{path}: requested value {value!r} must be non-negative; "
      "correction: use zero or a positive value"
    )

  if isinstance(value, Mapping):
    for key, item in value.items():
      _validate_constraints(f"{path}.{key}", item)
    return

  if isinstance(value, (list, tuple)):
    semantic_range = (
      ".ranges." in path
      or "_range." in path
      or leaf.endswith(("range", "ranges", "range_s"))
      or leaf == "clip"
    )
    if (
      semantic_range
      and len(value) == 2
      and all(_is_numeric(item) for item in value)
    ):
      low, high = value
      if low > high:
        raise TuningError(
          f"{path}: requested range {value!r} is reversed; "
          "correction: put the lower bound first"
        )
      strict_range = (
        path.startswith(
          (
            "env.commands.",
            "env.scene.terrain.",
          )
        )
        or path.endswith(
          (
            "_range_s",
            "difficulty_range",
            "step_height_range",
            "resampling_time_range",
          )
        )
      )
      if strict_range and low == high:
        raise TuningError(
          f"{path}: requested sampling range {value!r} has no span; "
          "correction: make the upper bound greater than the lower bound"
        )
      if path.endswith("difficulty_range") and not 0 <= low <= high <= 1:
        raise TuningError(
          f"{path}: difficulty bounds must stay in [0, 1]; "
          "correction: choose ordered bounds between zero and one"
        )

    probability_vector = (
      leaf.endswith(("probabilities", "_probs"))
      or leaf == "probability_vector"
    )
    if probability_vector and all(_is_numeric(item) for item in value):
      if any(item < 0 for item in value):
        raise TuningError(
          f"{path}: probability vector contains a negative value; "
          "correction: use non-negative entries"
        )
      if not math.isclose(sum(value), 1.0, abs_tol=1.0e-6):
        raise TuningError(
          f"{path}: probability vector must sum to one; "
          "correction: normalize all entries"
        )

    for index, item in enumerate(value):
      _validate_constraints(f"{path}.{index}", item)


def _is_contract_sensitive(path: str) -> bool:
  contract_patterns = (
    "agent.*.hidden_dims",
    "agent.*.activation",
    "agent.*.class_name",
    "agent.*.obs_normalization",
    "agent.*.cnn_cfg",
    "agent.*.distribution_cfg",
    "agent.*.distribution_cfg.*",
    "agent.obs_groups.*",
    "agent.algorithm.class_name",
    "agent.algorithm.share_cnn_encoders",
    "env.observations.*.concatenate_terms",
    "env.observations.*.concatenate_dim",
    "env.observations.*.history_length",
    "env.observations.*.flatten_history_dim",
    "env.observations.*.terms.*.history_length",
    "env.observations.*.terms.*.flatten_history_dim",
    "env.scene.sensors.*.fields",
    "env.scene.sensors.*.history_length",
    "env.scene.sensors.*.num_slots",
    "env.scene.sensors.*.pattern.*",
    "env.actions.*.scale",
  )
  return any(
    _path_pattern_matches(pattern, path) for pattern in contract_patterns
  )


def active_rl_tuning_overrides() -> dict[str, object]:
  overrides: dict[str, object] = {}
  for category, entries in RL_TUNING.items():
    for path, entry in entries.items():
      if entry.value is KEEP_TASK_DEFAULT:
        continue
      if "*" in path:
        raise TuningError(
          f"[{category}] wildcard rule {path!r} has a concrete value; "
          "add one exact path under the same category"
        )
      if path in overrides:
        raise TuningError(
          f"Concrete tuning path {path!r} is declared more than once."
        )
      overrides[path] = entry.value
  return overrides


def apply_tuning_entries(
  cfg: TrainConfig,
  overrides: dict[str, object],
  *,
  allow_contract_changes: bool = False,
) -> tuple[TrainConfig, dict[str, object]]:
  catalog = build_parameter_catalog(cfg)
  changed = cfg
  manifest: dict[str, object] = {}
  resume_value = overrides.get("agent.resume", cfg.agent.resume)

  for path, requested in overrides.items():
    record = catalog.get(path)
    if record is None:
      closest = difflib.get_close_matches(path, catalog, n=3, cutoff=0.45)
      suggestions = ", ".join(closest) if closest else "none"
      raise TuningError(
        f"Unknown tuning path {path!r}; closest: {suggestions}; "
        "correction: copy an exact path from the effective catalog"
      )
    if not record.editable:
      reason = record.read_only_reason or "structural configuration"
      raise TuningError(
        f"{path} is read-only: {reason}; requested={requested!r}; "
        "correction: change the owning task configuration in a separate feature"
      )

    if _is_contract_sensitive(path):
      if resume_value is True:
        raise TuningError(
          f"[{record.category}] {path} changes the model contract while "
          "resume=True; checkpoint compatibility cannot be guaranteed; "
          "correction: start a fresh run with resume=False"
        )
      if not allow_contract_changes:
        raise TuningError(
          f"[{record.category}] {path} changes the model contract; "
          "correction: set allow_contract_changes=True and start a fresh run"
        )

    old = record.default_value
    _validate_type(path, old, requested)
    normalized = _coerce_override_value(path, old, requested)
    _validate_finite(path, normalized)
    _validate_constraints(path, normalized)
    changed = replace_parameter_value(changed, path, normalized)
    manifest[path] = {
      "category": record.category,
      "default": stable_config_value(old),
      "tuning": stable_config_value(requested),
      "final": stable_config_value(normalized),
    }

  return changed, manifest


def diff_parameter_values(
  before: TrainConfig,
  after: TrainConfig,
) -> dict[str, object]:
  before_records = discover_parameters(before)
  after_records = discover_parameters(after)
  before_paths = set(before_records)
  after_paths = set(after_records)
  if before_paths != after_paths:
    removed = sorted(before_paths - after_paths)
    added = sorted(after_paths - before_paths)
    raise TuningError(
      "Configuration structures are incompatible; "
      f"removed={removed!r}, added={added!r}; "
      "correction: keep the registered task structure unchanged"
    )

  changed: dict[str, object] = {}
  for path in sorted(before_paths):
    if path in INSPECTION_CONTROL_PATHS:
      continue
    before_value = before_records[path].default_value
    after_value = after_records[path].default_value
    if stable_config_value(before_value) != stable_config_value(after_value):
      changed[path] = after_value
  return changed


def build_effective_catalog(
  baseline_cfg: TrainConfig,
  tuned_cfg: TrainConfig,
  profile_cfg: TrainConfig,
  final_cfg: TrainConfig,
  *,
  require_documented: bool,
) -> dict[str, ParameterRecord]:
  baseline = build_parameter_catalog(
    baseline_cfg, require_documented=require_documented
  )
  tuned = discover_parameters(tuned_cfg)
  profile = discover_parameters(profile_cfg)
  final = discover_parameters(final_cfg)
  if (
    set(baseline) != set(tuned)
    or set(baseline) != set(profile)
    or set(baseline) != set(final)
  ):
    raise TuningError(
      "Baseline, tuned, profile, and final configuration structures differ; "
      "correction: do not add or remove fields through a value override"
    )

  catalog: dict[str, ParameterRecord] = {}
  for path, record in baseline.items():
    default_value = record.default_value
    tuned_value = tuned[path].default_value
    profile_value = profile[path].default_value
    final_value = final[path].default_value
    tuning_value = (
      KEEP_TASK_DEFAULT
      if stable_config_value(default_value) == stable_config_value(tuned_value)
      else tuned_value
    )
    applied_profile_value = (
      KEEP_TASK_DEFAULT
      if stable_config_value(tuned_value) == stable_config_value(profile_value)
      else profile_value
    )
    cli_value = (
      KEEP_TASK_DEFAULT
      if stable_config_value(profile_value) == stable_config_value(final_value)
      else final_value
    )
    catalog[path] = replace(
      record,
      tuning_value=tuning_value,
      profile_value=applied_profile_value,
      cli_value=cli_value,
      final_value=final_value,
    )
  return catalog


def _stable_diff_record(record: ParameterRecord) -> dict[str, object]:
  return {
    "default": stable_config_value(record.default_value),
    "tuning": stable_config_value(record.tuning_value),
    "profile": stable_config_value(record.profile_value),
    "cli": stable_config_value(record.cli_value),
    "final": stable_config_value(record.final_value),
  }


def _catalog_layer_diff(
  catalog: Mapping[str, ParameterRecord],
  layer: Literal["tuning", "profile", "cli"],
) -> dict[str, dict[str, object]]:
  diff: dict[str, dict[str, object]] = {}
  for path, record in catalog.items():
    if path in INSPECTION_CONTROL_PATHS:
      continue
    if layer == "tuning":
      value = record.tuning_value
    elif layer == "profile":
      value = record.profile_value
    else:
      value = record.cli_value
    if value is KEEP_TASK_DEFAULT:
      continue
    diff[path] = _stable_diff_record(record)
  return diff


def stable_parameter_record(record: ParameterRecord) -> dict[str, object]:
  entry = record.tuning_entry
  metadata: dict[str, object] = {
    "definition": None,
    "unit_or_range": None,
    "effect": None,
    "dependencies": None,
    "requires_retraining": None,
    "contract_effect": None,
  }
  if entry is not None:
    metadata = {
      "definition": entry.definition,
      "unit_or_range": entry.unit_or_range,
      "effect": entry.effect,
      "dependencies": entry.dependencies,
      "requires_retraining": entry.requires_retraining,
      "contract_effect": entry.contract_effect,
    }
  return {
    "path": record.path,
    "type": record.value_type,
    "source": record.source,
    "editable": record.editable,
    "read_only_reason": record.read_only_reason,
    "default": stable_config_value(record.default_value),
    "tuning": stable_config_value(record.tuning_value),
    "profile": stable_config_value(record.profile_value),
    "cli": stable_config_value(record.cli_value),
    "final": stable_config_value(record.final_value),
    "category": record.category,
    "rule_pattern": record.rule_pattern,
    **metadata,
  }


def render_effective_config(prepared: PreparedTrainConfig) -> dict[str, object]:
  return {
    "schema_version": 2,
    "baseline_commit": "3ac7da5bd5a89074a48828ec0ee34fffc5bb03c6",
    "task_id": prepared.task_id,
    "profile_path": prepared.profile_path,
    "profile_revision": prepared.profile_revision,
    "categories": list(RL_TUNING),
    "tuning_overrides": stable_config_value(prepared.tuning_diff),
    "profile_overrides": stable_config_value(prepared.profile_diff),
    "cli_overrides": stable_config_value(prepared.cli_diff),
    "parameters": {
      path: stable_parameter_record(record)
      for path, record in prepared.catalog.items()
    },
  }


def _normalize_bare_inspection_flags(argv: Sequence[str]) -> list[str]:
  bare_bool_flags = {
    "--print-effective-config",
    "--allow-contract-changes",
  }
  normalized: list[str] = []
  for index, token in enumerate(argv):
    normalized.append(token)
    if token not in bare_bool_flags:
      continue
    next_token = argv[index + 1] if index + 1 < len(argv) else None
    if next_token is None or next_token.startswith("--"):
      normalized.append("True")
  return normalized


def prepare_train_config(
  task_id: str,
  argv: list[str],
) -> PreparedTrainConfig:
  require_documented = task_id == "Unitree-G1-Stairs"
  baseline_cfg = TrainConfig.from_task(task_id)
  build_parameter_catalog(
    baseline_cfg, require_documented=require_documented
  )
  tuning_overrides = active_rl_tuning_overrides()
  source_contract_opt_in = tuning_overrides.get(
    "allow_contract_changes", baseline_cfg.allow_contract_changes
  )
  if type(source_contract_opt_in) is not bool:
    raise TuningError(
      "allow_contract_changes must be a bool; "
      "correction: use true or false in RL_TUNING"
    )
  tuned_cfg, _ = apply_tuning_entries(
    baseline_cfg,
    tuning_overrides,
    allow_contract_changes=source_contract_opt_in,
  )
  profile_cfg = tuned_cfg

  parsed_cfg = tyro.cli(
    TrainConfig,
    args=_normalize_bare_inspection_flags(argv),
    default=profile_cfg,
    prog=f"scripts/train.py {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  if not isinstance(parsed_cfg, TrainConfig):
    raise TuningError(
      "Tyro did not return TrainConfig; correction: keep the training CLI schema intact"
    )

  cli_values = diff_parameter_values(profile_cfg, parsed_cfg)
  validated_cfg, _ = apply_tuning_entries(
    profile_cfg,
    cli_values,
    allow_contract_changes=parsed_cfg.allow_contract_changes,
  )
  final_cfg = replace(
    validated_cfg,
    print_effective_config=parsed_cfg.print_effective_config,
    dump_effective_config=parsed_cfg.dump_effective_config,
    allow_contract_changes=parsed_cfg.allow_contract_changes,
  )

  reconstruction_diff = diff_parameter_values(final_cfg, parsed_cfg)
  if reconstruction_diff:
    paths = ", ".join(sorted(reconstruction_diff))
    raise TuningError(
      "Validated CLI reconstruction differs from Tyro output at "
      f"{paths}; correction: add validation support before training"
    )

  catalog = build_effective_catalog(
    baseline_cfg,
    tuned_cfg,
    profile_cfg,
    final_cfg,
    require_documented=require_documented,
  )
  return PreparedTrainConfig(
    task_id=task_id,
    baseline_cfg=baseline_cfg,
    tuned_cfg=tuned_cfg,
    cfg=final_cfg,
    catalog=catalog,
    tuning_diff=_catalog_layer_diff(catalog, "tuning"),
    profile_snapshot=None,
    profile_path=None,
    profile_revision=None,
    profile_diff=_catalog_layer_diff(catalog, "profile"),
    cli_diff=_catalog_layer_diff(catalog, "cli"),
  )


def write_training_params(
  log_dir: Path,
  cfg: TrainConfig,
  tuning_manifest: dict[str, object],
  profile_snapshot: dict[str, object] | None = None,
) -> None:
  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
  dump_yaml(log_dir / "params" / "tuning.yaml", tuning_manifest)
  if profile_snapshot is not None:
    dump_yaml(log_dir / "params" / "profile.yaml", profile_snapshot)


def _build_programmatic_manifest(
  task_id: str,
  cfg: TrainConfig,
) -> dict[str, object]:
  require_documented = task_id == "Unitree-G1-Stairs"
  baseline_cfg = TrainConfig.from_task(task_id)
  changes = diff_parameter_values(baseline_cfg, cfg)
  validated_cfg, _ = apply_tuning_entries(
    baseline_cfg,
    changes,
    allow_contract_changes=cfg.allow_contract_changes,
  )
  validated_cfg = replace(
    validated_cfg,
    print_effective_config=cfg.print_effective_config,
    dump_effective_config=cfg.dump_effective_config,
    allow_contract_changes=cfg.allow_contract_changes,
  )
  reconstruction_diff = diff_parameter_values(validated_cfg, cfg)
  if reconstruction_diff:
    paths = ", ".join(sorted(reconstruction_diff))
    raise TuningError(
      f"Programmatic configuration validation differs at {paths}; "
      "correction: pass a configuration supported by the tuning catalog"
    )
  catalog = build_effective_catalog(
    baseline_cfg,
    baseline_cfg,
    baseline_cfg,
    validated_cfg,
    require_documented=require_documented,
  )
  prepared = PreparedTrainConfig(
    task_id=task_id,
    baseline_cfg=baseline_cfg,
    tuned_cfg=baseline_cfg,
    cfg=validated_cfg,
    catalog=catalog,
    tuning_diff={},
    profile_snapshot=None,
    profile_path=None,
    profile_revision=None,
    profile_diff={},
    cli_diff=_catalog_layer_diff(catalog, "cli"),
  )
  return render_effective_config(prepared)


def run_train(
  task_id: str,
  cfg: TrainConfig,
  log_dir: Path,
  tuning_manifest: dict[str, object],
  profile_snapshot: dict[str, object] | None = None,
) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_warp_cuda_compilation()
  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    write_training_params(log_dir, cfg, tuning_manifest, profile_snapshot)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def allocate_training_log_dir(
  cfg: TrainConfig,
  *,
  now: datetime | None = None,
) -> Path:
  """Return the legacy RSL-RL run directory without creating it."""
  timestamp = datetime.now(tz=timezone.utc).astimezone() if now is None else now
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_dir_name = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir_name += f"_{cfg.agent.run_name}"
  return log_root_path / log_dir_name


def launch_training(
  task_id: str,
  args: TrainConfig | None = None,
  tuning_manifest: dict[str, object] | None = None,
  profile_snapshot: dict[str, object] | None = None,
  log_dir: Path | None = None,
) -> Path:
  args = args or TrainConfig.from_task(task_id)
  if tuning_manifest is None:
    tuning_manifest = _build_programmatic_manifest(task_id, args)
  else:
    serialized_manifest = stable_config_value(tuning_manifest)
    if not isinstance(serialized_manifest, dict):
      raise TuningError(
        "Training tuning manifest must serialize to a mapping."
      )
    tuning_manifest = serialized_manifest

  # Allocate once before launching workers; console callers may own this path.
  if log_dir is None:
    log_dir = allocate_training_log_dir(args)

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir, tuning_manifest, profile_snapshot)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(
      run_train, task_id, args, log_dir, tuning_manifest, profile_snapshot
    )
  return log_dir


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks

  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  prepared = prepare_train_config(chosen_task, remaining_args)
  del remaining_args

  if prepared.cfg.print_effective_config:
    print(
      yaml.safe_dump(
        render_effective_config(prepared), sort_keys=False, allow_unicode=True
      ),
      end="",
    )
  prepared.dump_if_requested()
  if prepared.cfg.print_effective_config or prepared.cfg.dump_effective_config:
    return

  launch_training(
    task_id=chosen_task,
    args=prepared.cfg,
    tuning_manifest=render_effective_config(prepared),
    profile_snapshot=prepared.profile_snapshot,
  )


if __name__ == "__main__":
  main()
