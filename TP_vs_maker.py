import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re
import csv
from datetime import datetime


class AlgorithmPerformanceAnalyzer:
    """算法性能分析工具，用于处理CEC2020测试算法的性能数据"""

    def __init__(self, data_dir='.', output_dir='results'):
        """
        初始化分析器

        参数:
        data_dir: 包含性能数据文件的目录
        output_dir: 输出图表和数据的目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建子目录
        self.curves_dir = os.path.join(output_dir, 'convergence_curves')
        self.comparison_dir = os.path.join(output_dir, 'comparisons')

        for directory in [self.curves_dir, self.comparison_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # 性能指标列表
        self.metrics = ['IGDF', 'IGDX', 'RPSP', 'HV', 'SP']

        # 存储所有数据
        self.all_data = {}

        # 存储算法排名信息
        self.rankings = {}

    def load_data(self):
        """加载所有算法的性能数据文件"""
        # 寻找所有性能数据文件
        file_pattern = os.path.join(self.data_dir, '*_run_*_generational.txt')
        files = glob(file_pattern)

        print(f"找到{len(files)}个性能数据文件")

        for file_path in files:
            # 从文件名提取算法名称和运行ID
            file_name = os.path.basename(file_path)
            match = re.match(r'(.+)_run_(\d+)_generational\.txt', file_name)

            if match:
                algorithm, run_id = match.groups()
                run_id = int(run_id)

                # 解析文件内容
                data = self._parse_performance_file(file_path)

                # 存储数据
                if algorithm not in self.all_data:
                    self.all_data[algorithm] = {}

                self.all_data[algorithm][run_id] = data
                print(f"已加载 {algorithm} 算法运行 {run_id} 的数据，共{len(data)}条记录")

        return len(files) > 0

    def _parse_performance_file(self, file_path):
        """解析性能数据文件的内容"""
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            header_found = False

            for line in f:
                line = line.strip()

                # 跳过注释行和空行
                if not line or line.startswith('#'):
                    continue

                # 解析数据行
                values = line.split(',')
                if len(values) >= 7:  # 确保有足够的值 (迭代，计算次数，5个指标)
                    try:
                        iteration = int(values[0])
                        comp_count = int(values[1])
                        igdf = float(values[2])
                        igdx = float(values[3])
                        rpsp = float(values[4])
                        hv = float(values[5])
                        sp = float(values[6])

                        data.append({
                            'Iteration': iteration,
                            'ComputationCount': comp_count,
                            'IGDF': igdf,
                            'IGDX': igdx,
                            'RPSP': rpsp,
                            'HV': hv,
                            'SP': sp
                        })
                    except ValueError as e:
                        print(f"警告: 解析文件 {file_path} 时遇到错误: {e}")

        return data

    def plot_convergence_curves(self):
        """为每个算法生成收敛曲线"""
        if not self.all_data:
            print("没有数据可绘制")
            return

        for algorithm, runs in self.all_data.items():
            print(f"正在绘制 {algorithm} 的收敛曲线...")

            # 创建算法的专用目录
            algo_dir = os.path.join(self.curves_dir, algorithm)
            if not os.path.exists(algo_dir):
                os.makedirs(algo_dir)

            # 为每个指标生成一个图
            for metric in self.metrics:
                fig, ax = plt.subplots(figsize=(10, 6))

                # 计算每个迭代的平均值和标准差
                max_iteration = max(max(d['Iteration'] for d in run_data) for run_data in runs.values())
                iterations = list(range(max_iteration + 1))

                # 提取每次运行在每个迭代的指标值
                metric_values = {i: [] for i in iterations}

                for run_id, run_data in runs.items():
                    for data_point in run_data:
                        iter_num = data_point['Iteration']
                        if iter_num <= max_iteration:  # 确保不超出最大迭代次数
                            metric_values[iter_num].append(data_point[metric])

                # 计算平均值和标准差
                mean_values = []
                std_values = []

                for i in iterations:
                    if metric_values[i]:
                        valid_values = [v for v in metric_values[i] if not np.isnan(v)]
                        if valid_values:
                            mean_values.append(np.mean(valid_values))
                            std_values.append(np.std(valid_values))
                        else:
                            # 如果没有有效值，使用上一个有效的均值和标准差
                            if mean_values:
                                mean_values.append(mean_values[-1])
                                std_values.append(std_values[-1])
                            else:
                                mean_values.append(np.nan)
                                std_values.append(np.nan)
                    else:
                        # 如果该迭代没有值，使用线性插值
                        if mean_values:
                            mean_values.append(mean_values[-1])
                            std_values.append(std_values[-1])
                        else:
                            mean_values.append(np.nan)
                            std_values.append(np.nan)

                # 绘制平均值线
                line, = ax.plot(iterations, mean_values, '-', linewidth=2, label=f'平均 {metric}')

                # 添加置信区间
                fill = ax.fill_between(iterations,
                                       [m - s for m, s in zip(mean_values, std_values)],
                                       [m + s for m, s in zip(mean_values, std_values)],
                                       alpha=0.2, color=line.get_color(), label='±标准差区间')

                # 设置图表属性
                ax.set_xlabel('迭代次数', fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{algorithm} - {metric} 收敛曲线', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend([line, fill], ['平均值', '±标准差区间'])

                # 保存图表
                save_path = os.path.join(algo_dir, f'{metric}_convergence.png')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()

            # 生成一个包含所有指标的综合图
            self._plot_combined_metrics(algorithm, runs)

    def _plot_combined_metrics(self, algorithm, runs):
        """为算法生成一个包含所有指标的综合图"""
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 4 * len(self.metrics)))

        max_iteration = max(max(d['Iteration'] for d in run_data) for run_data in runs.values())
        iterations = list(range(max_iteration + 1))

        for i, metric in enumerate(self.metrics):
            ax = axes[i]

            # 提取每次运行在每个迭代的指标值
            metric_values = {iter_num: [] for iter_num in iterations}

            for run_id, run_data in runs.items():
                for data_point in run_data:
                    iter_num = data_point['Iteration']
                    if iter_num <= max_iteration:
                        metric_values[iter_num].append(data_point[metric])

            # 计算平均值和标准差
            mean_values = []
            std_values = []

            for iter_num in iterations:
                if metric_values[iter_num]:
                    valid_values = [v for v in metric_values[iter_num] if not np.isnan(v)]
                    if valid_values:
                        mean_values.append(np.mean(valid_values))
                        std_values.append(np.std(valid_values))
                    else:
                        if mean_values:
                            mean_values.append(mean_values[-1])
                            std_values.append(std_values[-1])
                        else:
                            mean_values.append(np.nan)
                            std_values.append(np.nan)
                else:
                    if mean_values:
                        mean_values.append(mean_values[-1])
                        std_values.append(std_values[-1])
                    else:
                        mean_values.append(np.nan)
                        std_values.append(np.nan)

            # 绘制平均值线
            line, = ax.plot(iterations, mean_values, '-', linewidth=2)

            # 添加置信区间
            fill = ax.fill_between(iterations,
                                   [m - s for m, s in zip(mean_values, std_values)],
                                   [m + s for m, s in zip(mean_values, std_values)],
                                   alpha=0.2, color=line.get_color(), label='±标准差区间')

            # 设置图表属性
            ax.set_xlabel('迭代次数')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} 收敛曲线')
            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加图例 - 明确说明平均值线和标准差区域
            ax.legend([line, fill], ['平均值', '±标准差区间'])

            # 设置Y轴范围
            if metric in ['IGDF', 'IGDX', 'RPSP', 'SP']:
                valid_means = [m for m in mean_values if not np.isnan(m)]
                if valid_means:
                    ax.set_ylim(0, max(valid_means) * 1.1)

        plt.tight_layout()
        save_path = os.path.join(self.curves_dir, algorithm, 'all_metrics.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_algorithm_comparisons(self):
        """绘制不同算法之间的性能对比"""
        if not self.all_data or len(self.all_data) < 2:
            print("没有足够的算法数据进行比较")
            return

        # 为每个指标创建比较图
        for metric in self.metrics:
            self._plot_metric_comparison(metric)

        # 创建性能箱线图
        self._plot_performance_boxplots()

    def _plot_metric_comparison(self, metric):
        """绘制特定指标的算法比较图"""
        fig, ax = plt.subplots(figsize=(12, 7))

        # 找出所有算法中的最大迭代次数
        max_iteration = 0
        for algorithm, runs in self.all_data.items():
            for run_id, run_data in runs.items():
                max_iter = max(d['Iteration'] for d in run_data)
                max_iteration = max(max_iteration, max_iter)

        iterations = list(range(max_iteration + 1))

        # 为每个算法计算平均值
        for algorithm, runs in self.all_data.items():
            # 提取每次运行在每个迭代的指标值
            metric_values = {iter_num: [] for iter_num in iterations}

            for run_id, run_data in runs.items():
                for data_point in run_data:
                    iter_num = data_point['Iteration']
                    if iter_num <= max_iteration:
                        metric_values[iter_num].append(data_point[metric])

            # 计算平均值
            mean_values = []

            for iter_num in iterations:
                if metric_values[iter_num]:
                    valid_values = [v for v in metric_values[iter_num] if not np.isnan(v)]
                    if valid_values:
                        mean_values.append(np.mean(valid_values))
                    else:
                        if mean_values:
                            mean_values.append(mean_values[-1])
                        else:
                            mean_values.append(np.nan)
                else:
                    if mean_values:
                        mean_values.append(mean_values[-1])
                    else:
                        mean_values.append(np.nan)

            # 绘制算法的平均值线
            ax.plot(iterations, mean_values, '-', linewidth=2, label=algorithm)

        # 设置图表属性
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'算法 {metric} 性能对比', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        # 设置Y轴范围（根据指标类型）
        if metric in ['IGDF', 'IGDX', 'RPSP', 'SP']:
            ax.set_ylim(bottom=0)  # 从0开始

        # 保存图表
        save_path = os.path.join(self.comparison_dir, f'{metric}_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_performance_boxplots(self):
        """绘制算法性能的小提琴图和箱线图，按照特定样式"""
        n_metrics = len(self.metrics)

        # 创建垂直排列的子图
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 6 * n_metrics))

        # 确保axes是列表，即使只有一个指标
        if n_metrics == 1:
            axes = [axes]

        # 指标显示名称映射
        metric_labels = {
            "IGDF": "IGDF",
            "IGDX": "IGDX",
            "RPSP": "RPSP",
            "HV": "HV",
            "SP": "SP"
        }

        # 导出全部数据CSV的路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violin_data_csv = os.path.join(self.output_dir, f'violin_plot_data_{timestamp}.csv')

        # 创建用于存储CSV数据的字典
        csv_data = {'Algorithm': [], 'Metric': [], 'Value': []}

        # 为每个指标创建小提琴图
        for i, metric in enumerate(self.metrics):
            # 使用所有迭代的数据收集
            data = []
            labels = []
            all_values_dict = {}  # 用于存储每个算法的所有值

            for algorithm, runs in self.all_data.items():
                # 收集该算法所有运行的所有迭代数据
                all_values = []
                for run_id, run_data in runs.items():
                    for data_point in run_data:
                        if not np.isnan(data_point[metric]):
                            all_values.append(data_point[metric])
                            # 添加到CSV数据
                            csv_data['Algorithm'].append(algorithm)
                            csv_data['Metric'].append(metric)
                            csv_data['Value'].append(data_point[metric])

                if all_values:
                    data.append(all_values)
                    labels.append(algorithm)
                    all_values_dict[algorithm] = all_values

            if data:
                # 获取当前子图
                ax = axes[i]

                # 设置配色方案
                colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
                color_dict = dict(zip(labels, colors))

                # 绘制小提琴图
                violin_parts = ax.violinplot(
                    data,
                    showmeans=False,
                    showmedians=False,  # 不显示中位线，后面自己添加
                    showextrema=False  # 不显示极值线
                )

                # 设置小提琴图颜色
                for j, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors[j])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                    pc.set_linewidth(1.0)

                # 添加箱线图
                bp = ax.boxplot(
                    data,
                    positions=range(1, len(data) + 1),
                    widths=0.15,
                    patch_artist=True,
                    showfliers=False,  # 不显示离群点，改用采样点显示
                    showmeans=True,
                    meanline=True
                )

                # 自定义箱线图样式
                for j, box in enumerate(bp['boxes']):
                    box.set(facecolor='white', alpha=0.5)
                    box.set(edgecolor='black')

                # 设置中位线样式
                for median in bp['medians']:
                    median.set_color('green')
                    median.set_linewidth(1.5)

                # 设置均值线样式
                for mean in bp['means']:
                    mean.set_color('red')
                    mean.set_linewidth(1.5)

                # 设置胡须线样式
                for whisker in bp['whiskers']:
                    whisker.set_color('black')
                    whisker.set_linestyle('--')
                    whisker.set_linewidth(1.0)

                # 设置封顶线样式
                for cap in bp['caps']:
                    cap.set_color('black')
                    cap.set_linewidth(1.0)

                # 为每个算法添加自定义元素
                for j, algorithm in enumerate(labels):
                    pos = j + 1  # 位置索引从1开始
                    values = all_values_dict[algorithm]

                    if values:
                        # 对值进行排序
                        sorted_values = sorted(values)
                        max_value = max(sorted_values)

                        # 采样保留0-4个离散点，确保包含最大值
                        n_points = min(4, len(sorted_values))
                        if n_points > 0:
                            # 计算采样间隔
                            indices = [int(i * (len(sorted_values) - 1) / (n_points - 1)) for i in range(n_points - 1)]
                            # 确保包含最大值
                            indices.append(len(sorted_values) - 1)
                            # 去重并排序
                            indices = sorted(set(indices))
                            # 获取采样点的值
                            sampled_values = [sorted_values[i] for i in indices]

                            # 绘制采样点
                            for val in sampled_values:
                                ax.plot([pos], [val], 'o', color='black', ms=5, mec='black', mew=1, alpha=0.7)

                        # 添加小提琴底部的蓝色线段
                        x_min, x_max = pos - 0.07, pos + 0.07  # 线段的长度
                        min_val = min(sorted_values)
                        ax.plot([x_min, x_max], [min_val, min_val], '-', color='blue', linewidth=2)

                        # 添加最大值处的蓝色线段
                        ax.plot([x_min, x_max], [max_value, max_value], '-', color='blue', linewidth=2)

                        # 添加连接线
                        ax.plot([pos, pos], [min_val, max_value], '-', color='blue', linewidth=1)

                # 设置图表标题和标签
                ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
                ax.set_title(f'{metric_labels.get(metric, metric.upper())} Performance', fontsize=14)

                # 设置X轴标签
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha='right')

                # 设置网格线
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')

                # 根据指标类型调整Y轴范围
                if metric in ["IGDF", "IGDX", "RPSP", "SP"]:  # 较小值更好
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(bottom=0)  # 从0开始

                # 美化布局
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                # 如果没有数据，显示提示信息
                ax = axes[i]
                ax.text(0.5, 0.5, f"No valid {metric.upper()} data",
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()

        # 设置总标题
        plt.suptitle(f'Algorithm Performance on TP1', fontsize=16)

        # 优化布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # 保存图像
        save_path = os.path.join(self.comparison_dir, 'all_metrics_boxplots.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 将收集的数据导出为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(violin_data_csv, index=False)
        print(f"小提琴图数据已保存到: {violin_data_csv}")

    def calculate_performance_metrics(self):
        """计算各算法的性能指标并输出CSV文件"""
        if not self.all_data:
            print("没有数据可计算")
            return

        # 存储每个算法在每个指标上的最终值
        metrics_data = {}

        for algorithm, runs in self.all_data.items():
            metrics_data[algorithm] = {metric: [] for metric in self.metrics}

            for run_id, run_data in runs.items():
                if run_data:
                    # 获取最后一次迭代的数据
                    last_data = max(run_data, key=lambda x: x['Iteration'])
                    for metric in self.metrics:
                        if not np.isnan(last_data[metric]):
                            metrics_data[algorithm][metric].append(last_data[metric])

        # 计算每个指标的均值和标准差
        mean_std_data = {}

        for algorithm in metrics_data:
            mean_std_data[algorithm] = {}
            for metric in self.metrics:
                values = metrics_data[algorithm][metric]
                if values:
                    mean_std_data[algorithm][f"{metric}_mean"] = np.mean(values)
                    mean_std_data[algorithm][f"{metric}_std"] = np.std(values)
                else:
                    mean_std_data[algorithm][f"{metric}_mean"] = np.nan
                    mean_std_data[algorithm][f"{metric}_std"] = np.nan

        # 计算每个指标的排名
        rankings = {algorithm: {} for algorithm in metrics_data}

        for metric in self.metrics:
            # 收集每个算法的平均值
            metric_values = {algo: mean_std_data[algo][f"{metric}_mean"] for algo in mean_std_data
                             if not np.isnan(mean_std_data[algo][f"{metric}_mean"])}

            if metric_values:
                # 对IGDF, IGDX, RPSP, SP排序（越小越好）
                if metric in ["IGDF", "IGDX", "RPSP", "SP"]:
                    sorted_algos = sorted(metric_values.keys(), key=lambda x: metric_values[x])
                # 对HV排序（越大越好）
                else:
                    sorted_algos = sorted(metric_values.keys(), key=lambda x: -metric_values[x])

                # 赋予排名
                for rank, algo in enumerate(sorted_algos, 1):
                    rankings[algo][f"{metric}_rank"] = float(rank)

        # 计算平均排名
        for algorithm in rankings:
            ranks = [rankings[algorithm].get(f"{metric}_rank", np.nan) for metric in self.metrics]
            valid_ranks = [r for r in ranks if not np.isnan(r)]
            if valid_ranks:
                rankings[algorithm]["平均排名"] = sum(valid_ranks) / len(valid_ranks)
            else:
                rankings[algorithm]["平均排名"] = np.nan

        # 按平均排名进行总体排序
        sorted_algos = sorted(rankings.keys(),
                              key=lambda x: rankings[x].get("平均排名", float('inf')))

        # 添加总体排名
        for rank, algo in enumerate(sorted_algos, 1):
            rankings[algo]["总体排名"] = rank

        # 存储排名信息以供后续使用
        self.rankings = rankings

        # 生成CSV文件1 - 详细版
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_csv_path = os.path.join(self.output_dir, f'algorithm_performance_raw_{timestamp}.csv')

        with open(raw_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            header = [""]
            for metric in self.metrics:
                header.extend([f"{metric}_mean", f"{metric}_std"])

            for metric in self.metrics:
                header.append(f"{metric}_rank")

            header.extend(["平均排名", "总体排名"])
            writer.writerow(header)

            # 按总体排名排序写入数据
            for algorithm in sorted_algos:
                row = [algorithm]

                # 添加均值和标准差
                for metric in self.metrics:
                    row.extend([
                        mean_std_data[algorithm].get(f"{metric}_mean", "N/A"),
                        mean_std_data[algorithm].get(f"{metric}_std", "N/A")
                    ])

                # 添加排名
                for metric in self.metrics:
                    row.append(rankings[algorithm].get(f"{metric}_rank", "N/A"))

                # 添加平均排名和总体排名
                row.append(rankings[algorithm].get("平均排名", "N/A"))
                row.append(rankings[algorithm].get("总体排名", "N/A"))

                writer.writerow(row)

        print(f"详细性能指标已保存到: {raw_csv_path}")

        # 生成CSV文件2 - 简化版带格式化
        comparison_csv_path = os.path.join(self.output_dir, f'algorithm_performance_comparison_{timestamp}.csv')

        with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ["", "总体排名", "平均排名"]
            header.extend(self.metrics)
            writer.writerow(header)

            # 按总体排名排序写入数据
            for algorithm in sorted_algos:
                row = [algorithm,
                       rankings[algorithm].get("总体排名", "N/A"),
                       f"{rankings[algorithm].get('平均排名', 'N/A'):.2f}"]

                # 添加格式化的性能指标
                for metric in self.metrics:
                    mean = mean_std_data[algorithm].get(f"{metric}_mean", np.nan)
                    std = mean_std_data[algorithm].get(f"{metric}_std", np.nan)

                    if not np.isnan(mean) and not np.isnan(std):
                        row.append(f"{mean:.6e}±{std:.6e}")
                    else:
                        row.append("N/A")

                writer.writerow(row)

        print(f"对比性能指标已保存到: {comparison_csv_path}")

        return raw_csv_path, comparison_csv_path

    def run_analysis(self):
        """执行完整的分析流程"""
        print("开始分析算法性能数据...")

        # 加载数据
        if not self.load_data():
            print("未找到性能数据文件，请检查目录设置")
            return False

        # 绘制收敛曲线
        self.plot_convergence_curves()

        # 绘制算法比较图
        self.plot_algorithm_comparisons()

        # 计算性能指标并输出CSV
        raw_csv, comparison_csv = self.calculate_performance_metrics()

        print("分析完成！")
        print(f"- 收敛曲线已保存到: {self.curves_dir}")
        print(f"- 比较图表已保存到: {self.comparison_dir}")
        print(f"- 详细性能指标已保存到: {raw_csv}")
        print(f"- 对比性能指标已保存到: {comparison_csv}")

        return True


# 主程序
if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 创建分析器实例
    analyzer = AlgorithmPerformanceAnalyzer(data_dir='.', output_dir='algorithm_results')

    # 运行分析
    analyzer.run_analysis()