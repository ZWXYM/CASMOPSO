import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import itertools  # 用于循环颜色和标记
import platform
from pathlib import Path
import warnings


# --- 中文字体配置函数 ---
def configure_chinese_font():
    """
    配置Matplotlib以支持中文字符，使用更健壮的方法
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    system = platform.system()
    default_font = None
    font_found = False

    # 根据操作系统确定可能的字体位置和名称
    if system == 'Windows':
        # Windows常见中文字体
        potential_fonts = [
            # 字体名称
            'SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STHeiti', 'STKaiti', 'STSong', 'STFangsong',
            # 字体文件路径
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc',
            'C:/Windows/Fonts/simkai.ttf',
            'C:/Windows/Fonts/simfang.ttf'
        ]
    elif system == 'Darwin':  # macOS
        potential_fonts = [
            'PingFang SC', 'STHeiti', 'Heiti SC', 'Hiragino Sans GB', 'Source Han Sans CN',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Songti.ttc'
        ]
    else:  # Linux和其他系统
        potential_fonts = [
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'WenQuanYi Zen Hei Mono',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        ]

    # 尝试找到支持中文的字体
    for font in potential_fonts:
        try:
            if Path(font).exists():  # 如果是路径，检查文件是否存在
                font_prop = fm.FontProperties(fname=font)
                default_font = font
                font_found = True
                print(f"找到中文字体文件: {font}")
                break
            elif fm.findfont(fm.FontProperties(family=font)) != fm.findfont(fm.FontProperties()):
                # 如果找到的字体不是回退字体
                font_prop = fm.FontProperties(family=font)
                default_font = font
                font_found = True
                print(f"找到中文字体: {font}")
                break
        except Exception as e:
            continue

    # 如果直接方法失败，尝试添加到sans-serif字体族
    if not font_found:
        try:
            # 尝试使用系统支持的字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                                               'WenQuanYi Micro Hei', 'AR PL UMing CN',
                                               'PingFang SC', 'Heiti SC']
            plt.rcParams['axes.unicode_minus'] = False
            print("已设置字体族, 但未找到特定中文字体。图表中的中文可能仍会显示为方块。")
        except Exception as e:
            print(f"设置字体族时出错: {e}")

    # 返回找到的字体或None
    return default_font


# --- 创建支持中文的图形和标签函数 ---
def create_figure_with_chinese(title=None, figsize=(10, 8)):
    """创建支持中文的图形对象"""
    fig = plt.figure(figsize=figsize)
    if title and default_chinese_font:
        try:
            if os.path.exists(default_chinese_font):  # 如果是字体文件路径
                font_prop = fm.FontProperties(fname=default_chinese_font)
                fig.suptitle(title, fontproperties=font_prop, fontsize=16)
            else:  # 如果是字体名称
                font_prop = fm.FontProperties(family=default_chinese_font)
                fig.suptitle(title, fontproperties=font_prop, fontsize=16)
        except:
            fig.suptitle(title, fontsize=16)  # 回退到默认字体
    elif title:
        fig.suptitle(title, fontsize=16)  # 回退到默认字体

    return fig


def set_chinese_labels(ax, xlabel=None, ylabel=None, zlabel=None, title=None):
    """设置支持中文的轴标签和标题"""
    if default_chinese_font:
        try:
            if os.path.exists(default_chinese_font):  # 如果是字体文件路径
                font_prop = fm.FontProperties(fname=default_chinese_font)
            else:  # 如果是字体名称
                font_prop = fm.FontProperties(family=default_chinese_font)

            if xlabel:
                ax.set_xlabel(xlabel, fontproperties=font_prop)
            if ylabel:
                ax.set_ylabel(ylabel, fontproperties=font_prop)
            if zlabel:
                ax.set_zlabel(zlabel, fontproperties=font_prop)
            if title:
                ax.set_title(title, fontproperties=font_prop)
        except:
            # 回退到默认字体
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if zlabel:
                ax.set_zlabel(zlabel)
            if title:
                ax.set_title(title)
    else:
        # 回退到默认字体
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        if title:
            ax.set_title(title)


def set_chinese_legend(ax, fontsize=None):
    """设置支持中文的图例"""
    if default_chinese_font:
        try:
            if os.path.exists(default_chinese_font):
                font_prop = fm.FontProperties(fname=default_chinese_font, size=fontsize)
            else:
                font_prop = fm.FontProperties(family=default_chinese_font, size=fontsize)
            ax.legend(prop=font_prop)
        except:
            if fontsize:
                ax.legend(fontsize=fontsize)
            else:
                ax.legend()
    else:
        if fontsize:
            ax.legend(fontsize=fontsize)
        else:
            ax.legend()


# 调用函数配置中文字体
default_chinese_font = configure_chinese_font()

# --- 配置 ---
# 将此路径更改为您上传的文件的实际路径
file_path = 'TP1_pareto_fronts.txt'
# --- 配置结束 ---

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：文件 '{file_path}' 未找到。请确保文件路径正确。")
    exit()

# 用于存储所有帕累托前沿数据的字典
# 结构: {'前沿名称': {'f1': [], 'f2': [], 'f3': []}, ...}
pareto_fronts = {}
# 用于存储识别出的算法名称的列表 (保持顺序)
algorithm_names = []
# 真实帕累托前沿的固定键名
true_pf_key = 'True PF'

# --- 解析文件 ---
print(f"正在读取和解析文件: {file_path} ...")
current_front_name = None

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()  # 去除行首尾的空白

            # 跳过空行
            if not line:
                continue

            # 检查是否是注释行或节标题
            if line.startswith('#'):
                # 检查是否是真实帕累托前沿的标题
                if '# 真实帕累托前沿' in line:
                    current_front_name = true_pf_key
                    if current_front_name not in pareto_fronts:
                        pareto_fronts[current_front_name] = {'f1': [], 'f2': [], 'f3': []}
                    print(f"识别到部分: {current_front_name}")

                # 检查是否是算法帕累托前沿的标题
                # 假设格式为 "# <算法名称>算法运行 ... 的帕累托前沿"
                elif '算法' in line and '帕累托前沿' in line:
                    try:
                        # 提取 '#' 和 '算法' 之间的部分作为算法名称
                        algo_name = line.split('算法')[0].lstrip('# ').strip()
                        if algo_name:  # 确保提取到了名称
                            current_front_name = algo_name
                            if current_front_name not in pareto_fronts:
                                pareto_fronts[current_front_name] = {'f1': [], 'f2': [], 'f3': []}
                                # 如果是新的算法名称，添加到列表中
                                if current_front_name not in algorithm_names:
                                    algorithm_names.append(current_front_name)
                            print(f"识别到部分: {current_front_name}")
                        else:
                            print(f"警告: 无法从标题行 {line_num} 提取算法名称: '{line}'")
                            current_front_name = None  # 重置当前部分名称
                    except Exception as e:
                        print(f"错误: 处理标题行 {line_num} ('{line}') 时出错: {e}")
                        current_front_name = None  # 重置当前部分名称
                # 其他注释行忽略
                else:
                    # 保持 current_front_name 不变，以便读取紧跟标题的数据
                    pass

                continue  # 处理完注释/标题行后，跳到下一行

            # 如果当前有确定的部分名称，并且该部分已初始化，则解析数据点
            if current_front_name and current_front_name in pareto_fronts:
                try:
                    parts = line.split(',')
                    if len(parts) == 3:
                        f1 = float(parts[0].strip())
                        f2 = float(parts[1].strip())
                        f3 = float(parts[2].strip())
                        pareto_fronts[current_front_name]['f1'].append(f1)
                        pareto_fronts[current_front_name]['f2'].append(f2)
                        pareto_fronts[current_front_name]['f3'].append(f3)
                    else:
                        # 忽略格式不正确的数据行
                        # print(f"警告: 跳过行 {line_num}, 格式无效: '{line}'")
                        pass
                except ValueError:
                    # 忽略无法转换为浮点数的数据行
                    # print(f"警告: 跳过行 {line_num}, 无法转换为数字: '{line}'")
                    pass
                except Exception as e:
                    print(f"错误: 解析数据行 {line_num} ('{line}') 时出错: {e}")

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 在读取过程中丢失。")
    exit()
except Exception as e:
    print(f"读取或解析文件时发生严重错误: {e}")
    exit()

print("\n文件解析完成。")
print(f"找到 {len(pareto_fronts.get(true_pf_key, {}).get('f1', []))} 个真实帕累托前沿点。")
print(f"找到 {len(algorithm_names)} 个算法: {algorithm_names}")
for name in algorithm_names:
    print(f" - {name}: 找到 {len(pareto_fronts.get(name, {}).get('f1', []))} 个点。")

# --- 检查是否有足够的数据进行绘图 ---
true_pf_data = pareto_fronts.get(true_pf_key)
if not true_pf_data or not true_pf_data.get('f1'):
    print("\n错误: 未找到或解析 '真实帕累托前沿' 数据，无法继续绘图。")
    exit()

if not algorithm_names:
    print("\n警告: 未在文件中找到任何算法的帕累托前沿数据。")
    # 也许只绘制真实前沿？或者退出？这里先退出。
    exit()

# --- 绘图颜色和标记 ---
# 为算法选择一组不同的颜色和标记
# 使用 tab10 调色板，它有 10 种不同的颜色
algo_colors = itertools.cycle(plt.cm.tab10.colors)
# 一组不同的标记
algo_markers = itertools.cycle(['^', 's', 'P', 'X', 'D', 'v', '<', '>'])  # 添加更多标记以备不时之需
true_pf_color = 'blue'
true_pf_marker = 'o'
true_pf_size = 50
algo_size = 20

# --- 绘制每个算法与真实前沿的对比图 ---
print("\n开始生成各算法与真实前沿的对比图...")

for algo_name in algorithm_names:
    algo_data = pareto_fronts.get(algo_name)
    if not algo_data or not algo_data.get('f1'):
        print(f"跳过算法 '{algo_name}' 的绘图，因为没有找到数据点。")
        continue

    print(f"正在生成 '{algo_name}' vs '{true_pf_key}' 的图像...")

    # 使用支持中文的图形创建函数
    fig = create_figure_with_chinese(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真实帕累托前沿
    ax.scatter(true_pf_data['f1'], true_pf_data['f2'], true_pf_data['f3'],
               c=true_pf_color,
               marker=true_pf_marker,
               s=true_pf_size,
               alpha=0.6,
               label=true_pf_key)

    # 绘制当前算法的帕累托前沿
    current_algo_color = next(algo_colors)
    current_algo_marker = next(algo_markers)
    ax.scatter(algo_data['f1'], algo_data['f2'], algo_data['f3'],
               c=[current_algo_color],  # 使用列表以避免 matplotlib 警告
               marker=current_algo_marker,
               s=algo_size,
               alpha=0.8,
               label=f'{algo_name} PF')

    # 设置标签、标题和图例（使用支持中文的函数）
    set_chinese_labels(
        ax,
        xlabel='$f_1$',
        ylabel='$f_2$',
        zlabel='$f_3$',
        title=f'帕累托前沿: {true_pf_key} vs {algo_name}'
    )

    # 设置支持中文的图例
    set_chinese_legend(ax)

    plt.tight_layout()
    plt.show()  # 显示当前图像

print("\n所有单个对比图已生成并显示。")

# --- 绘制包含所有前沿的组合图 ---
print("\n开始生成所有帕累托前沿的组合图...")

# 使用支持中文的图形创建函数
fig_combined = create_figure_with_chinese(figsize=(12, 10))
ax_combined = fig_combined.add_subplot(111, projection='3d')

# 在组合图中再次绘制真实帕累托前沿 (使用不同颜色/标记以示区别或作为基准)
ax_combined.scatter(true_pf_data['f1'], true_pf_data['f2'], true_pf_data['f3'],
                    c='black',  # 在组合图中用黑色表示真实前沿
                    marker='.',  # 用小点标记
                    s=15,
                    alpha=0.5,
                    label=true_pf_key)

# 重置颜色和标记循环，以确保与单个图的顺序一致（如果需要）或获得新的循环
algo_colors = itertools.cycle(plt.cm.tab10.colors)
algo_markers = itertools.cycle(['^', 's', 'P', 'X', 'D', 'v', '<', '>'])

# 绘制所有算法的帕累托前沿
for algo_name in algorithm_names:
    algo_data = pareto_fronts.get(algo_name)
    if not algo_data or not algo_data.get('f1'):
        continue  # 跳过没有数据的算法

    current_algo_color = next(algo_colors)
    current_algo_marker = next(algo_markers)
    ax_combined.scatter(algo_data['f1'], algo_data['f2'], algo_data['f3'],
                        c=[current_algo_color],
                        marker=current_algo_marker,
                        s=algo_size,  # 可以稍微调小一点以便观察重叠
                        alpha=0.7,
                        label=f'{algo_name} PF')

# 设置组合图的标签、标题和图例（使用支持中文的函数）
set_chinese_labels(
    ax_combined,
    xlabel='$f_1$',
    ylabel='$f_2$',
    zlabel='$f_3$',
    title='帕累托前沿组合对比图'
)

# 设置支持中文的图例（字体大小设为小）
set_chinese_legend(ax_combined, fontsize='small')

plt.tight_layout()
plt.show()  # 显示组合图像

print("\n组合对比图已生成并显示。")
print("所有绘图完成。")