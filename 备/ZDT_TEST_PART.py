"""
改进的多目标PSO算法框架及CEC2020标准化评估工具
包含所有必要的引用和导入
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import logging
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
import copy  # 新增: 用于深拷贝跟踪数据
from scipy.special import comb  # 用于组合数计算


# ======================  测试函数实现 ======================
class Problem:
    """多目标优化问题基类 (保持不变)"""

    def __init__(self, name, n_var, n_obj, xl, xu):
        self.name = name
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        self.pareto_front = None
        self.pareto_set = None

    def evaluate(self, x):
        raise NotImplementedError("每个问题类必须实现evaluate方法")

    def get_pareto_front(self):
        return self.pareto_front

    def get_pareto_set(self):
        return self.pareto_set

    def _generate_uniform_points_on_simplex(self, n_points_approx, M):
        """在 M 维单位单纯形上生成近似均匀的点"""
        H = 1
        # 使用 scipy.special.comb 计算组合数 C(H + M - 1, M - 1)
        while comb(H + M - 1, M - 1, exact=True) < n_points_approx:
            H += 1
        n_points_actual = comb(H + M - 1, M - 1, exact=True)

        points = []

        def generate_recursive(index, current_sum, current_point):
            if index == M - 1:
                current_point[index] = H - current_sum
                points.append(current_point.copy() / H)
                return
            for i in range(H - current_sum + 1):
                current_point[index] = i
                generate_recursive(index + 1, current_sum + i, current_point)

        generate_recursive(0, 0, np.zeros(M))
        return np.array(points), n_points_actual


# --- ZDT 函数 ---

class ZDT1(Problem):
    """ZDT1 测试函数 (两目标)"""

    def __init__(self, n_var=30):
        super().__init__("ZDT1", n_var, 2, [0] * n_var, [1] * n_var)
        self._generate_pf(100)

    def evaluate(self, x):
        x = np.asarray(x)
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return [f1, f2]

    def _generate_pf(self, n_points):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        self.pareto_front = np.column_stack((f1, f2))
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1


class ZDT2(Problem):
    """ZDT2 测试函数 (两目标, 非凸)"""

    def __init__(self, n_var=30):
        super().__init__("ZDT2", n_var, 2, [0] * n_var, [1] * n_var)
        self._generate_pf(100)

    def evaluate(self, x):
        x = np.asarray(x)
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        return [f1, f2]

    def _generate_pf(self, n_points):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        self.pareto_front = np.column_stack((f1, f2))
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1


class ZDT3(Problem):
    """ZDT3 测试函数 (两目标, 不连续)"""

    def __init__(self, n_var=30):
        super().__init__("ZDT3", n_var, 2, [0] * n_var, [1] * n_var)
        self._generate_pf(100)

    def evaluate(self, x):
        x = np.asarray(x)
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return [f1, f2]

    def _generate_pf(self, n_points):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        # 真实的 Pareto 前沿由多个不相交的连续段组成
        # 为了简化，我们仍然返回由线段连接的曲线作为近似
        self.pareto_front = np.column_stack((f1, f2))
        # Pareto解集: x1 in [0, 1], x_i = 0 for i=2,...,n_var
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1


class ZDT4(Problem):
    """ZDT4 测试函数 (两目标, 多峰)"""

    def __init__(self, n_var=10):
        xl = [0.0] + [-5.0] * (n_var - 1)
        xu = [1.0] + [5.0] * (n_var - 1)
        super().__init__("ZDT4", n_var, 2, xl, xu)
        self._generate_pf(100)

    def evaluate(self, x):
        x = np.asarray(x)
        f1 = x[0]
        g = 1 + 10 * (self.n_var - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return [f1, f2]

    def _generate_pf(self, n_points):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        self.pareto_front = np.column_stack((f1, f2))
        # Pareto解集: x1 in [0, 1], x_i = 0 for i=2,...,n_var
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1


class ZDT6(Problem):
    """ZDT6 测试函数 (两目标, 非均匀)"""

    def __init__(self, n_var=10):
        super().__init__("ZDT6", n_var, 2, [0] * n_var, [1] * n_var)
        self._generate_pf(100)

    def evaluate(self, x):
        x = np.asarray(x)
        f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)
        g = 1 + 9 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        return [f1, f2]

    def _generate_pf(self, n_points):
        # 真实 Pareto 前沿是 f2 = 1 - f1^2, 但 f1 的取值范围是 [0.281, 1]
        # 且解在 Pareto 前沿上分布不均匀
        # 为了可视化，生成理论上的完整前沿
        f1_theory = np.linspace(0, 1, n_points)
        f2 = 1 - f1_theory ** 2
        self.pareto_front = np.column_stack((f1_theory, f2))
        # Pareto解集: x_i = 0 for i=2,...,n_var. x1 需反解
        # 近似解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        x1_approx = np.linspace(0, 1, n_points)  # 粗略近似
        self.pareto_set[:, 0] = x1_approx


# --- DTLZ 函数 ---

class DTLZ1(Problem):
    """DTLZ1 测试函数"""

    def __init__(self, n_var=7, n_obj=3):
        k = 5
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ1", n_var, n_obj, [0] * n_var, [1] * n_var)

        # 增加前沿点采样密度
        simplex_points, n_actual = self._generate_uniform_points_on_simplex(500, self.n_obj)  # 增加到500点
        self.pareto_front = 0.5 * simplex_points
        self._generate_ps(n_actual)

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        g = 100 * (self.n_var - M + 1 + np.sum((x[M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[M - 1:] - 0.5))))
        f = np.zeros(M)
        factor = 0.5 * (1 + g)
        for i in range(M):
            prod = np.prod(x[:M - 1 - i])
            if i > 0:
                prod *= (1 - x[M - 1 - i])
            f[i] = factor * prod
        return list(f)

    def _generate_ps(self, n_points):
        M = self.n_obj
        self.pareto_set = np.zeros((n_points, self.n_var))
        # 反解 x0, ..., x_{M-2} from PF points
        pf = self.pareto_front / 0.5  # 归一化到单纯形
        for i in range(n_points):
            for j in range(M - 1):  # j = 0 to M-2
                prod = np.prod(self.pareto_set[i, :j])  # Product of x_0 to x_{j-1}
                if abs(prod) < 1e-9:  # Avoid division by zero if previous x is 0
                    self.pareto_set[i, j] = 0
                else:
                    # f_j = (1-x_{M-1-j}) * prod(x_k for k<M-1-j)
                    # We need x_j. Let's use the relation f_{M-1} = 1-x0, f_{M-2}=(1-x1)x0, ...
                    # From f_{M-1} = 1-x0 => x0 = 1 - f_{M-1}
                    # From f_{M-2} = (1-x1)x0 => 1-x1 = f_{M-2}/x0 => x1 = 1 - f_{M-2}/x0
                    # Generally, x_j = 1 - f_{M-1-j} / product(x_k for k<j)
                    idx_f = M - 1 - (j + 1)  # index for f corresponding to (1-x_j) term
                    if idx_f >= 0:
                        self.pareto_set[i, j] = 1 - pf[i, idx_f] / prod if prod > 1e-9 else 0
                    else:  # Last variable x_{M-2} related to f_0
                        self.pareto_set[i, j] = pf[i, 0] / prod if prod > 1e-9 else 0

        self.pareto_set = np.clip(self.pareto_set, 0, 1)  # Clip to bounds
        # Set last k variables to 0.5
        if self.n_var >= M:
            self.pareto_set[:, M - 1:] = 0.5


class DTLZ2(Problem):
    """DTLZ2 测试函数"""

    def __init__(self, n_var=12, n_obj=3):
        k = 10
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ2", n_var, n_obj, [0] * n_var, [1] * n_var)
        # Pareto前沿是单位超球面的一部分: sum(f_i^2) = 1
        ref_dirs, n_actual = self._generate_uniform_points_on_simplex(200, self.n_obj)
        # Map simplex points to sphere surface
        self.pareto_front = np.zeros((n_actual, self.n_obj))
        for i in range(n_actual):
            for j in range(self.n_obj):
                prod = 1.0
                for k in range(self.n_obj - 1 - j):
                    prod *= np.cos(ref_dirs[i, k] * np.pi / 2)
                if j > 0:
                    prod *= np.sin(ref_dirs[i, self.n_obj - 1 - j] * np.pi / 2)
                self.pareto_front[i, j] = prod
        self._generate_ps(n_actual, ref_dirs)

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        g = np.sum((x[M - 1:] - 0.5) ** 2)
        f = np.zeros(M)
        factor = 1 + g
        for i in range(M):
            prod = 1.0
            for j in range(M - 1 - i):
                prod *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                prod *= np.sin(0.5 * np.pi * x[M - 1 - i])
            f[i] = factor * prod
        return list(f)

    def _generate_ps(self, n_points, ref_dirs):
        M = self.n_obj
        self.pareto_set = np.zeros((n_points, self.n_var))
        # First M-1 variables are the reference directions (simplex points)
        self.pareto_set[:, :M - 1] = ref_dirs[:, :M - 1]
        # Last k variables are 0.5
        if self.n_var >= M:
            self.pareto_set[:, M - 1:] = 0.5


class DTLZ3(Problem):
    """DTLZ3 测试函数 (多峰)"""

    def __init__(self, n_var=12, n_obj=3):
        k = 10
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ3", n_var, n_obj, [0] * n_var, [1] * n_var)
        # Pareto前沿与DTLZ2相同，但g函数多峰
        ref_dirs, n_actual = self._generate_uniform_points_on_simplex(200, self.n_obj)
        # Map simplex points to sphere surface
        self.pareto_front = np.zeros((n_actual, self.n_obj))
        for i in range(n_actual):
            for j in range(self.n_obj):
                prod = 1.0
                for k in range(self.n_obj - 1 - j):
                    prod *= np.cos(ref_dirs[i, k] * np.pi / 2)
                if j > 0:
                    prod *= np.sin(ref_dirs[i, self.n_obj - 1 - j] * np.pi / 2)
                self.pareto_front[i, j] = prod
        # PS与DTLZ2相同
        self.pareto_set = np.zeros((n_actual, self.n_var))
        self.pareto_set[:, :n_obj - 1] = ref_dirs[:, :n_obj - 1]
        if self.n_var >= n_obj:
            self.pareto_set[:, n_obj - 1:] = 0.5

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        # DTLZ3 g函数
        g = 100 * (self.n_var - M + 1 + np.sum((x[M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[M - 1:] - 0.5))))
        f = np.zeros(M)
        factor = 1 + g
        for i in range(M):
            prod = 1.0
            for j in range(M - 1 - i):
                prod *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                prod *= np.sin(0.5 * np.pi * x[M - 1 - i])
            f[i] = factor * prod
        return list(f)


class DTLZ4(Problem):
    """DTLZ4 测试函数 (偏置)"""

    def __init__(self, n_var=12, n_obj=3, alpha=100):
        k = 10
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ4", n_var, n_obj, [0] * n_var, [1] * n_var)
        self.alpha = alpha

        # --- 修改PF生成逻辑 ---
        # 不再使用 DTLZ2 的方式生成 PF 点
        # 改为生成 Pareto Set 上的点，然后评估它们以获得 PF 点 (体现 alpha 偏差)
        n_points_approx = 200  # 生成大约 200 个真实前沿点用于可视化
        M = self.n_obj

        # 1. 在 M-1 维的决策变量空间 [0,1]^(M-1) 上生成均匀点
        #    这些点代表帕累托解集中的 x_0, ..., x_{M-2}
        #    可以使用简单的网格或随机采样，这里用 _generate_uniform_points_on_simplex
        #    生成 M-1 维单纯形上的点作为 x_0...x_{M-2} 的近似均匀采样
        #    注意：这只是为了生成可视化点，不代表真实的PS分布形状
        decision_vars_part1, n_actual = self._generate_uniform_points_on_simplex(n_points_approx, M - 1)
        n_points = n_actual

        # 2. 构建完整的帕累托解集 (PS) 上的采样点
        ps_samples = np.full((n_points, self.n_var), 0.5)  # 后 k 个变量设为 0.5
        ps_samples[:, :M - 1] = decision_vars_part1  # 前 M-1 个变量来自均匀采样

        # 3. 评估这些 PS 采样点以获得 PF 点
        self.pareto_front = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            # 直接调用 evaluate 方法，它内部会应用 alpha
            # 注意：evaluate 返回的是 list，需要转 array
            self.pareto_front[i, :] = np.array(self.evaluate(ps_samples[i, :]))
            # DTLZ 的 evaluate 包含 (1+g) 因子，在真实前沿 g=0
            # 所以需要除以 (1+g)，这里 g 对于 PS 点是 0
            # g = np.sum((ps_samples[i, M - 1:] - 0.5) ** 2) # g=0
            # factor_f = 1 + g # = 1
            # self.pareto_front[i, :] = self.pareto_front[i, :] / factor_f # 不需要除以1

        # Pareto解集 (PS): x0..x_{M-2} in [0,1], x_i=0.5 for i=M-1..n
        # 保存一个理论上的 PS 供参考 (虽然上面的 decision_vars_part1 只是采样)
        ps_theory_part1, n_ps_theory = self._generate_uniform_points_on_simplex(200, M - 1)
        self.pareto_set = np.full((n_ps_theory, self.n_var), 0.5)
        self.pareto_set[:, :M - 1] = ps_theory_part1
        # --- 修改结束 ---

    def evaluate(self, x):  # evaluate 方法保持不变
        x = np.asarray(x)
        M = self.n_obj
        # DTLZ4 使用 alpha 参数
        # 重要：确保这里使用了 alpha 次方
        x_pow = x[:M - 1] ** self.alpha
        g = np.sum((x[M - 1:] - 0.5) ** 2)
        f = np.zeros(M)
        factor = 1 + g
        for i in range(M):  # i=0..M-1
            prod = 1.0
            for j in range(M - 1 - i):  # j=0..M-2-i
                # 使用 x_pow
                prod *= np.cos(0.5 * np.pi * x_pow[j])
            if i > 0:
                # 使用 x_pow
                prod *= np.sin(0.5 * np.pi * x_pow[M - 1 - i])
            f[i] = factor * prod
        return list(f)


class DTLZ5(Problem):
    """DTLZ5 测试函数 (退化)"""

    def __init__(self, n_var=12, n_obj=3):
        k = 10
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ5", n_var, n_obj, [0] * n_var, [1] * n_var)
        # Pareto前沿是一条退化的曲线
        # x1 in [0,1], x_i=0.5 for i=M-1 to n
        # theta_i = pi/(4(1+g)) * (1+2gx_i) for i=1..M-2
        # theta_0 = x0*pi/2
        # PF: f_i = cos(theta_0)...cos(theta_{M-2})cos(theta_{M-1}) etc.
        # For M=3, theta_1 = pi/(4(1+g))*(1+2gx1), theta_0=x0*pi/2
        # f1 = cos(theta_0)cos(theta_1)
        # f2 = cos(theta_0)sin(theta_1)
        # f3 = sin(theta_0)
        # When g=0, theta_1 = pi/4. PF becomes f1=cos(x0*pi/2)*cos(pi/4), f2=cos(x0*pi/2)*sin(pi/4), f3=sin(x0*pi/2)
        # This is a curve on the sphere surface where f1=f2
        n_points = 100
        theta_0 = np.linspace(0, np.pi / 2, n_points)
        f1 = np.cos(theta_0) * np.cos(np.pi / 4)
        f2 = np.cos(theta_0) * np.sin(np.pi / 4)
        f3 = np.sin(theta_0)
        self.pareto_front = np.column_stack((f1, f2, f3))
        # PS: x0 in [0,1], x1=0.5, x_i=0.5 for i=M-1 to n
        self.pareto_set = np.full((n_points, self.n_var), 0.5)
        self.pareto_set[:, 0] = theta_0 / (np.pi / 2)  # x0 = theta_0 / (pi/2)

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        g = np.sum((x[M - 1:] - 0.5) ** 2)
        theta = np.zeros(M - 1)
        theta[0] = x[0] * np.pi / 2
        factor = np.pi / (4 * (1 + g))
        theta[1:] = factor * (1 + 2 * g * x[1:M - 1])

        f = np.zeros(M)
        factor_f = 1 + g
        for i in range(M):  # i=0..M-1
            prod = 1.0
            for j in range(M - 1 - i):  # j=0..M-2-i
                prod *= np.cos(theta[j])
            if i > 0:
                prod *= np.sin(theta[M - 1 - i])
            f[i] = factor_f * prod
        return list(f)


class DTLZ6(Problem):
    """DTLZ6 测试函数 (退化+多峰)"""

    def __init__(self, n_var=12, n_obj=3):
        k = 10
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ6", n_var, n_obj, [0] * n_var, [1] * n_var)
        # PF 与 DTLZ5 相同 (退化曲线)
        n_points = 100
        theta_0 = np.linspace(0, np.pi / 2, n_points)
        f1 = np.cos(theta_0) * np.cos(np.pi / 4)
        f2 = np.cos(theta_0) * np.sin(np.pi / 4)
        f3 = np.sin(theta_0)
        self.pareto_front = np.column_stack((f1, f2, f3))
        # PS 与 DTLZ5 相同
        self.pareto_set = np.full((n_points, self.n_var), 0.0)  # Optimal g requires x_i=0
        self.pareto_set[:, 0] = theta_0 / (np.pi / 2)
        # For DTLZ6, optimal g requires x_i = 0 for i >= M-1
        # Note: DTLZ6 definition uses M=3 explicitly in g function index
        if self.n_var >= 3:
            self.pareto_set[:, 2:] = 0.0  # Indices M-1=2 onwards

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        # DTLZ6 g函数 - 注意索引是从 x[2] 开始 (假设 M=3)
        g = np.sum(x[M - 1:] ** 0.1)  # Sum(x_i^0.1) for i=M-1..n-1
        theta = np.zeros(M - 1)
        theta[0] = x[0] * np.pi / 2
        factor = np.pi / (4 * (1 + g))
        theta[1:] = factor * (1 + 2 * g * x[1:M - 1])

        f = np.zeros(M)
        factor_f = 1 + g
        for i in range(M):
            prod = 1.0
            for j in range(M - 1 - i):
                prod *= np.cos(theta[j])
            if i > 0:
                prod *= np.sin(theta[M - 1 - i])
            f[i] = factor_f * prod
        return list(f)


class DTLZ7(Problem):
    """DTLZ7 测试函数 (不连续前沿)"""

    def __init__(self, n_var=22, n_obj=3):
        k = 20
        if n_var is None: n_var = n_obj + k - 1
        super().__init__("DTLZ7", n_var, n_obj, [0] * n_var, [1] * n_var)
        # PF 由多个不连续段组成
        # f_i = x_i for i=0..M-2
        # f_{M-1} = (1+g)h
        # h = M - sum_{i=0}^{M-2} [f_i/(1+g) * (1+sin(3*pi*f_i))]
        # 在 PF 上, g=0, h = M - sum_{i=0}^{M-2} [f_i * (1+sin(3*pi*f_i))]
        # f_{M-1} = h
        # 生成PF点 (需要更多点来捕捉不连续性)
        n_points_approx = 2000
        ref_dirs, n_actual = self._generate_uniform_points_on_simplex(n_points_approx,
                                                                      self.n_obj - 1)  # Use M-1 dimensions for x0..x_{M-2}
        n_points = n_actual

        pf = np.zeros((n_points, self.n_obj))
        pf[:, :self.n_obj - 1] = ref_dirs  # f_i = x_i
        h_sum = np.sum(pf[:, :self.n_obj - 1] * (1 + np.sin(3 * np.pi * pf[:, :self.n_obj - 1])), axis=1)
        pf[:, self.n_obj - 1] = self.n_obj - h_sum  # f_{M-1} = h
        self.pareto_front = pf

        # PS: x_i for i=0..M-2 are the ref_dirs, x_i=0 for i=M-1..n
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, :self.n_obj - 1] = ref_dirs
        # Last k variables are 0
        if self.n_var >= self.n_obj:
            self.pareto_set[:, self.n_obj - 1:] = 0.0

    def evaluate(self, x):
        x = np.asarray(x)
        M = self.n_obj
        k = self.n_var - M + 1
        g = 1 + 9 * np.sum(x[M - 1:]) / k
        f = np.zeros(M)
        f[:M - 1] = x[:M - 1]  # f_i = x_i for i=0..M-2

        h_sum = np.sum((f[:M - 1] / (1 + g)) * (1 + np.sin(3 * np.pi * f[:M - 1])))
        h = M - h_sum
        f[M - 1] = (1 + g) * h
        return list(f)


# ====================== 粒子群优化算法实现 ======================

class Particle:
    """粒子类，用于PSO算法"""

    def __init__(self, dimensions, bounds):
        """
        初始化粒子
        dimensions: 维度数（决策变量数量）
        bounds: 每个维度的取值范围列表，格式为[(min1,max1), (min2,max2),...]
        """
        self.dimensions = dimensions
        self.bounds = bounds

        # 初始化位置和速度
        self.position = self._initialize_position()
        self.velocity = np.zeros(dimensions)

        # 初始化个体最优位置和适应度
        self.best_position = self.position.copy()
        self.fitness = None
        self.best_fitness = None

    def _initialize_position(self):
        """初始化位置"""
        position = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            position[i] = min_val + np.random.random() * (max_val - min_val)
        return position

    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        更新速度
        w: 惯性权重
        c1: 个体认知系数
        c2: 社会认知系数
        """
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        """更新位置并确保在边界内"""
        # 更新位置
        self.position = self.position + self.velocity

        # 确保位置在合法范围内
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            self.position[i] = max(min_val, min(max_val, self.position[i]))


class CASMOPSO:
    """增强版多目标粒子群优化算法"""

    def __init__(self, problem, pop_size=200, max_iterations=300,
                 w_init=0.9, w_end=0.4, c1_init=2.5, c1_end=0.5,
                 c2_init=0.5, c2_end=2.5,
                 archive_size=100, mutation_rate=0.1, adaptive_grid_size=15,
                 k_vmax=0.5, use_archive=True, max_consecutive_leadership=0):
        """
        初始化 CASMOPSO 算法

        新增参数:
        max_consecutive_leadership: 同一粒子最多连续担任领导者的次数
            - 当设为正整数时，限制同一粒子的连续任期(如10表示最多连续10代)
            - 当设为1时，每个粒子最多担任一次领导者
            - 当设为0时，关闭任期限制功能，使用原始无限制逻辑
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.w_init = w_init
        self.w_end = w_end
        self.c1_init = c1_init
        self.c1_end = c1_end
        self.c2_init = c2_init
        self.c2_end = c2_end
        self.use_archive = use_archive
        self.archive_size = archive_size
        self.mutation_rate = mutation_rate
        self.adaptive_grid_size = adaptive_grid_size
        self.max_consecutive_leadership = max_consecutive_leadership  # 新增：最大连续担任领导者次数
        # 速度限制相关
        self.k_vmax = k_vmax
        self.vmax = np.zeros(self.problem.n_var)
        if hasattr(self.problem, 'xu') and hasattr(self.problem, 'xl'):
            self.vmax = self.k_vmax * (np.asarray(self.problem.xu) - np.asarray(self.problem.xl))
        else:
            print("警告: Problem对象缺少xu或xl属性，无法计算vmax。将不进行速度限制。")
            self.vmax = None

        # 粒子群和外部存档
        self.particles = []
        self.archive = []

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []
            }
        }

        # 新增: 领导者历史记录
        self.leader_history = []  # 存储历史领导者信息 [(position, fitness, crowding_distance), ...]
        self.leader_history_by_gen = {}  # 按代际存储领导者 {iteration: [(position, fitness, crowding_distance), ...]}
        self.avg_leader_crowding = 0.0  # 历史领导者平均拥挤度
        self.current_gen_leaders = []  # 当前代的领导者集合，用于统计分析

        # 新增: 领导者连续任期跟踪 - 仅当max_consecutive_leadership > 0时激活
        if self.max_consecutive_leadership > 0:
            self.current_leaders = {}  # 当前正在担任领导者的粒子 {粒子标识符: 连续担任次数}
            self.leader_blacklist = set()  # 已达到最大连续次数的领导者黑名单
        else:
            # 当max_consecutive_leadership = 0时，不启用任期跟踪
            self.current_leaders = None
            self.leader_blacklist = None
        self.stagnation_counter = 0  # 停滞计数器
        self.stagnation_threshold = 10  # 停滞阈值
        self.last_hv = 0.0  # 上一代的超体积值
        self.min_mutation_rate = 0.02  # 最小变异率
        self.distance_threshold = 1e-6  # 存档更新的距离阈值
        self.restart_percentage = 0.3  # 重启时重新初始化的种群比例

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程 - 添加停滞检测与重启机制"""
        # 初始化粒子群
        self._initialize_particles()

        # 初始评估
        for particle in self.particles:
            evaluation_result = self.problem.evaluate(particle.position)
            if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                objectives = evaluation_result[0]
            else:
                objectives = evaluation_result
            particle.fitness = np.array(objectives)
            particle.best_position = particle.position.copy()
            particle.best_fitness = particle.fitness.copy()

        # 初始化外部存档
        if self.use_archive:
            self._update_archive()

        # 优化迭代
        pbar = tqdm(range(self.max_iterations), desc=f"Optimizing {self.problem.name} with {self.__class__.__name__}",
                    disable=not verbose)

        for iteration in pbar:
            # 周期性重置策略 - 避免长期停滞
            if iteration > 100 and iteration % 50 == 0:
                # 1. 周期性重置预测模型历史
                if hasattr(self, 'leader_history') and len(self.leader_history) > 20:
                    keep_size = 20  # 只保留最近20个记录
                    self.leader_history = self.leader_history[-keep_size:]
                    if verbose:
                        print(f"[迭代 {iteration}] 周期性重置: 保留最近{keep_size}条历史记录")

                # 2. 周期性重置领导者任期限制
                if self.max_consecutive_leadership > 0:
                    self.current_leaders = {}
                    self.leader_blacklist = set()

                # 3. 周期性随机化部分种群 - 即使没有检测到停滞也执行
                if random.random() < 0.7:  # 70%概率执行
                    randomize_count = int(self.pop_size * 0.2)  # 随机化20%的种群
                    indices = np.random.choice(self.pop_size, randomize_count, replace=False)

                    for idx in indices:
                        bounds = list(zip(self.problem.xl, self.problem.xu))
                        self.particles[idx] = Particle(self.problem.n_var, bounds)

                        # 评估新粒子
                        evaluation_result = self.problem.evaluate(self.particles[idx].position)
                        if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                            objectives = evaluation_result[0]
                        else:
                            objectives = evaluation_result
                        self.particles[idx].fitness = np.array(objectives)
                        self.particles[idx].best_position = self.particles[idx].position.copy()
                        self.particles[idx].best_fitness = self.particles[idx].fitness.copy()

                    if verbose:
                        print(f"[迭代 {iteration}] 周期性随机化: 重新初始化{randomize_count}个粒子")

            # 每代开始时重置当前代的领导者集合
            self.current_gen_leaders = []
            if self.max_consecutive_leadership > 0:
                self.leader_blacklist = set()

            # 更新参数
            progress = iteration / self.max_iterations
            w = self.w_init - (self.w_init - self.w_end) * progress
            c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
            c2 = self.c2_init + (self.c2_end - self.c2_init) * progress

            # 对每个粒子
            for i, particle in enumerate(self.particles):
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self._select_leader(particle, iteration, i)
                else:
                    leader = self._select_leader_from_swarm(particle, i)

                # 如果没有领导者可选
                if leader is None:
                    leader = particle

                # 更新速度
                particle.update_velocity(leader.best_position, w, c1, c2)

                # 应用速度限制
                if self.vmax is not None:
                    particle.velocity = np.clip(particle.velocity, -self.vmax, self.vmax)

                # 更新位置
                particle.update_position()

                # 应用变异
                self._apply_mutation(particle, progress)

                # 评估新位置
                evaluation_result = self.problem.evaluate(particle.position)
                if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                    objectives = evaluation_result[0]
                else:
                    objectives = evaluation_result
                particle.fitness = np.array(objectives)

                # 更新个体最优 (pbest)
                if not self._dominates(particle.best_fitness, particle.fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness.copy()

            # 更新外部存档
            if self.use_archive:
                self._update_archive()

            # 保存当前代的领导者集合
            if self.current_gen_leaders:
                self.leader_history_by_gen[iteration] = self.current_gen_leaders.copy()

            # 更新领导者历史的拥挤度平均值
            if self.leader_history:
                crowding_values = [record[2] for record in self.leader_history]
                self.avg_leader_crowding = np.mean(crowding_values)

            # 检测停滞并执行重启策略
            if tracking and iteration > 0 and iteration % 5 == 0:
                self._check_stagnation_and_restart(iteration)

            # 跟踪性能指标
            if tracking and iteration % 1 == 0:
                self._track_performance(iteration)

            # 更新进度条
            if verbose and iteration % 10 == 0:
                pbar.set_postfix({"ArchiveSize": len(self.archive), "Stagnation": self.stagnation_counter})

        # 确保记录最后一次迭代的性能
        if tracking:
            self._track_performance(self.max_iterations - 1)

        # 关闭进度条
        pbar.close()

        if verbose:
            print(f"优化完成，最终存档大小: {len(self.archive)}")

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    # 方案2：改进停滞检测和重启机制
    def _check_stagnation_and_restart(self, iteration):
        """检测停滞并执行重启策略 - 增强版"""
        # 计算当前超体积指标
        current_front = self._get_pareto_front()
        if len(current_front) > 1:
            try:
                # 获取真实前沿用于参考点设置
                true_front = self.problem.get_pareto_front()
                if true_front is not None:
                    ref_point = np.max(true_front, axis=0) * 1.1
                else:
                    ref_point = np.max(current_front, axis=0) * 1.1

                current_hv = PerformanceIndicators.hypervolume(current_front, ref_point)

                # 1. 添加多样性检测 - 使用间距指标
                diversity = PerformanceIndicators.spacing(current_front)

                # 2. 综合评估停滞状态
                hv_stagnant = abs(current_hv - self.last_hv) < 1e-6
                diversity_stagnant = hasattr(self, 'last_diversity') and abs(diversity - self.last_diversity) < 1e-6

                # 如果超体积或多样性指标停滞，增加计数器
                if hv_stagnant or diversity_stagnant:
                    self.stagnation_counter += 1
                else:
                    # 仅在两个指标都有改善时减少计数器
                    if not hv_stagnant and not diversity_stagnant:
                        self.stagnation_counter = max(0, self.stagnation_counter - 1)

                # 更新上一次的指标值
                self.last_hv = current_hv
                self.last_diversity = diversity if not hasattr(self, 'last_diversity') else diversity

                # 3. 动态停滞阈值 - 随迭代降低
                progress = iteration / self.max_iterations
                # 早期允许更多迭代(最高10)，后期迅速降低阈值(最低5)
                dynamic_threshold = max(5, int(10 * (1 - progress * 0.5)))

                # 如果检测到停滞，执行增强的重启策略
                if self.stagnation_counter >= dynamic_threshold:
                    self._perform_restart(iteration, progress)
                    self.stagnation_counter = 0

            except Exception as e:
                print(f"超体积计算错误: {e}")

    # 方案3：增强重启策略
    def _perform_restart(self, iteration, progress=None):
        """执行增强版重启策略"""
        if progress is None:
            progress = iteration / self.max_iterations

        print(f"\n[迭代 {iteration}] 检测到停滞，执行增强重启...")

        # 保留一部分存档中的最优解
        archive_copy = self.archive.copy() if self.archive else []

        # 1. 动态调整重启比例 - 后期更激进
        restart_fraction = min(0.3 + progress * 0.4, 0.8)  # 从30%到最高70%
        restart_count = int(self.pop_size * restart_fraction)
        restart_indices = np.random.choice(self.pop_size, restart_count, replace=False)

        print(f"重启 {restart_count}个粒子 ({restart_fraction:.2f}的种群)")

        # 重新初始化选定的粒子
        bounds = list(zip(self.problem.xl, self.problem.xu))
        for idx in restart_indices:
            # 2. 多样化重启策略
            strategy = random.choices(
                ['random', 'archive_based', 'extreme'],
                weights=[0.6, 0.3, 0.1],  # 60%随机，30%基于存档，10%极端值
                k=1
            )[0]

            if strategy == 'random':
                # 完全随机重新初始化
                self.particles[idx] = Particle(self.problem.n_var, bounds)

            elif strategy == 'archive_based' and archive_copy:
                # 基于存档创建有扰动的粒子
                template = random.choice(archive_copy)
                self.particles[idx] = Particle(self.problem.n_var, bounds)

                # 复制但添加显著扰动
                for dim in range(self.problem.n_var):
                    if random.random() < 0.7:  # 70%的维度基于模板
                        range_width = self.problem.xu[dim] - self.problem.xl[dim]
                        # 较大扰动
                        perturbation = range_width * random.uniform(-0.2, 0.2)
                        self.particles[idx].position[dim] = template.best_position[dim] + perturbation
                        # 确保在边界内
                        self.particles[idx].position[dim] = np.clip(
                            self.particles[idx].position[dim],
                            self.problem.xl[dim],
                            self.problem.xu[dim]
                        )

            else:  # 'extreme'策略
                # 创建具有极端值的粒子
                self.particles[idx] = Particle(self.problem.n_var, bounds)
                for dim in range(self.problem.n_var):
                    # 30%概率使用边界值
                    if random.random() < 0.3:
                        self.particles[idx].position[dim] = random.choice([
                            self.problem.xl[dim],
                            self.problem.xu[dim]
                        ])

            # 评估新粒子
            evaluation_result = self.problem.evaluate(self.particles[idx].position)
            if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                objectives = evaluation_result[0]
            else:
                objectives = evaluation_result

            self.particles[idx].fitness = np.array(objectives)
            self.particles[idx].best_position = self.particles[idx].position.copy()
            self.particles[idx].best_fitness = self.particles[idx].fitness.copy()

        # 3. 动态调整变异率 - 指数增长但基于进度
        base_increase = 1.5 + progress  # 后期增长更快
        self.mutation_rate = min(self.mutation_rate * base_increase, 0.5)  # 提高上限到0.5

        # 4. 清除部分历史记录以减少历史影响
        history_keep = max(10, int(30 * (1 - progress * 0.7)))  # 后期只保留10-30个记录
        if len(self.leader_history) > history_keep:
            self.leader_history = self.leader_history[-history_keep:]

        # 5. 重置领导者任期限制
        if self.max_consecutive_leadership > 0:
            self.current_leaders = {}
            self.leader_blacklist = set()

        print(
            f"重启完成: {restart_count}个粒子被重新初始化，变异率提高到{self.mutation_rate:.4f}，历史记录保留{len(self.leader_history)}项")

    def _apply_mutation(self, particle, progress):
        """变异操作 - 修改为确保最小变异率"""
        # 根据迭代进度调整变异率，但确保最小值
        current_rate = max(self.mutation_rate * (1 - progress * 0.7), self.min_mutation_rate)

        # 周期性增加变异率以跳出局部最优
        if random.random() < 0.05:  # 5%的概率
            current_rate = max(current_rate, self.mutation_rate * 0.5)

        # 对每个维度
        for i in range(self.problem.n_var):
            if np.random.random() < current_rate:
                # 多项式变异
                eta_m = 5  # 分布指数

                delta1 = (particle.position[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - particle.position[i]) / (self.problem.xu[i] - self.problem.xl[i])

                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                particle.position[i] += delta_q * (self.problem.xu[i] - self.problem.xl[i])
                particle.position[i] = max(self.problem.xl[i], min(self.problem.xu[i], particle.position[i]))

    def _initialize_particles(self):
        """粒子初始化"""
        self.particles = []
        bounds = list(zip(self.problem.xl, self.problem.xu))

        # 标准化初始化代码，移除特殊处理
        for i in range(self.pop_size):
            particle = Particle(self.problem.n_var, bounds)

            # 特殊初始化前20%的粒子 - 保留这个通用的多样性策略
            if i < self.pop_size // 5:
                # 均匀分布粒子位置以提高多样性
                for j in range(self.problem.n_var):
                    alpha = i / (self.pop_size // 5)
                    particle.position[j] = self.problem.xl[j] + alpha * (self.problem.xu[j] - self.problem.xl[j])

            self.particles.append(particle)

    def _get_particle_identifier(self, particle, particle_idx=None):
        """
        获取粒子的唯一标识符
        可以使用位置的哈希值，或者粒子在种群中的索引
        """
        if particle_idx is not None:
            # 使用粒子在种群中的索引作为标识符
            return f"particle_{particle_idx}"
        else:
            # 使用位置的哈希值作为标识符
            return hash(tuple(particle.position))

    def _select_leader(self, particle, iteration, particle_idx=None):
        """选择领导者 - 动态持续更新的改进版，增加多样性和随机性"""
        if not self.archive:
            return None

        # 如果存档太小，随机选择
        if len(self.archive) <= 2:
            selected_leader = random.choice(self.archive)
            # 当启用了任期限制时，记录领导者信息
            if self.max_consecutive_leadership > 0:
                self._update_leader_tenure(selected_leader, particle_idx)
            return selected_leader

        # 前100代使用自适应拥挤度选择并收集数据
        if iteration < 500:
            # 获取候选领导者
            leader = self._crowding_distance_leader(particle)

            # 当max_consecutive_leadership > 0时，检查是否满足连续任期限制
            if self.max_consecutive_leadership > 0:
                leader_id = self._get_particle_identifier(leader)

                # 如果领导者已经在黑名单中，选择另一个领导者
                if leader_id in self.leader_blacklist:
                    # 尝试找到不在黑名单中的领导者
                    alternate_leader = self._find_alternate_leader(particle)
                    if alternate_leader:
                        leader = alternate_leader
                        leader_id = self._get_particle_identifier(leader)
                    else:
                        # 如果找不到替代领导者，暂时允许使用黑名单中的领导者
                        print(f"警告: 无法找到不在黑名单中的领导者，暂时使用黑名单中的领导者")

                # 更新领导者任期记录
                self._update_leader_tenure(leader, particle_idx)

            # 记录领导者信息
            leader_position = leader.best_position.copy()
            leader_fitness = leader.best_fitness.copy()

            # 计算领导者周围的拥挤度
            archive_fitnesses = [a.best_fitness for a in self.archive]
            crowding_distances = self._calculate_crowding_distance(archive_fitnesses)

            # 找到领导者在存档中的索引
            leader_idx = None
            for i, a in enumerate(self.archive):
                if np.array_equal(a.best_position, leader_position):
                    leader_idx = i
                    break

            # 记录领导者的拥挤度
            if leader_idx is not None:
                leader_crowding = crowding_distances[leader_idx]
                leader_record = (leader_position, leader_fitness, leader_crowding)
                self.leader_history.append(leader_record)
                self.current_gen_leaders.append(leader_record)

            return leader

        # 100代之后使用动态预测机制
        else:
            # 如果历史记录不足，回退到普通选择
            if len(self.leader_history) < 10:
                leader = self._crowding_distance_leader(particle)

                # 当启用任期限制时，检查领导者连续任期限制
                if self.max_consecutive_leadership > 0:
                    leader_id = self._get_particle_identifier(leader)

                    # 如果领导者已经在黑名单中，选择另一个领导者
                    if leader_id in self.leader_blacklist:
                        alternate_leader = self._find_alternate_leader(particle)
                        if alternate_leader:
                            leader = alternate_leader
                            leader_id = self._get_particle_identifier(leader)

                    # 更新领导者任期记录
                    self._update_leader_tenure(leader, particle_idx)

                # 仍然记录领导者信息用于未来迭代
                leader_position = leader.best_position.copy()
                leader_fitness = leader.best_fitness.copy()

                archive_fitnesses = [a.best_fitness for a in self.archive]
                crowding_distances = self._calculate_crowding_distance(archive_fitnesses)

                leader_idx = None
                for i, a in enumerate(self.archive):
                    if np.array_equal(a.best_position, leader_position):
                        leader_idx = i
                        break

                if leader_idx is not None:
                    leader_crowding = crowding_distances[leader_idx]
                    leader_record = (leader_position, leader_fitness, leader_crowding)
                    self.leader_history.append(leader_record)
                    self.current_gen_leaders.append(leader_record)

                return leader

            # 重新计算平均拥挤度 - 使用缩小的滑动窗口
            window_size = 30  # 考虑最近30个领导者的拥挤度
            recent_leaders = self.leader_history[-min(window_size, len(self.leader_history)):]
            crowding_values = [record[2] for record in recent_leaders]
            self.avg_leader_crowding = np.mean(crowding_values)

            # 根据最近的趋势动态拟合 - 使用最近的数据点
            # 确定拟合窗口大小 - 随机从5-15之间选择以增加多样性
            fit_window_size = random.randint(5, min(15, len(self.leader_history)))
            recent_history = self.leader_history[-fit_window_size:]

            # 提取位置和适应度
            positions = np.array([record[0] for record in recent_history])
            fitnesses = np.array([record[1] for record in recent_history])

            # 使用更复杂的预测模型
            try:
                # 计算迭代进度
                progress = iteration / self.max_iterations

                # 初始化预测位置
                predicted_position = np.zeros(self.problem.n_var)

                # 检测停滞状态 - 降低阈值使其更早触发
                stagnation_detected = hasattr(self, 'stagnation_counter') and self.stagnation_counter > 2

                # 动态调整拟合窗口大小 - 后期使用更小的窗口
                if progress > 0.7:
                    # 后期使用较小的拟合窗口
                    fit_window_size = random.randint(3, min(10, len(self.leader_history)))
                else:
                    # 前期使用较大的拟合窗口
                    fit_window_size = random.randint(5, min(15, len(self.leader_history)))

                recent_history = self.leader_history[-fit_window_size:]
                positions = np.array([record[0] for record in recent_history])

                # 对每个决策变量维度拟合趋势
                for dim in range(self.problem.n_var):
                    # 使用时间步作为x值
                    x = np.arange(len(positions))
                    y = positions[:, dim]

                    # 根据进度动态调整多项式阶数
                    if progress > 0.7:
                        poly_order = 1  # 后期只使用线性拟合
                    else:
                        poly_order = min(2, len(x) - 1)

                    try:
                        # 尝试多项式拟合
                        z = np.polyfit(x, y, poly_order)
                        p = np.poly1d(z)

                        # 预测下一步 - 增加随机性
                        next_x = len(positions) + random.uniform(0.5, 2.0)
                        predicted_position[dim] = p(next_x)
                    except:
                        # 回退到线性拟合
                        A = np.vstack([x, np.ones(len(x))]).T
                        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
                        next_x = len(positions) + random.uniform(0.5, 2.0)
                        predicted_position[dim] = a * next_x + b

                    # 根据进度和停滞情况添加不同程度的随机扰动
                    range_width = self.problem.xu[dim] - self.problem.xl[dim]

                    # 1. 基础扰动 - 随进度增加
                    base_perturbation = range_width * 0.01 * (1.0 + progress * 3.0) * random.uniform(-1, 1)
                    predicted_position[dim] += base_perturbation

                    # 2. 停滞时的额外扰动
                    if stagnation_detected:
                        stag_perturbation = range_width * 0.05 * self.stagnation_counter / 5.0 * random.uniform(-1, 1)
                        predicted_position[dim] += stag_perturbation

                # 3. 强力探索 - 随机选择少数维度进行大幅扰动
                if random.random() < 0.3 + 0.4 * progress:  # 后期更频繁
                    n_dims = max(1, int(self.problem.n_var * 0.2))  # 扰动20%的维度
                    dims = random.sample(range(self.problem.n_var), n_dims)
                    for dim in dims:
                        range_width = self.problem.xu[dim] - self.problem.xl[dim]
                        big_perturbation = range_width * random.uniform(0.1, 0.3) * random.choice([-1, 1])
                        predicted_position[dim] += big_perturbation

                # 确保在边界内
                predicted_position = np.clip(predicted_position, self.problem.xl, self.problem.xu)

                # 确保预测位置在合法范围内
                predicted_position = np.clip(predicted_position, self.problem.xl, self.problem.xu)

                # 评估预测位置的适应度
                predicted_fitness = np.array(self.problem.evaluate(predicted_position))

                # 创建临时粒子
                temp_particle = Particle(self.problem.n_var, list(zip(self.problem.xl, self.problem.xu)))
                temp_particle.position = predicted_position.copy()
                temp_particle.best_position = predicted_position.copy()
                temp_particle.fitness = predicted_fitness.copy()
                temp_particle.best_fitness = predicted_fitness.copy()

                # 生成临时粒子的标识符 - 仅当启用任期限制时需要
                if self.max_consecutive_leadership > 0:
                    temp_particle_id = hash(tuple(temp_particle.position))

                    # 检查预测的粒子是否已经达到最大连续任期
                    if temp_particle_id in self.leader_blacklist:
                        # 如果在黑名单中，添加一些随机扰动以创建新的不同领导者
                        for dim in range(self.problem.n_var):
                            # 添加随机扰动，但保持预测趋势的大致方向
                            perturbation = (self.problem.xu[dim] - self.problem.xl[dim]) * 0.08 * random.uniform(-1, 1)
                            temp_particle.position[dim] += perturbation
                            temp_particle.best_position[dim] += perturbation

                        # 确保在边界内
                        temp_particle.position = np.clip(temp_particle.position, self.problem.xl, self.problem.xu)
                        temp_particle.best_position = np.clip(temp_particle.best_position, self.problem.xl,
                                                              self.problem.xu)

                        # 重新评估
                        temp_particle.fitness = np.array(self.problem.evaluate(temp_particle.position))
                        temp_particle.best_fitness = temp_particle.fitness.copy()

                        # 更新粒子ID
                        temp_particle_id = hash(tuple(temp_particle.position))

                    # 更新领导者任期记录
                    self._update_leader_tenure(temp_particle, particle_idx, temp_particle_id)

                # 计算预测位置的拥挤度
                # 首先将临时粒子添加到存档副本中
                temp_archive = self.archive.copy()
                temp_archive.append(temp_particle)

                # 计算拥挤度
                temp_fitnesses = [a.best_fitness for a in temp_archive]
                crowding_distances = self._calculate_crowding_distance(temp_fitnesses)
                predicted_crowding = crowding_distances[-1]  # 最后一个是临时粒子

                # 对于每个周期，追求不同的拥挤度目标
                # 随机化目标拥挤度 - 在平均值周围波动，促进前沿均衡探索
                target_crowding = self.avg_leader_crowding * random.uniform(0.7, 1.3)  # 增加波动范围

                # 检测到停滞时偶尔进行极端探索
                if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 5 and random.random() < 0.3:
                    # 以30%概率选择极端目标 - 非常低或非常高的拥挤度
                    target_crowding = self.avg_leader_crowding * (2.0 if random.random() < 0.5 else 0.5)

                # 搜索适合的预测位置
                search_step = 1
                max_search_steps = 8  # 减少最大搜索步数以避免过度调整

                # 只有在差异较大且未达到最大搜索步数时才调整
                # 注意：增加接受阈值，减少过度微调
                while abs(
                        predicted_crowding - target_crowding) > 0.3 * target_crowding and search_step < max_search_steps:
                    # 确定搜索方向 - 如果拥挤度太高则前进，太低则后退
                    direction = 1 if predicted_crowding > target_crowding else -1

                    # 沿趋势线移动
                    for dim in range(self.problem.n_var):
                        # 每个维度使用不同的随机因子，增加多样性
                        random_factor = random.uniform(0.8, 1.2)

                        # 扰动大小随搜索步骤增加
                        step_size = search_step * 0.1 * random_factor

                        # 沿拟合曲线方向调整，但加入随机扰动
                        delta = (self.problem.xu[dim] - self.problem.xl[dim]) * step_size * direction
                        predicted_position[dim] += delta

                    # 确保在合法范围内
                    predicted_position = np.clip(predicted_position, self.problem.xl, self.problem.xu)

                    # 重新评估
                    predicted_fitness = np.array(self.problem.evaluate(predicted_position))
                    temp_particle.position = predicted_position.copy()
                    temp_particle.best_position = predicted_position.copy()
                    temp_particle.fitness = predicted_fitness.copy()
                    temp_particle.best_fitness = predicted_fitness.copy()

                    # 更新拥挤度
                    temp_archive[-1] = temp_particle
                    temp_fitnesses = [a.best_fitness for a in temp_archive]
                    crowding_distances = self._calculate_crowding_distance(temp_fitnesses)
                    predicted_crowding = crowding_distances[-1]

                    search_step += 1

                    # 如果启用任期限制，检查更新后的粒子是否在黑名单中
                    if self.max_consecutive_leadership > 0:
                        temp_particle_id = hash(tuple(temp_particle.position))

                        # 检查更新后的粒子是否在黑名单中
                        if temp_particle_id in self.leader_blacklist and search_step < max_search_steps:
                            # 添加额外的随机扰动以避开黑名单
                            for dim in range(self.problem.n_var):
                                perturbation = (self.problem.xu[dim] - self.problem.xl[dim]) * 0.03 * random.uniform(-1,
                                                                                                                     1)
                                temp_particle.position[dim] += perturbation
                                temp_particle.best_position[dim] += perturbation

                            # 确保在边界内
                            temp_particle.position = np.clip(temp_particle.position, self.problem.xl, self.problem.xu)
                            temp_particle.best_position = np.clip(temp_particle.best_position, self.problem.xl,
                                                                  self.problem.xu)

                            # 重新评估
                            temp_particle.fitness = np.array(self.problem.evaluate(temp_particle.position))
                            temp_particle.best_fitness = temp_particle.fitness.copy()

                # 记录这个预测领导者
                leader_record = (temp_particle.position.copy(), temp_particle.fitness.copy(), predicted_crowding)
                self.leader_history.append(leader_record)
                self.current_gen_leaders.append(leader_record)

                # 返回预测位置作为领导者
                return temp_particle

            except Exception as e:
                print(f"预测领导者时出错: {e}")
                # 出错时回退到标准选择
                leader = self._crowding_distance_leader(particle)

                # 当启用任期限制时，检查领导者连续任期限制
                if self.max_consecutive_leadership > 0:
                    leader_id = self._get_particle_identifier(leader)

                    # 如果领导者已经在黑名单中，选择另一个领导者
                    if leader_id in self.leader_blacklist:
                        alternate_leader = self._find_alternate_leader(particle)
                        if alternate_leader:
                            leader = alternate_leader
                            leader_id = self._get_particle_identifier(leader)

                    # 更新领导者任期记录
                    self._update_leader_tenure(leader, particle_idx)

                # 记录这个备用领导者
                leader_position = leader.best_position.copy()
                leader_fitness = leader.best_fitness.copy()

                archive_fitnesses = [a.best_fitness for a in self.archive]
                crowding_distances = self._calculate_crowding_distance(archive_fitnesses)

                leader_idx = None
                for i, a in enumerate(self.archive):
                    if np.array_equal(a.best_position, leader_position):
                        leader_idx = i
                        break

                if leader_idx is not None:
                    leader_crowding = crowding_distances[leader_idx]
                    leader_record = (leader_position, leader_fitness, leader_crowding)
                    self.leader_history.append(leader_record)
                    self.current_gen_leaders.append(leader_record)

                return leader

    def _update_leader_tenure(self, leader, particle_idx=None, leader_id=None):
        """更新领导者的任期记录 - 仅当max_consecutive_leadership > 0时执行"""
        if self.max_consecutive_leadership <= 0:
            return  # 如果关闭了任期限制，直接返回

        if leader_id is None:
            leader_id = self._get_particle_identifier(leader, particle_idx)

        # 更新领导者任期记录
        if leader_id in self.current_leaders:
            self.current_leaders[leader_id] += 1
            # 检查是否达到最大连续任期限制
            if self.current_leaders[leader_id] >= self.max_consecutive_leadership:
                self.leader_blacklist.add(leader_id)
        else:
            # 新领导者，初始化任期计数
            self.current_leaders[leader_id] = 1

    def _find_alternate_leader(self, particle):
        """寻找一个不在黑名单中的替代领导者 - 仅当max_consecutive_leadership > 0时执行"""
        if self.max_consecutive_leadership <= 0:
            return None  # 如果关闭了任期限制，不需要寻找替代领导者

        if not self.archive:
            return None

        # 从存档中随机选择不在黑名单中的候选领导者
        eligible_leaders = []
        for archive_particle in self.archive:
            leader_id = self._get_particle_identifier(archive_particle)
            if leader_id not in self.leader_blacklist:
                eligible_leaders.append(archive_particle)

        if eligible_leaders:
            # 如果有多个合格的候选者，根据拥挤度选择
            if len(eligible_leaders) > 1:
                # 计算拥挤度
                fitnesses = [p.best_fitness for p in eligible_leaders]
                crowding_distances = self._calculate_crowding_distance(fitnesses)

                # 从拥挤度高的领导者中随机选择 (偏好拥挤度高的领导者)
                # 使用锦标赛选择方法
                tournament_size = min(5, len(eligible_leaders))
                selected_indices = np.random.choice(len(eligible_leaders), tournament_size, replace=False)

                max_crowding = -float('inf')
                best_idx = -1

                for idx in selected_indices:
                    if crowding_distances[idx] > max_crowding:
                        max_crowding = crowding_distances[idx]
                        best_idx = idx

                return eligible_leaders[best_idx]
            else:
                # 只有一个合格的候选者
                return eligible_leaders[0]
        else:
            # 如果所有粒子都在黑名单中
            # 可以临时解除限制允许一个粒子再次成为领导者
            if self.leader_blacklist:
                # 随机选择一个黑名单中的粒子解除限制
                freed_leader_id = random.choice(list(self.leader_blacklist))
                self.leader_blacklist.remove(freed_leader_id)

                # 重置该粒子的连续任期计数
                self.current_leaders[freed_leader_id] = 0

                # 在存档中找到对应的粒子
                for archive_particle in self.archive:
                    if self._get_particle_identifier(archive_particle) == freed_leader_id:
                        return archive_particle

            # 如果没有找到合适的粒子，随机选择一个存档中的粒子
            return random.choice(self.archive) if self.archive else None

    def _select_leader_from_swarm(self, particle, particle_idx=None):
        """从粒子群中选择领导者"""
        # 获取非支配解
        non_dominated = []
        for p in self.particles:
            is_dominated = False
            for other in self.particles:
                if self._dominates(other.best_fitness, p.best_fitness):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p)

        if not non_dominated:
            return particle

        # 当启用领导者任期限制时，过滤掉已达到最大连续任期的领导者
        if self.max_consecutive_leadership > 0:
            eligible_leaders = []
            for p in non_dominated:
                leader_id = self._get_particle_identifier(p, particle_idx)
                if leader_id not in self.leader_blacklist:
                    eligible_leaders.append(p)

            # 如果没有合格的领导者，使用所有非支配解
            if not eligible_leaders:
                eligible_leaders = non_dominated

            # 随机选择一个非支配解
            selected_leader = random.choice(eligible_leaders)

            # 更新所选领导者的任期记录
            self._update_leader_tenure(selected_leader, particle_idx)

            return selected_leader
        else:
            # 领导者任期限制未启用，直接随机选择
            return random.choice(non_dominated)

    def _crowding_distance_leader(self, particle):
        """基于拥挤度选择领导者"""
        if not self.archive: return None
        if len(self.archive) <= 1:
            return self.archive[0]

        # 锦标赛选择
        tournament_size = min(7, len(self.archive))
        candidates_idx = np.random.choice(len(self.archive), tournament_size, replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算拥挤度
        fitnesses = [c.best_fitness for c in candidates]
        crowding_distances = self._calculate_crowding_distance(fitnesses)

        max_idx_in_candidates = np.argmax(crowding_distances)
        return candidates[max_idx_in_candidates]

    def _update_archive(self):
        """更新外部存档 - 使用距离阈值替代严格相等检查"""
        # 将当前粒子的个体最优位置添加到存档中
        for particle in self.particles:
            is_dominated = False
            archive_copy = self.archive.copy()

            # 检查是否被存档中的解支配
            for solution in archive_copy:
                if self._dominates(solution.best_fitness, particle.best_fitness):
                    is_dominated = True
                    break
                # 检查是否支配存档中的解
                elif self._dominates(particle.best_fitness, solution.best_fitness):
                    self.archive.remove(solution)

            # 如果不被支配，检查与存档中解的距离
            if not is_dominated:
                # 计算最小距离，使用欧几里得距离
                min_distance = float('inf')
                for solution in self.archive:
                    dist = np.linalg.norm(particle.best_position - solution.best_position)
                    min_distance = min(min_distance, dist)

                # 只有当距离大于阈值时才添加到存档
                if min_distance > self.distance_threshold or len(self.archive) == 0:
                    # 深拷贝粒子
                    archive_particle = Particle(particle.dimensions, particle.bounds)
                    archive_particle.position = particle.best_position.copy()
                    archive_particle.best_position = particle.best_position.copy()
                    archive_particle.fitness = particle.best_fitness.copy()
                    archive_particle.best_fitness = particle.best_fitness.copy()

                    self.archive.append(archive_particle)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    def _prune_archive(self):
        """使用拥挤度排序修剪存档"""
        # 计算拥挤度
        crowding_distances = self._calculate_crowding_distance([a.best_fitness for a in self.archive])

        # 按拥挤度排序并保留前archive_size个
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    def _calculate_crowding_distance(self, fitnesses):
        """计算拥挤度"""
        n = len(fitnesses)
        if n <= 2:
            return [float('inf')] * n

        # 将fitnesses转换为numpy数组
        points = np.array(fitnesses)

        # 初始化距离
        distances = np.zeros(n)

        # 对每个目标
        for i in range(self.problem.n_obj):
            # 按该目标排序
            idx = np.argsort(points[:, i])

            # 边界点设为无穷
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')

            # 计算中间点
            if n > 2:
                # 目标范围
                f_range = points[idx[-1], i] - points[idx[0], i]

                # 避免除以零
                if f_range > 0:
                    for j in range(1, n - 1):
                        distances[idx[j]] += (points[idx[j + 1], i] - points[idx[j - 1], i]) / f_range

        return distances

    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2（最小化问题）"""
        # 至少一个目标更好，其他不差
        better = False
        for i in range(len(fitness1)):
            if fitness1[i] > fitness2[i]:  # 假设最小化
                return False
            if fitness1[i] < fitness2[i]:
                better = True

        return better

    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        if self.use_archive and self.archive:
            return np.array([p.best_fitness for p in self.archive])
        else:
            # 从粒子群中提取非支配解
            non_dominated = []
            for p in self.particles:
                if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                    non_dominated.append(p.best_fitness)
            return np.array(non_dominated)

    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        if self.use_archive and self.archive:
            return np.array([p.best_position for p in self.archive])
        else:
            # 从粒子群中提取非支配解
            non_dominated = []
            for p in self.particles:
                if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                    non_dominated.append(p.best_position)
            return np.array(non_dominated)

    def _track_performance(self, iteration):
        """跟踪性能指标"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class MOPSO:
    """基础多目标粒子群优化算法 (支持动态参数版本)"""

    # 修改 __init__ 方法以接受动态参数范围
    def __init__(self, problem, pop_size=100, max_iterations=100,
                 w_init=0.9, w_end=0.4,  # 惯性权重初始/结束值
                 c1_init=1.5, c1_end=1.5,  # 个体学习因子初始/结束值 (默认保持不变)
                 c2_init=1.5, c2_end=1.5,  # 社会学习因子初始/结束值 (默认保持不变)
                 use_archive=True, archive_size=100):  # 添加 archive_size
        """
        初始化MOPSO算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_iterations: 最大迭代次数
        w_init, w_end: 惯性权重的初始和结束值
        c1_init, c1_end: 个体学习因子的初始和结束值
        c2_init, c2_end: 社会学习因子的初始和结束值
        use_archive: 是否使用外部存档
        archive_size: 存档大小限制
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        # 存储动态参数范围
        self.w_init = w_init
        self.w_end = w_end
        self.c1_init = c1_init
        self.c1_end = c1_end
        self.c2_init = c2_init
        self.c2_end = c2_end
        # 其他参数
        self.use_archive = use_archive
        self.archive_size = archive_size  # 确保处理 archive_size

        # 粒子群和外部存档
        self.particles = []
        self.archive = []
        # 保持与原MOPSO相同的领导者选择和存档修剪逻辑
        self.leader_selector = self._crowding_distance_leader
        # self.archive_size = 100 # archive_size 从参数传入

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化粒子群
        bounds = list(zip(self.problem.xl, self.problem.xu))
        self.particles = [Particle(self.problem.n_var, bounds) for _ in range(self.pop_size)]

        # 初始化存档
        self.archive = []

        # 初始评估
        for particle in self.particles:
            # --- 修改 TP9/TP10 后 evaluate 的返回值 ---
            evaluation_result = self.problem.evaluate(particle.position)
            if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                # 如果返回的是 (objectives, constraints)，只取 objectives
                objectives = evaluation_result[0]
            else:
                # 否则，假设只返回 objectives
                objectives = evaluation_result
            particle.fitness = np.array(objectives)  # 使用 numpy 数组存储适应度
            # --- 修改结束 ---
            particle.best_position = particle.position.copy()  # 初始化 best_position
            particle.best_fitness = particle.fitness.copy()  # 使用 fitness 初始化 best_fitness

        # 初始化外部存档
        if self.use_archive:
            self._update_archive()  # 使用初始 pbest 更新存档

        # 优化迭代
        for iteration in range(self.max_iterations):
            if verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}/{self.max_iterations}，当前存档大小: {len(self.archive)}")

            # --- 计算当前迭代的动态参数 ---
            progress = iteration / self.max_iterations
            current_w = self.w_init - (self.w_init - self.w_end) * progress
            current_c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
            current_c2 = self.c2_init + (self.c2_end - self.c2_init) * progress
            # --- 参数计算结束 ---

            # 对每个粒子
            for particle in self.particles:
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self.leader_selector(particle)
                else:
                    # 如果没有存档或不使用存档，需要一个备选策略
                    # 可以从种群本身的非支配解中选，或随机选一个粒子
                    # 这里我们调用与 CASMOPSO 类似的内部选择函数 (如果需要，可单独实现)
                    leader = self._select_leader_from_swarm(particle)  # 确保这个方法存在或被正确调用

                # 如果没有领导者可选 (例如初始时存档为空且无法从种群选)
                if leader is None:
                    # 可以让粒子使用自己的 pbest 作为引导，或者跳过更新
                    # 这里选择让粒子使用自己的 pbest
                    leader = particle  # 让它飞向自己的历史最优

                # 更新速度和位置 (使用当前计算出的 w, c1, c2)
                particle.update_velocity(leader.best_position, current_w, current_c1, current_c2)
                particle.update_position()

                # 评估新位置
                # --- 同样处理 evaluate 的返回值 ---
                evaluation_result = self.problem.evaluate(particle.position)
                if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                    objectives = evaluation_result[0]
                else:
                    objectives = evaluation_result
                particle.fitness = np.array(objectives)  # 更新 fitness
                # --- 修改结束 ---

                # 更新个体最优 (pbest)
                # 需要比较 fitness 和 best_fitness
                if self._dominates(particle.fitness, particle.best_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness.copy()
                # 如果是非支配关系，可以考虑随机更新或不更新
                elif not self._dominates(particle.best_fitness, particle.fitness):
                    # 如果两个解互不支配，可以随机选择是否更新 pbest
                    if random.random() < 0.5:
                        particle.best_position = particle.position.copy()
                        particle.best_fitness = particle.fitness.copy()

            # 更新外部存档 (使用更新后的 pbest)
            if self.use_archive:
                self._update_archive()

            # 跟踪性能指标
            if tracking and iteration % 1 == 0:
                # 确保 _track_performance 使用的是存档或种群的 pbest
                self._track_performance(iteration)

        # 最终评估
        if tracking:
            self._track_performance(self.max_iterations - 1)

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    # --- _update_archive 方法保持不变，但确保它使用 best_fitness ---
    def _update_archive(self):
        """更新外部存档"""
        # 将当前粒子的个体最优位置添加到存档中
        current_pbest_positions = [p.best_position for p in self.particles]
        current_pbest_fitness = [p.best_fitness for p in self.particles]

        combined_solutions = []
        # 添加当前存档
        if self.archive:
            combined_solutions.extend([(p.best_position, p.best_fitness) for p in self.archive])
        # 添加当前种群的 pbest
        combined_solutions.extend(zip(current_pbest_positions, current_pbest_fitness))

        # 提取非支配解来构建新存档
        new_archive_solutions = []
        if combined_solutions:
            positions = np.array([s[0] for s in combined_solutions])
            fitnesses = np.array([s[1] for s in combined_solutions])

            # 查找非支配解的索引
            is_dominated = np.zeros(len(fitnesses), dtype=bool)
            for i in range(len(fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(fitnesses[i], fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(fitnesses[j], fitnesses[i]):
                        is_dominated[i] = True
                        break  # i被支配，跳出内层循环

            non_dominated_indices = np.where(~is_dominated)[0]

            # 重新创建存档粒子列表
            self.archive = []
            unique_positions = set()  # 用于去重
            for idx in non_dominated_indices:
                pos_tuple = tuple(positions[idx])
                if pos_tuple not in unique_positions:
                    archive_particle = Particle(self.problem.n_var, list(zip(self.problem.xl, self.problem.xu)))
                    archive_particle.position = positions[idx].copy()  # 当前位置设为最优位置
                    archive_particle.best_position = positions[idx].copy()
                    archive_particle.fitness = fitnesses[idx].copy()  # 当前适应度设为最优适应度
                    archive_particle.best_fitness = fitnesses[idx].copy()
                    self.archive.append(archive_particle)
                    unique_positions.add(pos_tuple)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    # --- _prune_archive 方法保持不变 ---
    def _prune_archive(self):
        """使用拥挤度排序修剪存档"""
        if len(self.archive) <= self.archive_size:
            return
        # 使用拥挤度排序保留前N个解
        fitnesses = [a.best_fitness for a in self.archive]
        crowding_distances = self._calculate_crowding_distance(fitnesses)

        # 按拥挤度降序排序
        sorted_indices = np.argsort(crowding_distances)[::-1]
        # 保留前 archive_size 个
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    # --- _crowding_distance_leader 方法保持不变 ---
    def _crowding_distance_leader(self, particle):
        """基于拥挤度选择领导者"""
        if not self.archive:  # 如果存档为空，返回 None 或其他策略
            return None  # 或者返回粒子自身？ particle
        if len(self.archive) == 1:
            return self.archive[0]

        # 随机选择候选 (锦标赛选择)
        tournament_size = min(3, len(self.archive))  # 锦标赛大小
        candidates_idx = np.random.choice(len(self.archive), tournament_size, replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算候选的拥挤度
        fitnesses = [c.best_fitness for c in candidates]
        crowding_distances = self._calculate_crowding_distance(fitnesses)

        # 选择拥挤度最大的
        best_idx_in_candidates = np.argmax(crowding_distances)
        return candidates[best_idx_in_candidates]

    # --- 添加 _select_leader_from_swarm (如果需要) ---
    def _select_leader_from_swarm(self, particle):
        """从粒子群的pbest中选择领导者 (如果存档为空或不使用)"""
        # 提取当前种群的所有 pbest fitness
        pbest_fitnesses = [p.best_fitness for p in self.particles]
        pbest_positions = [p.best_position for p in self.particles]

        # 找出非支配的 pbest
        non_dominated_indices = []
        is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
        for i in range(len(pbest_fitnesses)):
            if is_dominated[i]: continue
            for j in range(i + 1, len(pbest_fitnesses)):
                if is_dominated[j]: continue
                if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                    is_dominated[j] = True
                elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                    is_dominated[i] = True
                    break
            if not is_dominated[i]:
                non_dominated_indices.append(i)

        if not non_dominated_indices:
            # 如果没有非支配解 (不太可能发生，除非所有解都相同)
            # 返回粒子自身或者随机选一个
            return particle  # 让它飞向自己的历史最优

        # 从非支配的 pbest 中随机选择一个作为领导者
        leader_idx = random.choice(non_dominated_indices)
        # 返回一个临时的 "leader" 对象，包含 best_position
        # 或者直接返回 pbest_position? update_velocity 需要 best_position
        temp_leader = Particle(self.problem.n_var, [])  # 临时对象
        temp_leader.best_position = pbest_positions[leader_idx]
        return temp_leader

    # --- _calculate_crowding_distance 方法保持不变 ---
    def _calculate_crowding_distance(self, fitnesses):
        n = len(fitnesses)
        if n <= 2:
            return [float('inf')] * n
        points = np.array(fitnesses)
        distances = np.zeros(n)
        for i in range(self.problem.n_obj):
            idx = np.argsort(points[:, i])
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')
            if n > 2:
                f_range = points[idx[-1], i] - points[idx[0], i]
                if f_range > 1e-8:  # 避免除零
                    for j in range(1, n - 1):
                        distances[idx[j]] += (points[idx[j + 1], i] - points[idx[j - 1], i]) / f_range
        return distances

    # --- _dominates 方法保持不变 ---
    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2"""
        f1 = np.asarray(fitness1)  # 确保是数组
        f2 = np.asarray(fitness2)  # 确保是数组
        # 检查维度是否匹配，以防万一
        if f1.shape != f2.shape:
            print(f"警告: 支配比较时维度不匹配: {f1.shape} vs {f2.shape}")
            return False  # 或者抛出错误
        # 至少一个目标严格更好，且没有目标更差
        return np.all(f1 <= f2) and np.any(f1 < f2)

    # --- _get_pareto_front 方法保持不变 ---
    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        if self.use_archive and self.archive:
            # 确保返回的是 best_fitness
            return np.array([p.best_fitness for p in self.archive])
        else:
            # 从粒子群的 pbest 中提取非支配解
            pbest_fitnesses = [p.best_fitness for p in self.particles]
            if not pbest_fitnesses: return np.array([])  # 处理空种群

            non_dominated = []
            is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
            for i in range(len(pbest_fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(pbest_fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                        is_dominated[i] = True
                        break
                if not is_dominated[i]:
                    non_dominated.append(pbest_fitnesses[i])
            return np.array(non_dominated)

    # --- _get_pareto_set 方法保持不变 ---
    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        if self.use_archive and self.archive:
            # 确保返回的是 best_position
            return np.array([p.best_position for p in self.archive])
        else:
            # 从粒子群的 pbest 中提取非支配解
            pbest_fitnesses = [p.best_fitness for p in self.particles]
            pbest_positions = [p.best_position for p in self.particles]
            if not pbest_fitnesses: return np.array([])

            non_dominated_indices = []
            is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
            for i in range(len(pbest_fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(pbest_fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                        is_dominated[i] = True
                        break
                if not is_dominated[i]:
                    non_dominated_indices.append(i)
            return np.array([pbest_positions[i] for i in non_dominated_indices])

    # --- _track_performance 方法保持不变 ---
    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # SP
        if len(front) > 1:
            self.tracking['metrics']['sp'].append(PerformanceIndicators.spacing(front))
        else:
            self.tracking['metrics']['sp'].append(float('nan'))
        # IGDF
        if true_front is not None and len(front) > 0:
            self.tracking['metrics']['igdf'].append(PerformanceIndicators.igdf(front, true_front))
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))
        # IGDX
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            self.tracking['metrics']['igdx'].append(PerformanceIndicators.igdx(solution_set, true_set))
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))
        # RPSP
        if true_front is not None and len(front) > 0:
            self.tracking['metrics']['rpsp'].append(PerformanceIndicators.rpsp(front, true_front))
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))
        # HV
        # 计算HV指标
        if len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


# ====================== 多目标遗传算法======================

class MOEAD:
    """基于分解的多目标进化算法(MOEA/D)"""

    def __init__(self, problem, pop_size=300, max_generations=300, T=20, delta=0.9, nr=2):
        """
        初始化MOEA/D算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        T: 邻居大小
        delta: 邻居选择概率
        nr: 更新的最大解数量
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.T = min(T, pop_size)  # 邻居数量
        self.delta = delta  # 从邻居中选择父代的概率
        self.nr = nr  # 每个子代最多更新的解数量

        # 种群
        self.population = []
        self.weights = []
        self.neighbors = []
        self.z = None  # 参考点

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [],
                'igdx': [],
                'rpsp': [],
                'hv': [],
                'sp': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化权重向量和邻居
        self._initialize_weights()
        self._initialize_neighbors()

        # 初始化种群
        self._initialize_population()

        # 初始化理想点
        self.z = np.min([ind['objectives'] for ind in self.population], axis=0)

        # 迭代优化
        for gen in range(self.max_generations):
            if verbose and gen % 10 == 0:
                print(f"迭代 {gen}/{self.max_generations}")

            # 对每个权重向量
            for i in range(self.pop_size):
                # 选择父代
                if np.random.random() < self.delta:
                    # 从邻居中选择
                    p_indices = np.random.choice(self.neighbors[i], 2, replace=False)
                else:
                    # 从整个种群中选择
                    p_indices = np.random.choice(self.pop_size, 2, replace=False)

                # 产生子代（交叉+变异）
                child = self._reproduction(p_indices)

                # 评估子代
                child_obj = np.array(self.problem.evaluate(child))

                # 更新理想点
                self.z = np.minimum(self.z, child_obj)

                # 更新邻居解
                self._update_neighbors(i, child, child_obj)

            # 跟踪性能指标
            if tracking and gen % 1 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    def _initialize_weights(self):
        """初始化权重向量，确保生成足够数量的向量"""
        if self.problem.n_obj == 3:
            # 三目标问题使用改进的方法生成权重
            self.weights = self._generate_uniform_weights(self.problem.n_obj, self.pop_size)
        else:
            # 其他维度使用随机权重
            self.weights = np.random.random((self.pop_size, self.problem.n_obj))
            # 归一化
            self.weights = self.weights / np.sum(self.weights, axis=1)[:, np.newaxis]

    def _generate_uniform_weights(self, n_obj, pop_size):
        """改进的权重向量生成方法，确保生成足够的权重"""

        # 添加组合数计算函数
        def choose(n, k):
            """计算组合数C(n,k)"""
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1

            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result

        if n_obj == 3:
            # 计算合适的H值
            H = 1
            while choose(H + n_obj - 1, n_obj - 1) < pop_size:
                H += 1

            # 生成权重向量
            weights = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    if k >= 0:  # 确保三个权重的和为H
                        weight = np.array([i, j, k], dtype=float) / H
                        weights.append(weight)

            # 如果生成的权重过多，随机选择
            if len(weights) > pop_size:
                indices = np.random.choice(len(weights), pop_size, replace=False)
                weights = [weights[i] for i in indices]

            return np.array(weights)
        else:
            # 对于其他维度，使用简单的均匀生成方法
            weights = []
            for _ in range(pop_size):
                weight = np.random.random(n_obj)
                weight = weight / np.sum(weight)  # 归一化
                weights.append(weight)

            return np.array(weights)

    def _generate_weight_vectors(self, n_obj, pop_size):
        """为三目标问题生成系统的权重向量"""
        # 确定每个维度上的点数
        h = int((pop_size * 2) ** (1.0 / n_obj))

        # 递归生成权重向量
        def _generate_recursive(n_remain, weights, depth, result):
            if depth == n_obj - 1:
                weights[depth] = n_remain / h
                result.append(weights.copy())
                return

            for i in range(n_remain + 1):
                weights[depth] = i / h
                _generate_recursive(n_remain - i, weights, depth + 1, result)

        weights_list = []
        _generate_recursive(h, np.zeros(n_obj), 0, weights_list)

        # 转换为numpy数组
        weights = np.array(weights_list)

        # 如果生成的权重向量过多，随机选择
        if len(weights) > pop_size:
            indices = np.random.choice(len(weights), pop_size, replace=False)
            weights = weights[indices]

        return weights

    def _initialize_neighbors(self):
        """初始化邻居关系，添加安全检查"""
        n = len(self.weights)
        self.neighbors = []

        # 计算权重向量之间的距离
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.sum((self.weights[i] - self.weights[j]) ** 2)

        # 调整邻居数量，确保不超过种群大小
        self.T = min(self.T, n - 1)

        # 对每个权重向量找到T个最近的邻居
        for i in range(n):
            self.neighbors.append(np.argsort(dist[i])[:self.T])

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for i in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = np.array(self.problem.evaluate(x))

            # 添加到种群
            self.population.append({
                'x': x,
                'objectives': objectives
            })

    def _reproduction(self, parent_indices):
        """产生子代"""
        # 获取父代
        parent1 = self.population[parent_indices[0]]['x']
        parent2 = self.population[parent_indices[1]]['x']

        # 模拟二进制交叉(SBX)
        child = np.zeros(self.problem.n_var)

        # 交叉
        for i in range(self.problem.n_var):
            if np.random.random() < 0.5:
                # 执行交叉
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    eta = 20  # 分布指数

                    # 计算beta值
                    beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                    alpha = 2.0 - beta ** (-eta - 1)
                    rand = np.random.random()

                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                    # 生成子代
                    child[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                    # 边界处理
                    child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))
                else:
                    child[i] = parent1[i]
            else:
                child[i] = parent1[i]

        # 多项式变异
        for i in range(self.problem.n_var):
            if np.random.random() < 1.0 / self.problem.n_var:
                eta_m = 20  # 变异分布指数

                delta1 = (child[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - child[i]) / (self.problem.xu[i] - self.problem.xl[i])

                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                child[i] = child[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))

        return child

    def _update_neighbors(self, idx, child_x, child_obj):
        """更新邻居解 - 标准MOEA/D实现"""
        # 原始更新代码
        count = 0
        perm = np.random.permutation(self.neighbors[idx])

        for j in perm:
            old_fit = self._tchebycheff(self.population[j]['objectives'], self.weights[j])
            new_fit = self._tchebycheff(child_obj, self.weights[j])

            if new_fit <= old_fit:
                self.population[j]['x'] = child_x.copy()
                self.population[j]['objectives'] = child_obj.copy()
                count += 1

            if count >= self.nr:
                break

    def _tchebycheff(self, objectives, weights):
        """计算切比雪夫距离"""
        return np.max(weights * np.abs(objectives - self.z))

    def _get_pareto_front(self):
        """获取Pareto前沿"""
        # 提取所有目标值
        objectives = np.array([ind['objectives'] for ind in self.population])

        # 提取非支配解
        is_dominated = np.full(self.pop_size, False)

        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i != j and not is_dominated[j]:
                    if self._dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break

        # 返回非支配解的目标值
        return objectives[~is_dominated]

    def _get_pareto_set(self):
        """获取Pareto解集"""
        # 提取所有目标值和解
        objectives = np.array([ind['objectives'] for ind in self.population])
        solutions = np.array([ind['x'] for ind in self.population])

        # 提取非支配解
        is_dominated = np.full(self.pop_size, False)

        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i != j and not is_dominated[j]:
                    if self._dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break

        # 返回非支配解的解集
        return solutions[~is_dominated]

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class NSGAII:
    """NSGA-II算法实现"""

    def __init__(self, problem, pop_size=100, max_generations=100,
                 pc=0.9,  # 交叉概率 (Crossover probability)
                 eta_c=20,  # SBX 交叉分布指数 (Distribution index for SBX)
                 pm_ratio=1.0,  # 变异概率因子 (pm = pm_ratio / n_var)
                 eta_m=20):  # 多项式变异分布指数 (Distribution index for polynomial mutation)
        """
        初始化NSGA-II算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        pc: 模拟二进制交叉 (SBX) 的概率
        eta_c: SBX 的分布指数
        pm_ratio: 变异概率 pm = pm_ratio / n_var (n_var 是变量数)
        eta_m: 多项式变异的分布指数
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        # --- 存储交叉和变异参数 ---
        self.pc = pc
        self.eta_c = eta_c
        # 计算实际的变异概率 pm (每个变量独立变异的概率)
        self.pm = pm_ratio / self.problem.n_var
        self.eta_m = eta_m
        # --- 参数存储结束 ---

        # 种群
        self.population = None

        # 性能指标跟踪
        self.tracking = {
            'iterations': [], 'fronts': [],
            'metrics': {'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []}
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化种群
        self.population = self._initialize_population()

        # 评估种群
        self._evaluate_population(self.population)

        # 非支配排序
        fronts = self._fast_non_dominated_sort(self.population)

        # 分配拥挤度 - 添加空前沿检查
        for front in fronts:
            if front:  # 确保前沿不为空
                self._crowding_distance_assignment(front)

        # 迭代优化
        for generation in range(self.max_generations):
            if verbose and generation % 10 == 0:
                print(f"迭代 {generation}/{self.max_generations}")

            # 选择
            parents = self._tournament_selection(self.population)

            # 交叉和变异
            offspring = self._crossover_and_mutation(parents)

            # 评估子代
            self._evaluate_population(offspring)

            # 合并种群
            combined = self.population + offspring

            # 非支配排序
            fronts = self._fast_non_dominated_sort(combined)

            # 分配拥挤度
            for front in fronts:
                if front:  # 确保前沿不为空
                    self._crowding_distance_assignment(front)

            # 环境选择
            self.population = self._environmental_selection(fronts)

            # 跟踪性能指标
            if tracking and generation % 1 == 0:
                self._track_performance(generation)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            # 随机生成个体
            individual = {}
            individual['x'] = np.array(
                [np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
            individual['rank'] = None
            individual['crowding_distance'] = None
            individual['objectives'] = None
            population.append(individual)

        return population

    def _evaluate_population(self, population):
        """评估种群"""
        for individual in population:
            if individual['objectives'] is None:
                individual['objectives'] = np.array(self.problem.evaluate(individual['x']))

    def _fast_non_dominated_sort(self, population):
        """快速非支配排序 - 改进版"""
        # 初始化
        fronts = [[]]  # 存储不同等级的前沿
        for p in population:
            p['domination_count'] = 0  # 被多少个体支配
            p['dominated_solutions'] = []  # 支配的个体

            for q in population:
                if self._dominates(p['objectives'], q['objectives']):
                    # p支配q
                    p['dominated_solutions'].append(q)
                elif self._dominates(q['objectives'], p['objectives']):
                    # q支配p
                    p['domination_count'] += 1

            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)

        # 生成其他前沿
        i = 0
        # 修复：添加边界检查确保i不会超出fronts的范围
        while i < len(fronts):
            next_front = []

            if not fronts[i]:  # 如果当前前沿为空，跳过
                i += 1
                continue

            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)

        # 移除空前沿
        fronts = [front for front in fronts if front]

        return fronts

    def _crowding_distance_assignment(self, front):
        """改进的拥挤度计算"""
        if not front:
            return

        n = len(front)
        for p in front:
            p['crowding_distance'] = 0.0

        # 提取目标值
        fitnesses = np.array([ind['objectives'] for ind in front])

        # 对每个目标
        for m in range(self.problem.n_obj):
            # 按目标排序
            sorted_indices = np.argsort(fitnesses[:, m])

            # 边界点设为无穷
            front[sorted_indices[0]]['crowding_distance'] = float('inf')
            front[sorted_indices[-1]]['crowding_distance'] = float('inf')

            # 计算中间点距离
            if n > 2:
                f_max = fitnesses[sorted_indices[-1], m]
                f_min = fitnesses[sorted_indices[0], m]

                # 避免除以零
                norm = max((f_max - f_min), 1e-10)

                for i in range(1, n - 1):
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    current_idx = sorted_indices[i]

                    delta = (fitnesses[next_idx, m] - fitnesses[prev_idx, m]) / norm
                    front[current_idx]['crowding_distance'] += delta

    def _tournament_selection(self, population):
        """锦标赛选择"""
        selected = []
        while len(selected) < self.pop_size:
            # 随机选择两个个体
            a = random.choice(population)
            b = random.choice(population)

            # 锦标赛比较
            if (a['rank'] < b['rank']) or \
                    (a['rank'] == b['rank'] and a['crowding_distance'] > b['crowding_distance']):
                selected.append(a.copy())
            else:
                selected.append(b.copy())

        return selected

    def _crossover_and_mutation(self, parents):
        """交叉和变异 - 使用 self 中的参数"""
        offspring = []
        n_var = self.problem.n_var
        xl = self.problem.xl
        xu = self.problem.xu

        # 确保进行偶数次交叉，生成 pop_size 个子代
        parent_indices = list(range(len(parents)))
        random.shuffle(parent_indices)  # 打乱父代顺序

        for i in range(0, self.pop_size, 2):
            # 选择父代索引，处理最后一个父代可能落单的情况
            idx1 = parent_indices[i]
            idx2 = parent_indices[i + 1] if (i + 1) < len(parents) else parent_indices[0]  # 落单则与第一个配对

            # 深拷贝父代以产生子代（避免修改原始父代）
            p1 = parents[idx1].copy()
            p2 = parents[idx2].copy()
            # 确保子代有独立的 'x' 副本
            p1['x'] = parents[idx1]['x'].copy()
            p2['x'] = parents[idx2]['x'].copy()

            # SBX交叉
            # 使用 self.pc 和 self.eta_c
            if random.random() < self.pc:
                for j in range(n_var):
                    if random.random() < 0.5:  # 对每个变量 50% 概率交叉
                        y1, y2 = p1['x'][j], p2['x'][j]
                        if abs(y1 - y2) > 1e-10:
                            if y1 > y2: y1, y2 = y2, y1  # 确保 y1 <= y2

                            rand = random.random()
                            beta = 1.0 + (2.0 * (y1 - xl[j]) / (y2 - y1)) if (y2 - y1) > 1e-10 else 1.0
                            alpha = 2.0 - beta ** -(self.eta_c + 1.0)
                            if rand <= (1.0 / alpha):
                                beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))

                            c1 = 0.5 * ((1.0 + beta_q) * y1 + (1.0 - beta_q) * y2)
                            c2 = 0.5 * ((1.0 - beta_q) * y1 + (1.0 + beta_q) * y2)

                            # 边界处理
                            c1 = np.clip(c1, xl[j], xu[j])
                            c2 = np.clip(c2, xl[j], xu[j])

                            # 随机分配给子代
                            if random.random() < 0.5:
                                p1['x'][j], p2['x'][j] = c1, c2
                            else:
                                p1['x'][j], p2['x'][j] = c2, c1

            # 多项式变异
            # 使用 self.pm 和 self.eta_m
            for child in [p1, p2]:
                for j in range(n_var):
                    if random.random() < self.pm:  # 使用 self.pm
                        y = child['x'][j]
                        delta1 = (y - xl[j]) / (xu[j] - xl[j]) if (xu[j] - xl[j]) > 1e-10 else 0.5
                        delta2 = (xu[j] - y) / (xu[j] - xl[j]) if (xu[j] - xl[j]) > 1e-10 else 0.5
                        delta1 = np.clip(delta1, 0, 1)  # 确保在[0,1]
                        delta2 = np.clip(delta2, 0, 1)  # 确保在[0,1]

                        rand = random.random()
                        mut_pow = 1.0 / (self.eta_m + 1.0)  # 使用 self.eta_m

                        if rand < 0.5:
                            xy = 1.0 - delta1
                            if xy < 0: xy = 0
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                            delta_q = val ** mut_pow - 1.0
                        else:
                            xy = 1.0 - delta2
                            if xy < 0: xy = 0
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                            delta_q = 1.0 - val ** mut_pow

                        y = y + delta_q * (xu[j] - xl[j])
                        child['x'][j] = np.clip(y, xl[j], xu[j])  # 边界处理

            # 重置子代的评估状态
            p1['objectives'] = None
            p1['rank'] = None
            p1['crowding_distance'] = None
            p2['objectives'] = None
            p2['rank'] = None
            p2['crowding_distance'] = None

            offspring.append(p1)
            # 确保只添加 pop_size 个子代
            if len(offspring) < self.pop_size:
                offspring.append(p2)

        return offspring[:self.pop_size]  # 返回精确 pop_size 个子代

    def _environmental_selection(self, fronts):
        """环境选择"""
        # 选择下一代种群
        next_population = []
        i = 0

        # 添加完整的前沿 - 增加额外的边界检查
        while i < len(fronts) and fronts[i] and len(next_population) + len(fronts[i]) <= self.pop_size:
            next_population.extend(fronts[i])
            i += 1

        # 处理最后一个前沿
        if len(next_population) < self.pop_size and i < len(fronts) and fronts[i]:
            # 按拥挤度排序
            last_front = sorted(fronts[i], key=lambda x: x['crowding_distance'], reverse=True)

            # 添加拥挤度最大的个体
            next_population.extend(last_front[:self.pop_size - len(next_population)])

        return next_population

    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        # 提取非支配解
        fronts = self._fast_non_dominated_sort(self.population)
        return np.array([individual['objectives'] for individual in fronts[0]])

    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        # 提取非支配解
        fronts = self._fast_non_dominated_sort(self.population)
        return np.array([individual['x'] for individual in fronts[0]])

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class SPEA2:
    """强度Pareto进化算法2"""

    def __init__(self, problem, pop_size=100, archive_size=100, max_generations=100):
        """
        初始化SPEA2算法
        problem: 优化问题实例
        pop_size: 种群大小
        archive_size: 存档大小
        max_generations: 最大代数
        """
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations

        # 种群和存档
        self.population = []
        self.archive = []

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [],
                'igdx': [],
                'rpsp': [],
                'hv': [],
                'sp': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化种群
        self._initialize_population()

        # 初始化存档
        self.archive = []

        # 计算初始适应度
        self._calculate_fitness(self.population + self.archive)

        # 更新存档
        self._update_archive()

        # 迭代优化
        for gen in range(self.max_generations):
            if verbose and gen % 10 == 0:
                print(f"迭代 {gen}/{self.max_generations}，存档大小: {len(self.archive)}")

            # 环境选择
            mating_pool = self._environmental_selection()

            # 产生下一代
            offspring = self._generate_offspring(mating_pool)

            # 替换种群
            self.population = offspring

            # 计算适应度
            self._calculate_fitness(self.population + self.archive)

            # 更新存档
            self._update_archive()

            # 跟踪性能指标
            if tracking and gen % 1 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for _ in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = np.array(self.problem.evaluate(x))  # 确保是numpy数组

            # 添加到种群
            self.population.append({
                'x': x,
                'objectives': objectives,
                'fitness': 0.0,
                'strength': 0,
                'raw_fitness': 0.0,
                'distance': 0.0
            })

    def _calculate_fitness(self, combined_pop):
        """计算适应度"""
        # 计算每个个体支配的个体数量(strength)
        for p in combined_pop:
            p['strength'] = 0
            for q in combined_pop:
                if self._dominates(p['objectives'], q['objectives']):
                    p['strength'] += 1

        # 计算raw fitness(被支配情况)
        for p in combined_pop:
            p['raw_fitness'] = 0.0
            for q in combined_pop:
                if self._dominates(q['objectives'], p['objectives']):
                    p['raw_fitness'] += q['strength']

        # 计算密度信息
        for i, p in enumerate(combined_pop):
            # 计算到其他个体的距离
            distances = []
            p_obj = np.array(p['objectives'])  # 确保是numpy数组

            for j, q in enumerate(combined_pop):
                if i != j:
                    q_obj = np.array(q['objectives'])  # 确保是numpy数组
                    dist = np.sqrt(np.sum((p_obj - q_obj) ** 2))
                    distances.append(dist)

            # 找到第k个最近邻居的距离
            k = int(np.sqrt(len(combined_pop)))
            if len(distances) > k:
                distances.sort()
                p['distance'] = 1.0 / (distances[k] + 2.0)
            else:
                p['distance'] = 0.0

        # 最终适应度 = raw fitness + density
        for p in combined_pop:
            p['fitness'] = p['raw_fitness'] + p['distance']

    def _update_archive(self):
        """更新存档"""
        # 合并种群和存档
        combined = self.population + self.archive

        # 选择适应度小于1的个体(非支配解)
        new_archive = [p for p in combined if p['fitness'] < 1.0]

        # 如果非支配解太少
        if len(new_archive) < self.archive_size:
            # 按适应度排序
            remaining = [p for p in combined if p['fitness'] >= 1.0]
            remaining.sort(key=lambda x: x['fitness'])

            # 添加适应度最小的个体
            new_archive.extend(remaining[:self.archive_size - len(new_archive)])

        # 如果非支配解太多
        elif len(new_archive) > self.archive_size:
            # 基于密度截断
            while len(new_archive) > self.archive_size:
                self._remove_most_crowded(new_archive)

        # 更新存档
        self.archive = new_archive

    def _remove_most_crowded(self, archive):
        """移除最拥挤的个体"""
        # 计算所有个体间的距离
        if len(archive) <= 1:
            return

        min_dist = float('inf')
        min_i = 0
        min_j = 0

        for i in range(len(archive)):
            i_obj = np.array(archive[i]['objectives'])  # 确保是numpy数组

            for j in range(i + 1, len(archive)):
                j_obj = np.array(archive[j]['objectives'])  # 确保是numpy数组
                dist = np.sqrt(np.sum((i_obj - j_obj) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j

        # 找到距离其他个体更近的那个
        i_dist = 0.0
        j_dist = 0.0

        for k in range(len(archive)):
            if k != min_i and k != min_j:
                k_obj = np.array(archive[k]['objectives'])  # 确保是numpy数组
                i_obj = np.array(archive[min_i]['objectives'])  # 确保是numpy数组
                j_obj = np.array(archive[min_j]['objectives'])  # 确保是numpy数组

                i_dist += np.sqrt(np.sum((i_obj - k_obj) ** 2))
                j_dist += np.sqrt(np.sum((j_obj - k_obj) ** 2))

        # 移除最拥挤的个体
        if i_dist < j_dist:
            archive.pop(min_i)
        else:
            archive.pop(min_j)

    def _environmental_selection(self):
        """环境选择，选择用于交配的个体"""
        # 创建交配池
        mating_pool = []

        # 二元锦标赛选择
        for _ in range(self.pop_size):
            # 随机选择两个个体
            if len(self.archive) > 0:
                idx1 = np.random.randint(0, len(self.archive))
                idx2 = np.random.randint(0, len(self.archive))

                # 选择适应度更好的个体
                if self.archive[idx1]['fitness'] < self.archive[idx2]['fitness']:
                    mating_pool.append(self.archive[idx1])
                else:
                    mating_pool.append(self.archive[idx2])
            else:
                # 如果存档为空，从种群中选择
                idx1 = np.random.randint(0, len(self.population))
                idx2 = np.random.randint(0, len(self.population))

                if self.population[idx1]['fitness'] < self.population[idx2]['fitness']:
                    mating_pool.append(self.population[idx1])
                else:
                    mating_pool.append(self.population[idx2])

        return mating_pool

    def _generate_offspring(self, mating_pool):
        """生成子代"""
        offspring = []

        for _ in range(self.pop_size):
            # 选择父代
            if len(mating_pool) > 1:
                parent1_idx = np.random.randint(0, len(mating_pool))
                parent2_idx = np.random.randint(0, len(mating_pool))

                # 确保选择不同的父代
                while parent1_idx == parent2_idx:
                    parent2_idx = np.random.randint(0, len(mating_pool))

                parent1 = mating_pool[parent1_idx]['x']
                parent2 = mating_pool[parent2_idx]['x']
            else:
                # 如果交配池只有一个个体，复制它并添加变异
                parent1 = mating_pool[0]['x']
                parent2 = parent1.copy()

            # 模拟二进制交叉(SBX)
            child_x = np.zeros(self.problem.n_var)

            # 交叉
            for i in range(self.problem.n_var):
                if np.random.random() < 0.5:
                    # 执行交叉
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                        eta = 20  # 分布指数

                        # 计算beta值
                        beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                        alpha = 2.0 - beta ** (-eta - 1)
                        rand = np.random.random()

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                        # 生成子代
                        child_x[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                        # 边界处理
                        child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))
                    else:
                        child_x[i] = parent1[i]
                else:
                    child_x[i] = parent1[i]

            # 多项式变异
            for i in range(self.problem.n_var):
                if np.random.random() < 1.0 / self.problem.n_var:
                    eta_m = 20  # 变异分布指数

                    delta1 = (child_x[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                    delta2 = (self.problem.xu[i] - child_x[i]) / (self.problem.xu[i] - self.problem.xl[i])

                    rand = np.random.random()
                    mut_pow = 1.0 / (eta_m + 1.0)

                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                        delta_q = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                        delta_q = 1.0 - val ** mut_pow

                    child_x[i] = child_x[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                    child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))

            # 评估子代
            try:
                child_obj = np.array(self.problem.evaluate(child_x))  # 确保是numpy数组

                # 添加到子代种群
                offspring.append({
                    'x': child_x,
                    'objectives': child_obj,
                    'fitness': 0.0,
                    'strength': 0,
                    'raw_fitness': 0.0,
                    'distance': 0.0
                })
            except Exception as e:
                print(f"评估子代时出错: {e}")
                # 如果评估失败，添加一个随机解
                x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
                objectives = np.array(self.problem.evaluate(x))  # 确保是numpy数组
                offspring.append({
                    'x': x,
                    'objectives': objectives,
                    'fitness': 0.0,
                    'strength': 0,
                    'raw_fitness': 0.0,
                    'distance': 0.0
                })

        return offspring

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _get_pareto_front(self):
        """获取Pareto前沿"""
        # 返回存档中的非支配解的目标值
        non_dominated = [ind for ind in self.archive if ind['fitness'] < 1.0]
        if not non_dominated and self.archive:
            # 如果没有严格非支配解，使用整个存档
            non_dominated = self.archive
        return np.array([ind['objectives'] for ind in non_dominated])

    def _get_pareto_set(self):
        """获取Pareto解集"""
        # 返回存档中的非支配解的决策变量
        non_dominated = [ind for ind in self.archive if ind['fitness'] < 1.0]
        if not non_dominated and self.archive:
            # 如果没有严格非支配解，使用整个存档
            non_dominated = self.archive
        return np.array([ind['x'] for ind in non_dominated])

    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class GDE3:
    """Generalized Differential Evolution 3 (GDE3) 算法实现"""

    def __init__(self, problem, pop_size=100, max_generations=100,
                 F=0.5,  # 缩放因子 (Scaling Factor)
                 CR=0.9):  # 交叉概率 (Crossover Rate)
        """
        初始化 GDE3 算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        F: DE 的缩放因子 (通常在 [0.4, 1.0])
        CR: DE 的交叉概率 (通常在 [0, 1])
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR

        # 种群
        self.population = []  # 存储个体字典: {'x': array, 'objectives': array, 'rank': int, 'crowding_distance': float}

        # 性能指标跟踪
        self.tracking = {
            'iterations': [], 'fronts': [],
            'metrics': {'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []}
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 1. 初始化种群 P
        self.population = self._initialize_population()
        self._evaluate_population(self.population)  # 计算初始种群的目标值

        # 初始排序和拥挤度计算 (可选，但有助于跟踪)
        fronts = self._fast_non_dominated_sort(self.population)
        for front in fronts:
            if front:
                self._crowding_distance_assignment(front)

        # 迭代优化
        pbar = tqdm(range(self.max_generations), desc=f"Optimizing {self.problem.name} with {self.__class__.__name__}",
                    disable=not verbose)
        for generation in pbar:
            # 临时存储下一代候选解
            next_pop_candidates = []

            # 2. 对 P 中的每个个体 x_i (target vector)
            for i in range(self.pop_size):
                target_vector = self.population[i]

                # 3. 生成变异向量 v_i (使用 rand/1 策略)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector_x = self.population[a]['x'] + self.F * (self.population[b]['x'] - self.population[c]['x'])
                # 边界处理
                mutant_vector_x = np.clip(mutant_vector_x, self.problem.xl, self.problem.xu)

                # 4. 生成试验向量 u_i (二项式交叉)
                trial_vector_x = target_vector['x'].copy()
                j_rand = np.random.randint(self.problem.n_var)
                for j in range(self.problem.n_var):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial_vector_x[j] = mutant_vector_x[j]

                # 5. 评估试验向量 u_i
                trial_vector_obj = np.array(self.problem.evaluate(trial_vector_x))
                trial_vector = {'x': trial_vector_x, 'objectives': trial_vector_obj, 'rank': None,
                                'crowding_distance': None}

                # 6. 选择 (GDE3 核心)
                target_obj = target_vector['objectives']
                trial_obj = trial_vector['objectives']

                if self._dominates(trial_obj, target_obj):
                    # 试验向量支配目标向量 -> 保留试验向量
                    next_pop_candidates.append(trial_vector)
                elif not self._dominates(target_obj, trial_obj):
                    # 目标向量不支配试验向量 (包括 trial_obj == target_obj 或互不支配) -> 保留试验向量
                    next_pop_candidates.append(trial_vector)
                # else: (目标向量支配试验向量) -> 不保留试验向量，目标向量隐式地保留在下一轮合并中

            # 7. 合并父代和被选中的子代
            combined_population = self.population + next_pop_candidates

            # 8. 从合并种群中选择下一代 P_{g+1} (大小为 N)
            #    使用非支配排序和拥挤度
            fronts = self._fast_non_dominated_sort(combined_population)
            for front in fronts:
                if front:
                    self._crowding_distance_assignment(front)

            self.population = []
            front_idx = 0
            while len(self.population) + len(fronts[front_idx]) <= self.pop_size:
                self.population.extend(fronts[front_idx])
                front_idx += 1
                if front_idx >= len(fronts):  # 防止索引越界
                    break

            # 如果空间不足，根据拥挤度填充剩余位置
            if len(self.population) < self.pop_size and front_idx < len(fronts):
                remaining_needed = self.pop_size - len(self.population)
                # 对最后一个需要部分添加的前沿按拥挤度降序排序
                sorted_last_front = sorted(fronts[front_idx], key=lambda x: x['crowding_distance'], reverse=True)
                self.population.extend(sorted_last_front[:remaining_needed])

            # 跟踪性能指标 (每隔一定代数或最后一次记录)
            if tracking and (generation % 1 == 0 or generation == self.max_generations - 1):
                self._track_performance(generation)

            # 更新进度条
            if verbose and generation % 10 == 0:
                pbar.set_postfix({"PopSize": len(self.population)})

        pbar.close()
        if verbose:
            print(f"优化完成，最终种群大小: {len(self.population)}")

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            individual = {}
            individual['x'] = np.array(
                [np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
            individual['objectives'] = None
            individual['rank'] = None
            individual['crowding_distance'] = float('nan')  # 初始化为 NaN
            population.append(individual)
        return population

    def _evaluate_population(self, population):
        """评估种群中未评估的个体"""
        for individual in population:
            if individual['objectives'] is None:
                individual['objectives'] = np.array(self.problem.evaluate(individual['x']))

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2 (最小化问题)"""
        # 确保是 numpy 数组
        obj1 = np.asarray(obj1)
        obj2 = np.asarray(obj2)
        # 至少一个目标严格更好，且没有目标更差
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    # --- 复用 NSGA-II 的排序和拥挤度计算方法 ---
    # (确保这些方法在 GDE3 类中可用，或者从外部调用)
    # 这里直接复制过来以保证 GDE3 类独立可用

    def _fast_non_dominated_sort(self, population):
        """快速非支配排序 (同 NSGAII)"""
        fronts = [[]]
        for p in population:
            p['domination_count'] = 0
            p['dominated_solutions'] = []
            for q in population:
                # 确保比较目标值不为None
                if p['objectives'] is None or q['objectives'] is None: continue
                if self._dominates(p['objectives'], q['objectives']):
                    p['dominated_solutions'].append(q)
                elif self._dominates(q['objectives'], p['objectives']):
                    p['domination_count'] += 1
            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:  # 添加 fronts[i] 非空检查
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        # 移除可能的空前沿列表
        fronts = [f for f in fronts if f]
        return fronts

    def _crowding_distance_assignment(self, front):
        """拥挤度计算 (同 NSGAII)"""
        if not front:
            return
        n = len(front)
        for p in front:
            p['crowding_distance'] = 0.0

        num_objectives = self.problem.n_obj
        fitnesses = np.array([ind['objectives'] for ind in front])

        for m in range(num_objectives):
            sorted_indices = np.argsort(fitnesses[:, m])
            # 边界点设为无穷
            front[sorted_indices[0]]['crowding_distance'] = float('inf')
            front[sorted_indices[-1]]['crowding_distance'] = float('inf')

            if n > 2:
                f_max = fitnesses[sorted_indices[-1], m]
                f_min = fitnesses[sorted_indices[0], m]
                norm = max(f_max - f_min, 1e-10)  # 避免除零

                for i in range(1, n - 1):
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    current_idx = sorted_indices[i]
                    # 避免对已经是inf的点累加
                    if np.isfinite(front[current_idx]['crowding_distance']):
                        delta = (fitnesses[next_idx, m] - fitnesses[prev_idx, m]) / norm
                        front[current_idx]['crowding_distance'] += delta

    # --- 获取结果和性能跟踪 ---
    def _get_pareto_front(self):
        """获取算法生成的最终Pareto前沿 (Rank 0)"""
        final_fronts = self._fast_non_dominated_sort(self.population)
        if not final_fronts:  # 如果没有非支配解
            return np.array([])
        # 返回第一个前沿 (Rank 0) 的目标值
        return np.array([ind['objectives'] for ind in final_fronts[0]])

    def _get_pareto_set(self):
        """获取算法生成的最终Pareto解集 (Rank 0)"""
        final_fronts = self._fast_non_dominated_sort(self.population)
        if not final_fronts:
            return np.array([])
        # 返回第一个前沿 (Rank 0) 的决策变量
        return np.array([ind['x'] for ind in final_fronts[0]])

    def _track_performance(self, iteration):
        """跟踪性能指标 - 复用基类或其他算法的实现"""
        # 获取当前Pareto前沿和解集 (基于当前种群的Rank 0)
        current_front = self._get_pareto_front()  # 获取当前非支配前沿
        current_set = self._get_pareto_set()  # 获取当前非支配解集

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(current_front)  # 保存当前前沿

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # --- 计算各项指标 ---
        # SP (Spacing)
        if len(current_front) > 1:
            sp = PerformanceIndicators.spacing(current_front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # IGDF (Inverted Generational Distance - Front)
        if true_front is not None and len(current_front) > 0:
            igdf = PerformanceIndicators.igdf(current_front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # IGDX (Inverted Generational Distance - Set)
        if true_set is not None and current_set is not None and len(current_set) > 0:
            igdx = PerformanceIndicators.igdx(current_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # RPSP (r-Pareto Set Proximity)
        if true_front is not None and len(current_front) > 0:
            rpsp = PerformanceIndicators.rpsp(current_front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # HV (Hypervolume)
        if len(current_front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(current_front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(current_front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class MOGWO:
    """
    多目标灰狼优化算法 (Multi-Objective Grey Wolf Optimizer) - Rewritten Version
    Focuses on robust archive management and leader selection.
    """

    def __init__(self, problem, pop_size=100, max_iterations=100,
                 archive_size=100, a_init=2.0, a_final=0.0):
        """
        初始化 MOGWO 算法
        problem: 优化问题实例
        pop_size: 灰狼种群大小
        max_iterations: 最大迭代次数
        archive_size: 存档大小 (存储非支配解)
        a_init: 初始 a 参数值 (控制收敛因子)
        a_final: 最终 a 参数值
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size
        self.a_init = a_init
        self.a_final = a_final

        # 种群和存档
        # Wolves: list of dicts {'x': np.array, 'objectives': np.array}
        self.wolves = []
        # Archive: list of dicts {'x': np.array, 'objectives': np.array, 'crowding_distance': float}
        self.archive = []

        # 领导狼 (Alpha, Beta, Delta) - store the wolf dictionary
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []
            }
        }
        # Add logger for warnings/info
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 1. 初始化狼群
        self._initialize_population()

        # 2. 计算初始适应度并更新存档
        self._update_archive()  # Initial archive update

        # 3. 选择初始领导者
        self._update_leaders()

        # 迭代优化
        pbar = tqdm(range(self.max_iterations), desc=f"Optimizing {self.problem.name} with {self.__class__.__name__}",
                    disable=not verbose)
        for iteration in pbar:
            # 计算当前迭代的 a 值 (线性递减)
            a = self.a_init - (self.a_init - self.a_final) * (iteration / self.max_iterations)

            # 确保有领导者可选
            if not self.alpha_wolf or not self.beta_wolf or not self.delta_wolf:
                self.logger.warning(f"迭代 {iteration}: 领导者缺失，可能存档为空。尝试更新领导者。")
                self._update_leaders()
                if not self.alpha_wolf:  # 如果仍然没有领导者
                    self.logger.error(f"迭代 {iteration}: 无法选择领导者，终止优化。")
                    # 返回当前存档作为结果，或者可以抛出错误
                    return self._get_pareto_front()

            # 4. 更新每只狼的位置
            next_wolves = []  # Store newly generated wolves temporarily
            for i in range(self.pop_size):
                current_wolf = self.wolves[i]
                new_x = self._calculate_new_position(current_wolf['x'], a)

                # 边界处理
                new_x = np.clip(new_x, self.problem.xl, self.problem.xu)

                # 评估新位置
                new_objectives = np.array(self.problem.evaluate(new_x))

                # 创建新的 wolf 字典
                next_wolves.append({'x': new_x, 'objectives': new_objectives})

            # 用新生成的狼替换旧种群
            self.wolves = next_wolves

            # 5. 更新存档 (基于新的狼群和旧存档)
            self._update_archive()

            # 6. 更新领导狼 (从更新后的存档中选择)
            self._update_leaders()

            # 7. 跟踪性能指标
            if tracking:  # Track every iteration
                self._track_performance(iteration)

            # Update progress bar postfix
            if verbose:
                pbar.set_postfix({"ArchiveSize": len(self.archive)})

        pbar.close()
        if verbose:
            self.logger.info(f"优化完成，最终存档大小: {len(self.archive)}")

        # 获取Pareto前沿
        pareto_front = self._get_pareto_front()

        # 返回归一化后的Pareto前沿
        if hasattr(self.problem, 'pareto_front') and self.problem.pareto_front is not None:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front, self.problem.pareto_front)
        else:
            normalized_front = PerformanceIndicators.normalize_pareto_front(pareto_front)

        return normalized_front

    def _initialize_population(self):
        """初始化狼群"""
        self.wolves = []
        for _ in range(self.pop_size):
            x = np.random.uniform(self.problem.xl, self.problem.xu, self.problem.n_var)
            objectives = np.array(self.problem.evaluate(x))
            self.wolves.append({'x': x, 'objectives': objectives})
        self.logger.info(f"Initialized population with {len(self.wolves)} wolves.")

    def _calculate_new_position(self, current_x, a):
        """根据 Alpha, Beta, Delta 计算狼的新位置"""
        new_x = np.zeros_like(current_x)

        # Ensure leaders exist
        if not self.alpha_wolf or not self.beta_wolf or not self.delta_wolf:
            self.logger.warning("Calculating new position but leaders are missing!")
            # Fallback: return current position or random walk?
            # Returning current position might lead to stagnation
            # Let's try a small random perturbation
            return current_x + np.random.normal(0, 0.01, size=current_x.shape) * (self.problem.xu - self.problem.xl)

        alpha_pos = self.alpha_wolf['x']
        beta_pos = self.beta_wolf['x']
        delta_pos = self.delta_wolf['x']

        for j in range(self.problem.n_var):
            # Alpha influence
            r1_a, r2_a = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1_a - a
            C1 = 2 * r2_a
            D_alpha = abs(C1 * alpha_pos[j] - current_x[j])
            X1 = alpha_pos[j] - A1 * D_alpha

            # Beta influence
            r1_b, r2_b = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1_b - a
            C2 = 2 * r2_b
            D_beta = abs(C2 * beta_pos[j] - current_x[j])
            X2 = beta_pos[j] - A2 * D_beta

            # Delta influence
            r1_d, r2_d = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1_d - a
            C3 = 2 * r2_d
            D_delta = abs(C3 * delta_pos[j] - current_x[j])
            X3 = delta_pos[j] - A3 * D_delta

            # Combine influences
            new_x[j] = (X1 + X2 + X3) / 3.0

        return new_x

    def _dominates(self, obj1, obj2):
        """判断 obj1 是否支配 obj2 (最小化问题)"""
        # Ensure numpy arrays
        obj1 = np.asarray(obj1)
        obj2 = np.asarray(obj2)
        # Check for strict inequality in at least one objective
        # and no objective where obj1 is worse than obj2
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _update_archive(self):
        """更新存档，包含当前狼群和旧存档中的非支配解"""
        # 1. 合并当前狼群和存档
        combined_solutions = self.wolves + self.archive

        # 2. 找出所有非支配解
        non_dominated_solutions = []
        if not combined_solutions:  # Handle empty case
            self.archive = []
            return

        objectives_list = [sol['objectives'] for sol in combined_solutions]
        is_dominated = np.zeros(len(combined_solutions), dtype=bool)

        for i in range(len(combined_solutions)):
            if is_dominated[i]: continue
            for j in range(i + 1, len(combined_solutions)):
                if is_dominated[j]: continue
                if self._dominates(objectives_list[i], objectives_list[j]):
                    is_dominated[j] = True
                elif self._dominates(objectives_list[j], objectives_list[i]):
                    is_dominated[i] = True
                    break  # i is dominated, no need to check further for i

            if not is_dominated[i]:
                # Check for duplicates before adding (optional, based on position)
                is_duplicate = False
                # for nd_sol in non_dominated_solutions:
                #    if np.allclose(combined_solutions[i]['x'], nd_sol['x']):
                #        is_duplicate = True
                #        break
                # if not is_duplicate:
                non_dominated_solutions.append(combined_solutions[i])

        # 3. 更新存档
        self.archive = non_dominated_solutions

        # 4. 如果存档过大，进行修剪
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    def _calculate_crowding_distance(self, archive_solutions):
        """计算存档中解的拥挤度距离 (标准 NSGA-II 方法)"""
        # Input: list of solution dictionaries
        # Output: list of distances corresponding to input solutions
        n = len(archive_solutions)
        if n == 0:
            return []
        if n <= 2:
            # Assign infinite distance to encourage keeping few solutions
            for sol in archive_solutions:
                sol['crowding_distance'] = float('inf')
            return [float('inf')] * n

        # Extract objectives
        objectives = np.array([sol['objectives'] for sol in archive_solutions])
        n_obj = self.problem.n_obj

        # Initialize distances
        distances = np.zeros(n)

        # Reset crowding distance in dictionaries
        for sol in archive_solutions:
            sol['crowding_distance'] = 0.0

        # Calculate distance per objective
        for m in range(n_obj):
            # Sort by objective m
            # Create tuples of (index, value) to handle sorting
            indexed_objectives = sorted([(i, objectives[i, m]) for i in range(n)], key=lambda x: x[1])
            sorted_indices = [item[0] for item in indexed_objectives]

            # Assign infinite distance to boundaries
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            archive_solutions[sorted_indices[0]]['crowding_distance'] = float('inf')
            archive_solutions[sorted_indices[-1]]['crowding_distance'] = float('inf')

            # Range for normalization
            f_max = objectives[sorted_indices[-1], m]
            f_min = objectives[sorted_indices[0], m]
            f_range = f_max - f_min

            if f_range > 1e-10:  # Avoid division by zero
                # Calculate for intermediate solutions
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]

                    # Add normalized distance difference
                    delta = (objectives[next_idx, m] - objectives[prev_idx, m]) / f_range
                    distances[idx] += delta
                    # Update dictionary as well
                    archive_solutions[idx]['crowding_distance'] += delta

        # Return the calculated distances as a list
        return distances.tolist()

    def _prune_archive(self):
        """使用拥挤度距离修剪存档，保留最不拥挤的解"""
        if len(self.archive) <= self.archive_size:
            return  # No need to prune

        # Calculate crowding distance for the current archive members
        # This also adds/updates 'crowding_distance' key in the dictionaries
        self._calculate_crowding_distance(self.archive)

        # Sort the archive based on crowding distance (descending, higher is better)
        self.archive.sort(key=lambda sol: sol.get('crowding_distance', 0.0), reverse=True)

        # Keep only the top 'archive_size' solutions
        self.archive = self.archive[:self.archive_size]

    def _update_leaders(self):
        """
        从存档中选择 Alpha, Beta, Delta 狼.
        选择策略：优先选择拥挤度最高的（最不拥挤的）不同个体。
        """
        if not self.archive:
            self.logger.warning("Archive is empty, cannot select leaders.")
            # Keep previous leaders or set to None? Setting to None might be safer.
            self.alpha_wolf = None
            self.beta_wolf = None
            self.delta_wolf = None
            return

        # Ensure crowding distances are calculated/updated for the current archive
        self._calculate_crowding_distance(self.archive)

        # Sort archive by crowding distance (descending)
        sorted_archive = sorted(self.archive, key=lambda sol: sol.get('crowding_distance', 0.0), reverse=True)

        # Select top 3 unique leaders
        num_archive = len(sorted_archive)
        if num_archive >= 3:
            self.alpha_wolf = sorted_archive[0]
            self.beta_wolf = sorted_archive[1]
            self.delta_wolf = sorted_archive[2]
        elif num_archive == 2:
            self.alpha_wolf = sorted_archive[0]
            self.beta_wolf = sorted_archive[1]
            self.delta_wolf = sorted_archive[0]  # Repeat alpha as delta
            self.logger.warning("Archive size < 3, repeating leaders.")
        elif num_archive == 1:
            self.alpha_wolf = sorted_archive[0]
            self.beta_wolf = sorted_archive[0]  # Repeat alpha
            self.delta_wolf = sorted_archive[0]  # Repeat alpha
            self.logger.warning("Archive size < 3, repeating leaders.")
        # else: archive is empty, handled at the start

    def _get_pareto_front(self):
        """获取最终存档中的 Pareto 前沿 (目标值)"""
        if not self.archive:
            return np.array([])
        return np.array([sol['objectives'] for sol in self.archive])

    def _get_pareto_set(self):
        """获取最终存档中的 Pareto 解集 (决策变量)"""
        if not self.archive:
            return np.array([])
        return np.array([sol['x'] for sol in self.archive])

    def _track_performance(self, iteration):
        """跟踪性能指标 - (Copy from other algorithms in the framework)"""
        # Get current Pareto front and set from the archive
        current_front = self._get_pareto_front()
        current_set = self._get_pareto_set()

        # Save iteration and front
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(current_front)  # Save current non-dominated front

        # Get true front and set
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # --- Calculate metrics ---
        # SP (Spacing)
        if len(current_front) > 1:
            sp = PerformanceIndicators.spacing(current_front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # IGDF (Inverted Generational Distance - Front)
        if true_front is not None and len(current_front) > 0:
            igdf = PerformanceIndicators.igdf(current_front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # IGDX (Inverted Generational Distance - Set)
        if true_set is not None and current_set is not None and len(current_set) > 0:
            # Ensure current_set is 2D array even if only one solution
            if current_set.ndim == 1:
                current_set_2d = current_set.reshape(1, -1)
            else:
                current_set_2d = current_set

            # Ensure true_set is 2D array
            if true_set.ndim == 1:
                true_set_2d = true_set.reshape(1, -1)
            else:
                true_set_2d = true_set

            # Check dimensions match before calculating distance
            if current_set_2d.shape[1] == true_set_2d.shape[1]:
                igdx = PerformanceIndicators.igdx(current_set_2d, true_set_2d)
                self.tracking['metrics']['igdx'].append(igdx)
            else:
                self.logger.warning(
                    f"IGDX calculation skipped: Dimension mismatch between current set ({current_set_2d.shape}) and true set ({true_set_2d.shape})")
                self.tracking['metrics']['igdx'].append(float('nan'))
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # RPSP (r-Pareto Set Proximity)
        if true_front is not None and len(current_front) > 0:
            rpsp = PerformanceIndicators.rpsp(current_front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # HV (Hypervolume)
        if len(current_front) > 0:
            # Define reference point (crucial for HV)
            if true_front is not None:
                # Use slightly larger than the max values of the true front
                ref_point = np.max(true_front, axis=0) * 1.1
                # Handle cases where true front max is zero or negative
                ref_point[ref_point <= 0] = 0.1  # Assign a small positive value
            else:
                # Use slightly larger than the max values of the current front
                ref_point = np.max(current_front, axis=0) * 1.1
                ref_point[ref_point <= 0] = 0.1  # Assign a small positive value

            try:
                # Ensure front is 2D
                if current_front.ndim == 1:
                    current_front_2d = current_front.reshape(1, -1)
                else:
                    current_front_2d = current_front

                # Check dimensions match
                if current_front_2d.shape[1] == len(ref_point):
                    hv = PerformanceIndicators.hypervolume(current_front_2d, ref_point)
                    self.tracking['metrics']['hv'].append(hv)
                else:
                    self.logger.warning(
                        f"HV calculation skipped: Dimension mismatch between current front ({current_front_2d.shape}) and ref point ({len(ref_point)})")
                    self.tracking['metrics']['hv'].append(float('nan'))

            except Exception as e:
                self.logger.error(f"HV calculation error: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


# ====================== 性能评估指标 ======================

class PerformanceIndicators:
    """性能评估指标类，包含各种常用指标的计算方法"""

    @staticmethod
    def normalize_pareto_front(front, reference_front=None):
        """
        将帕累托前沿归一化到[0,1]范围

        参数:
        front: 需要归一化的帕累托前沿
        reference_front: 参考前沿(通常是真实前沿)，如果提供则用它的范围进行归一化

        返回:
        归一化后的帕累托前沿
        """
        if len(front) == 0:
            return front

        if reference_front is not None and len(reference_front) > 0:
            # 使用参考前沿的范围进行归一化
            min_vals = np.min(reference_front, axis=0)
            max_vals = np.max(reference_front, axis=0)
        else:
            # 使用前沿自身的范围
            min_vals = np.min(front, axis=0)
            max_vals = np.max(front, axis=0)

        # 确保范围非零
        range_vals = max_vals - min_vals
        # 避免除以零，如果某个维度范围为0，则设置为1
        range_vals[range_vals < 1e-10] = 1.0

        # 执行归一化
        normalized_front = (front - min_vals) / range_vals

        return normalized_front

    @staticmethod
    def spacing(front):
        """计算Pareto前沿的均匀性指标SP"""
        if len(front) < 2:
            return float('nan')  # 改为返回NaN而不是0

        try:
            # 对前沿进行归一化
            normalized_front = PerformanceIndicators.normalize_pareto_front(front)

            # 计算每对解之间的欧几里得距离
            distances = []
            for i in range(len(normalized_front)):
                min_dist = float('inf')
                for j in range(len(normalized_front)):
                    if i != j:
                        # 确保正确的数组转换
                        dist = np.sqrt(np.sum((normalized_front[i] - normalized_front[j]) ** 2))
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)

            # 计算平均距离
            d_mean = np.mean(distances)

            # 计算标准差
            sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

            return sp
        except Exception as e:
            print(f"SP计算错误: {e}")
            return float('nan')

    @staticmethod
    @staticmethod
    def igd(approximation_front, true_front):
        """
        计算反向代际距离(IGD)
        从真实Pareto前沿到近似前沿的平均距离
        值越小表示质量越高
        """
        if len(approximation_front) == 0 or len(true_front) == 0:
            return float('inf')

        # 对两个前沿使用相同的归一化方法
        norm_approx = PerformanceIndicators.normalize_pareto_front(approximation_front, true_front)
        norm_true = PerformanceIndicators.normalize_pareto_front(true_front, true_front)

        # 计算每个点到前沿的最小距离
        distances = cdist(norm_true, norm_approx, 'euclidean')
        min_distances = np.min(distances, axis=1)

        # 返回平均距离
        return np.mean(min_distances)

    @staticmethod
    def hypervolume(front, reference_point):
        """改进的超体积计算，支持2D和3D，包含归一化处理"""
        if len(front) == 0:
            return 0

        # 归一化前沿
        norm_front = PerformanceIndicators.normalize_pareto_front(front)
        # 归一化后使用(1.1, 1.1, ...)作为参考点
        norm_reference_point = np.ones(front.shape[1]) * 1.1

        try:
            from pymoo.indicators.hv import HV
            return HV(ref_point=norm_reference_point).do(norm_front)
        except ImportError:
            # 如果没有pymoo，根据维度使用不同的计算方法
            n_dim = norm_front.shape[1]

            if n_dim == 2:
                # 2D问题特殊处理 - 按x坐标排序计算面积
                sorted_front = norm_front[norm_front[:, 0].argsort()]
                hv = 0

                # 计算超体积
                for i in range(len(sorted_front)):
                    if i == 0:
                        width = sorted_front[i, 0]
                    else:
                        width = sorted_front[i, 0] - sorted_front[i - 1, 0]

                    height = max(0, norm_reference_point[1] - sorted_front[i, 1])
                    hv += width * height

                # 添加最后一块区域
                if len(sorted_front) > 0:
                    last_width = max(0, norm_reference_point[0] - sorted_front[-1, 0])
                    last_height = norm_reference_point[1]
                    hv += last_width * last_height

                return max(0, hv)
            else:
                # 3D及更高维度使用蒙特卡洛方法
                n_samples = 50000
                dominated_count = 0

                for _ in range(n_samples):
                    # 生成参考点和前沿之间的随机点
                    point = np.random.uniform(0, norm_reference_point)

                    # 检查是否被任何前沿点支配
                    dominated = False
                    for sol in norm_front:
                        if np.all(sol <= point):
                            dominated = True
                            break

                    if dominated:
                        dominated_count += 1

                # 计算超体积
                volume = np.prod(norm_reference_point)
                return (dominated_count / n_samples) * volume

    @staticmethod
    def igdf(approximation_front, true_front):
        """
        计算前沿空间中的IGD (IGDF)
        从真实Pareto前沿到近似前沿的平均距离
        """
        if len(approximation_front) == 0 or len(true_front) == 0:
            return float('nan')

        try:
            # 对两个前沿使用相同的归一化方法
            norm_approx = PerformanceIndicators.normalize_pareto_front(approximation_front, true_front)
            norm_true = PerformanceIndicators.normalize_pareto_front(true_front, true_front)

            # 计算每个真实前沿点到近似前沿的最小距离
            distances = cdist(norm_true, norm_approx, 'euclidean')
            min_distances = np.min(distances, axis=1)

            # 返回平均距离
            return np.mean(min_distances)
        except Exception as e:
            print(f"IGDF计算错误: {e}")
            return float('nan')

    @staticmethod
    def igdx(approximation_set, true_set):
        """
        计算决策变量空间中的IGD (IGDX)
        从真实Pareto解集到近似解集的平均距离
        """
        if approximation_set is None or true_set is None:
            return float('nan')
        if len(approximation_set) == 0 or len(true_set) == 0:
            return float('nan')

        try:
            # 对两个解集使用相同的归一化方法
            norm_approx = PerformanceIndicators.normalize_pareto_front(approximation_set, true_set)
            norm_true = PerformanceIndicators.normalize_pareto_front(true_set, true_set)

            # 计算每个真实解集点到近似解集的最小距离
            distances = cdist(norm_true, norm_approx, 'euclidean')
            min_distances = np.min(distances, axis=1)

            # 返回平均距离
            return np.mean(min_distances)
        except Exception as e:
            print(f"IGDX计算错误: {e}")
            return float('nan')

    @staticmethod
    def rpsp(front, reference_front, r=0.1):
        """
        计算r-PSP (Radial-based Pareto Set Proximity)
        r: 径向扩展参数(默认0.1)
        """
        if len(front) < 2 or len(reference_front) < 2:
            return float('nan')

        try:
            # 归一化两个前沿
            norm_front = PerformanceIndicators.normalize_pareto_front(front, reference_front)
            norm_ref = PerformanceIndicators.normalize_pareto_front(reference_front, reference_front)

            # 计算径向距离
            n_obj = front.shape[1]
            rpsp_sum = 0
            count = 0

            for ref_point in norm_ref:
                # 找最近的近似前沿点
                min_dist = float('inf')
                for approx_point in norm_front:
                    # 径向距离计算
                    diff_vector = approx_point - ref_point
                    angle_penalty = np.linalg.norm(diff_vector) * r
                    dist = np.linalg.norm(diff_vector) + angle_penalty
                    min_dist = min(min_dist, dist)

                rpsp_sum += min_dist
                count += 1

            return rpsp_sum / count if count > 0 else float('nan')
        except Exception as e:
            print(f"RPSP计算错误: {e}")
            return float('nan')


def get_optimal_casmopso_params(problem_name, max_iterations=100):
    """为不同的问题返回最优CASMOPSO参数，并确保max_iterations一致"""

    # 基础参数配置
    base_params = {
        "pop_size": 200,
        "max_iterations": max_iterations,  # 确保与主函数中的一致
        "w_init": 0.9,
        "w_end": 0.4,
        "c1_init": 2.5,
        "c1_end": 0.5,
        "c2_init": 0.5,
        "c2_end": 2.5,
        "use_archive": True,
        "archive_size": 100,
        "mutation_rate": 0.1,
        "adaptive_grid_size": 15,
        "k_vmax": 0.5
    }

    # 问题特定参数配置
    problem_specific_params = {
        "ZDT1": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.9, "w_end": 0.4,
            "c1_init": 2.0, "c1_end": 0.8,
            "c2_init": 0.8, "c2_end": 2.0,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 20,
            "k_vmax": 0.5
        },
        "ZDT2": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.9, "w_end": 0.4,
            "c1_init": 2.0, "c1_end": 0.8,
            "c2_init": 0.8, "c2_end": 2.0,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 20,
            "k_vmax": 0.5
        },
        "ZDT3": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.9, "w_end": 0.4,
            "c1_init": 2.0, "c1_end": 0.8,
            "c2_init": 0.8, "c2_end": 2.0,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 20,
            "k_vmax": 0.5
        },
        "ZDT4": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.9, "w_end": 0.4,
            "c1_init": 2.0, "c1_end": 0.8,
            "c2_init": 0.8, "c2_end": 2.0,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 20,
            "k_vmax": 0.5
        },
        "ZDT6": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.9, "w_end": 0.4,
            "c1_init": 2.0, "c1_end": 0.8,
            "c2_init": 0.8, "c2_end": 2.0,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 20,
            "k_vmax": 0.5
        },
        "DTLZ1": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ2": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ3": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ4": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ5": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ6": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        },
        "DTLZ7": {
            "pop_size": 100,
            "max_iterations": max_iterations,  # 确保与主函数中的一致
            "w_init": 0.8, "w_end": 0.3,
            "c1_init": 2.2, "c1_end": 0.5,
            "c2_init": 0.6, "c2_end": 2.5,
            "archive_size": 100,
            "mutation_rate": 0.15,
            "adaptive_grid_size": 20,
            "k_vmax": 0.4
        }
    }

    # 返回问题特定参数或基础参数
    return problem_specific_params.get(problem_name, base_params)


class DataExporter:
    """数据导出工具类，用于收集和导出多目标优化结果"""

    def __init__(self, export_dir="exported_data"):
        """
        初始化数据导出器

        export_dir: 导出数据的目录
        """
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        # 主目录创建
        self.performance_dir = os.path.join(export_dir, "performance")
        self.pareto_front_dir = os.path.join(export_dir, "pareto_fronts")
        self.pareto_set_dir = os.path.join(export_dir, "pareto_sets")

        for directory in [self.performance_dir, self.pareto_front_dir, self.pareto_set_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def export_generational_performance_all_runs(self, problem_name, algorithm_name, all_tracking_data,
                                                 algorithm_obj=None):
        """
        (修改版) 导出所有运行的实际记录的代际性能数据

        problem_name: 问题名称
        algorithm_name: 算法名称
        all_tracking_data: 所有运行的跟踪数据列表
        algorithm_obj: 算法对象，用于获取每步运算次数
        """
        # 创建问题特定的子文件夹
        problem_dir = os.path.join(self.performance_dir, problem_name)
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)

        # 获取算法每步的运算次数倍率 (通常是种群大小)
        comp_per_iter = 1
        if hasattr(algorithm_obj, 'pop_size'):
            comp_per_iter = algorithm_obj.pop_size
        # 对于遗传算法类，通常也是pop_size或类似
        elif hasattr(algorithm_obj, 'pop_size') and algorithm_name in ["NSGAII", "GDE3"]:
            comp_per_iter = algorithm_obj.pop_size
        elif hasattr(algorithm_obj, 'pop_size') and algorithm_name == "MOEAD":
            # MOEA/D的计算复杂度略有不同，但通常也与pop_size相关
            comp_per_iter = algorithm_obj.pop_size

        # 为每次运行创建单独的性能记录文件
        for run_id, tracking_data in enumerate(all_tracking_data):
            file_path = os.path.join(problem_dir,
                                     f"{algorithm_name}_run_{run_id + 1}_generational.txt")  # 文件名加后缀

            with open(file_path, 'w') as f:
                f.write(f"# 问题: {problem_name}, 算法: {algorithm_name}, 运行: {run_id + 1} 的实际记录性能数据\n")
                f.write("# 注意: 本文件只包含实际跟踪记录的迭代数据点\n")
                f.write("# 格式: 迭代次数, 运算次数, IGDF, IGDX, RPSP, HV, SP\n\n")

                # 检查是否有迭代数据
                if 'iterations' not in tracking_data or not tracking_data['iterations']:
                    f.write("# 无迭代数据\n")
                    continue

                # 获取原始跟踪数据
                orig_iterations = tracking_data['iterations']
                metrics = tracking_data['metrics']

                # --- 修改核心逻辑 ---
                # 直接遍历原始记录的迭代次数
                for i, iteration in enumerate(orig_iterations):
                    # 计算运算次数
                    # 注意：这里的 'iteration' 是实际记录的迭代次数，
                    #      如果算法中跳过了某些迭代的记录，运算次数的计算可能需要更精确的逻辑
                    #      但这里我们仍基于记录的迭代次数来估算
                    computation_count = iteration * comp_per_iter

                    # 构造数据行
                    line = f"{iteration}, {computation_count}"

                    # 获取对应迭代的指标值
                    for metric in ['igdf', 'igdx', 'rpsp', 'hv', 'sp']:
                        if metric in metrics and i < len(metrics[metric]):
                            # 直接使用原始记录的值
                            value = metrics[metric][i]
                            # 处理 NaN 值，可以选择输出 'NaN' 或其他标记
                            if np.isnan(value):
                                line += ", NaN"
                            else:
                                line += f", {value}"
                        else:
                            # 如果某个指标在这次迭代没有记录（理论上不应发生，除非跟踪逻辑有问题）
                            line += ", NaN"

                    f.write(line + "\n")
                # --- 修改结束 ---

                # 添加解集信息 (保持不变)
                f.write("\n# 最终解集大小信息\n")
                if 'fronts' in tracking_data and tracking_data['fronts']:
                    last_front = tracking_data['fronts'][-1]
                    f.write(f"# 最终帕累托前沿大小: {len(last_front)} 个解\n")

    def export_pareto_front(self, problem_name, problem, algorithms_results):
        """
        导出帕累托前沿数据

        problem_name: 问题名称
        problem: 问题对象
        algorithms_results: 各算法结果
        """
        file_path = os.path.join(self.pareto_front_dir, f"{problem_name}_pareto_fronts.txt")

        with open(file_path, 'w') as f:
            f.write(f"# 问题: {problem_name} 的帕累托前沿数据\n\n")

            # 导出真实帕累托前沿
            if hasattr(problem, 'pareto_front') and problem.pareto_front is not None:
                f.write(f"# 真实帕累托前沿 (共 {len(problem.pareto_front)} 个点)\n")
                f.write("# 格式: f1, f2, f3\n")
                for point in problem.pareto_front:
                    f.write(f"{', '.join(map(str, point))}\n")
            else:
                f.write("# 真实帕累托前沿不可用\n")

            # 导出各算法的帕累托前沿
            for algo_name, result in algorithms_results.items():
                if "pareto_front" in result:
                    pareto_front = result["pareto_front"]
                    f.write(f"\n\n# {algo_name} 算法的帕累托前沿 (共 {len(pareto_front)} 个点)\n")
                    f.write("# 格式: f1, f2, f3\n")
                    for point in pareto_front:
                        f.write(f"{', '.join(map(str, point))}\n")

                # 导出所有运行的帕累托前沿
                if "pareto_fronts" in result:
                    for run_id, front in enumerate(result["pareto_fronts"]):
                        f.write(f"\n\n# {algo_name} 算法运行 {run_id + 1} 的帕累托前沿 (共 {len(front)} 个点)\n")
                        f.write("# 格式: f1, f2, f3\n")
                        for point in front:
                            f.write(f"{', '.join(map(str, point))}\n")

    def export_pareto_set(self, problem_name, problem, algorithms_results):
        """
        导出帕累托解集数据

        problem_name: 问题名称
        problem: 问题对象
        algorithms_results: 各算法结果
        """
        file_path = os.path.join(self.pareto_set_dir, f"{problem_name}_pareto_sets.txt")

        with open(file_path, 'w') as f:
            f.write(f"# 问题: {problem_name} 的帕累托解集数据\n\n")

            # 导出真实帕累托解集
            if hasattr(problem, 'pareto_set') and problem.pareto_set is not None:
                f.write(f"# 真实帕累托解集 (共 {len(problem.pareto_set)} 个解)\n")
                f.write(f"# 格式: x1, x2, ..., x{problem.n_var}\n")
                for solution in problem.pareto_set:
                    f.write(f"{', '.join(map(str, solution))}\n")
            else:
                f.write("# 真实帕累托解集不可用\n")

            # 导出各算法的帕累托解集
            for algo_name, result in algorithms_results.items():
                if "pareto_set" in result:
                    pareto_set = result["pareto_set"]
                    f.write(f"\n\n# {algo_name} 算法的帕累托解集 (共 {len(pareto_set)} 个解)\n")
                    f.write(f"# 格式: x1, x2, ..., x{problem.n_var}\n")
                    for solution in pareto_set:
                        f.write(f"{', '.join(map(str, solution))}\n")

                # 导出所有运行的帕累托解集
                if "pareto_sets" in result:
                    for run_id, pareto_set in enumerate(result["pareto_sets"]):
                        f.write(f"\n\n# {algo_name} 算法运行 {run_id + 1} 的帕累托解集 (共 {len(pareto_set)} 个解)\n")
                        f.write(f"# 格式: x1, x2, ..., x{problem.n_var}\n")
                        for solution in pareto_set:
                            f.write(f"{', '.join(map(str, solution))}\n")

    def export_summary_metrics(self, problem_name, algorithms_results):
        """
        导出汇总性能指标 - 按问题组织子文件夹

        problem_name: 问题名称
        algorithms_results: 各算法结果
        """
        # 创建问题特定的子文件夹
        problem_dir = os.path.join(self.performance_dir, problem_name)
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)

        file_path = os.path.join(problem_dir, "summary_metrics.txt")

        with open(file_path, 'w') as f:
            f.write(f"# 问题: {problem_name} 的汇总性能指标\n\n")
            f.write("# 格式: 算法, 指标, 平均值, 标准差, 最小值, 最大值\n\n")

            for algo_name, result in algorithms_results.items():
                if "metrics" not in result:
                    continue

                metrics = result["metrics"]
                f.write(f"# {algo_name} 算法的性能指标\n")

                for metric in ['igdf', 'igdx', 'rpsp', 'hv', 'sp']:
                    if metric in metrics and metrics[metric]:
                        values = [v for v in metrics[metric] if not np.isnan(v)]
                        if values:
                            avg = np.mean(values)
                            std = np.std(values)
                            min_val = np.min(values)
                            max_val = np.max(values)
                            f.write(f"{algo_name}, {metric}, {avg}, {std}, {min_val}, {max_val}\n")

                # 添加运行时间信息
                if "runtimes" in result:
                    runtimes = result["runtimes"]
                    avg_time = np.mean(runtimes)
                    std_time = np.std(runtimes)
                    min_time = np.min(runtimes)
                    max_time = np.max(runtimes)
                    f.write(f"{algo_name}, runtime, {avg_time}, {std_time}, {min_time}, {max_time}\n")

                f.write("\n")


# ====================== 可视化功能 ======================

class Visualizer:
    """可视化工具类，用于绘制Pareto前沿、解集和性能指标"""

    @staticmethod
    def plot_pareto_front_comparison(problem, algorithms_results, save_path=None, plot_true_front=True):
        """
        比较不同算法的Pareto前沿，根据目标数量绘制2D或3D图
        """
        n_obj = problem.n_obj
        fig = plt.figure(figsize=(12, 10))

        if n_obj == 2:
            ax = fig.add_subplot(111)  # 创建 2D 子图
            plot_3d = False
        elif n_obj == 3:
            ax = fig.add_subplot(111, projection='3d')  # 创建 3D 子图
            plot_3d = True
        else:
            print(f"警告: 不支持绘制 {n_obj} 维目标空间，将只绘制前两个维度。")
            ax = fig.add_subplot(111)
            plot_3d = False
            n_obj = 2  # 强制绘制 2D

        # 绘制算法结果
        markers = ['o', 's', '^', 'D', 'p', '*']
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))
        all_points_list = []  # 用于存储所有点以确定范围

        # 获取真实前沿作为归一化参考
        reference_front = problem.pareto_front if hasattr(problem,
                                                          'pareto_front') and problem.pareto_front is not None else None

        for (algo_name, result), marker, color in zip(algorithms_results.items(), markers, colors):
            if "pareto_front" in result:
                pareto_front = result["pareto_front"]
                if pareto_front.shape[1] >= n_obj:  # 确保数据维度足够
                    # 归一化帕累托前沿
                    norm_front = PerformanceIndicators.normalize_pareto_front(pareto_front, reference_front)

                    if plot_3d:
                        ax.scatter(norm_front[:, 0], norm_front[:, 1], norm_front[:, 2],
                                   marker=marker, s=30, color=color, label=f"{algo_name}")
                        all_points_list.append(norm_front[:, :3])
                    else:
                        ax.scatter(norm_front[:, 0], norm_front[:, 1],
                                   marker=marker, s=30, color=color, label=f"{algo_name}")
                        all_points_list.append(norm_front[:, :2])

        # 绘制真实Pareto前沿
        if plot_true_front and reference_front is not None:
            true_pf = reference_front
            if true_pf.shape[1] >= n_obj:
                # 归一化真实前沿
                norm_true = PerformanceIndicators.normalize_pareto_front(true_pf, true_pf)

                if plot_3d:
                    ax.scatter(norm_true[:, 0], norm_true[:, 1], norm_true[:, 2],
                               marker='+', s=10, color='red', alpha=0.5, label='True PF')
                    all_points_list.append(norm_true[:, :3])
                else:
                    ax.scatter(norm_true[:, 0], norm_true[:, 1],
                               marker='+', s=10, color='red', alpha=0.5, label='True PF')
                    all_points_list.append(norm_true[:, :2])

        # 设置图例和标签
        ax.set_xlabel('$f_1$', labelpad=10)
        ax.set_ylabel('$f_2$', labelpad=10)
        if plot_3d:
            ax.set_zlabel('$f_3$', labelpad=10)
        ax.set_title(f'Pareto front for {problem.name} (Normalized)')

        # 设置轴限制 - 统一使用[0,1]范围加上少量边距
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        if plot_3d:
            ax.set_zlim(-0.05, 1.05)
            ax.view_init(elev=30, azim=45)  # 3D视角

        # 调整图例位置和大小
        ax.legend(loc='best')  # 尝试自动选择最佳位置

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.3)

        # 保存或显示图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_pareto_set_comparison(problem, algorithms_results, save_path=None, plot_true_set=True):
        """
        比较不同算法的Pareto解集

        problem: 测试问题实例
        algorithms_results: 字典，键为算法名称，值为算法结果
        save_path: 保存图像的路径
        plot_true_set: 是否绘制真实Pareto解集
        """
        # 检查问题是否为三维变量
        if problem.n_var < 3:
            print("警告: 问题变量维度小于3，无法绘制3D解集")
            return None, None

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制算法结果
        markers = ['o', 's', '^', 'D', 'p', '*']
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))

        # 记录所有点的坐标范围
        all_points = []

        for (algo_name, result), marker, color in zip(algorithms_results.items(), markers, colors):
            if "pareto_set" in result:
                pareto_set = result["pareto_set"]
                # 只绘制前三个维度
                ax.scatter(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2],
                           marker=marker, s=30, color=color, label=f"{algo_name} (PS)")
                all_points.append(pareto_set[:, :3])

        # 绘制真实Pareto解集
        if plot_true_set and problem.pareto_set is not None:
            ax.scatter(problem.pareto_set[:, 0], problem.pareto_set[:, 1], problem.pareto_set[:, 2],
                       marker='+', s=10, color='red', alpha=0.5, label='True PS')
            all_points.append(problem.pareto_set[:, :3])

        # 设置图例和标签
        ax.set_xlabel('$x_1$', labelpad=10)
        ax.set_ylabel('$x_2$', labelpad=10)
        ax.set_zlabel('$x_3$', labelpad=10)
        ax.set_title(f'Pareto set for {problem.name}')

        # 确保完整显示坐标轴
        if all_points:
            all_points = np.vstack(all_points)
            min_vals = np.min(all_points, axis=0)
            max_vals = np.max(all_points, axis=0)

            # 添加边距
            padding = (max_vals - min_vals) * 0.1

            # 设置轴限制
            ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
            ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
            ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

        # 调整图例位置和大小
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # 设置视角
        ax.view_init(elev=30, azim=45)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.3)

        # 保存或显示图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_algorithm_performance_boxplots(algorithms_results, problem_name,
                                            metrics=["igdf", "igdx", "rpsp", "hv", "sp"],
                                            save_path=None, sample_interval=10):
        """
        改进的箱线图函数，支持新指标，改为纵向排布

        sample_interval: 采样间隔，只使用每sample_interval代的数据点计算
        """
        n_metrics = len(metrics)
        # 改为 n_metrics 行, 1 列
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 6 * n_metrics))

        if n_metrics == 1:
            # 如果只有一个指标，确保 axes 是一个列表或可迭代对象
            axes = [axes]

        metric_labels = {
            "igdf": "IGDF",
            "igdx": "IGDX",
            "rpsp": "RPSP",
            "hv": "HV",
            "sp": "SP",
            "igd": "IGD"
        }

        # 调试输出
        print(f"Creating performance boxplots for {problem_name}...")
        for metric in metrics:
            print(f"  Checking {metric} data:")
            for algo_name, result in algorithms_results.items():
                # 检查指标是否在tracking结构中
                if "tracking" in result and "metrics" in result["tracking"] and metric in result["tracking"]["metrics"]:
                    metric_values = result["tracking"]["metrics"][metric]
                    # 采样数据，每sample_interval个点取一个
                    sampled_values = metric_values[::sample_interval]
                    valid_values = [v for v in sampled_values if not np.isnan(v)]
                    print(
                        f"    {algo_name}: {len(valid_values)} valid data points (sampled every {sample_interval} generations)")
                else:
                    print(f"    {algo_name}: No data for {metric}")

        for i, metric in enumerate(metrics):
            # 收集所有算法的指标值
            data = []
            labels = []

            for algo_name, result in algorithms_results.items():
                # 从tracking结构中获取数据（实际数据存储位置）
                if "tracking" in result and "metrics" in result["tracking"] and metric in result["tracking"]["metrics"]:
                    metric_values = result["tracking"]["metrics"][metric]
                    # 采样数据，每sample_interval个点取一个
                    sampled_values = metric_values[::sample_interval]
                    values = [v for v in sampled_values if not np.isnan(v)]

                    if values:
                        data.append(values)
                        labels.append(algo_name)

            if data:
                # 检查数据是否合理，特别是SP值往往很小
                if metric == "sp":
                    print(f"  SP data ranges: {[min(d) for d in data]} to {[max(d) for d in data]}")
                    # 对于异常小的值，可以进行数据转换以便更好地可视化
                    if any(min(d) < 0.001 for d in data):
                        print("  Warning: Very small SP values detected, consider log transformation")

                try:
                    # 创建小提琴图和箱线图
                    if len(data) > 0:
                        # 获取当前子图
                        ax = axes[i]

                        violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)

                        # 设置颜色
                        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
                        for j, pc in enumerate(violin_parts['bodies']):
                            pc.set_facecolor(colors[j])
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)

                        # 添加boxplot
                        bp = ax.boxplot(data, positions=range(1, len(data) + 1),
                                        widths=0.15, patch_artist=True,
                                        showfliers=True, showmeans=True, meanline=True)

                        # 自定义boxplot颜色
                        for j, box in enumerate(bp['boxes']):
                            box.set(facecolor='white', alpha=0.5)

                        # 设置标签和标题
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.set_ylabel(metric_labels.get(metric, metric.upper()))
                        ax.set_title(
                            f'{metric_labels.get(metric, metric.upper())} Performance (sampled every {sample_interval} generations)')
                        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

                        # 设置更好的y轴范围
                        if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:  # 较小值更好
                            if min([min(d) for d in data]) < 0.1:
                                ax.set_ylim(bottom=0)  # 从0开始

                        # 优化布局
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    else:
                        # 确保使用正确的 axes 对象
                        ax = axes[i]
                        ax.text(0.5, 0.5, f"No valid {metric.upper()} data",
                                ha='center', va='center', fontsize=12)
                        ax.set_axis_off()
                except Exception as e:
                    # 确保使用正确的 axes 对象
                    ax = axes[i]
                    print(f"Error plotting {metric}: {e}")
                    ax.text(0.5, 0.5, f"Error plotting {metric} data",
                            ha='center', va='center', fontsize=12)
                    ax.set_axis_off()
            else:
                # 确保使用正确的 axes 对象
                ax = axes[i]
                ax.text(0.5, 0.5, f"No valid {metric.upper()} data",
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()

        # 设置总标题
        plt.suptitle(f'Algorithm Performance on {problem_name} (Sampled every {sample_interval} generations)',
                     fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 调整布局以适应总标题和X轴标签

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_convergence(algorithms_results, metric_name="igdf", problem_name="", save_path=None, sample_interval=10):
        """
        绘制收敛曲线，支持多种性能指标，每隔sample_interval代数据绘制一个点

        algorithms_results: 字典，键为算法名称，值为算法结果
        metric_name: 指标名称（igdf, igdx, rpsp, hv, sp）
        problem_name: 问题名称
        save_path: 保存图像的路径
        sample_interval: 采样间隔，默认为10（每10代采样一个点）
        """
        plt.figure(figsize=(10, 6))

        # 调试信息
        print(f"绘制{problem_name}问题的{metric_name.upper()}收敛曲线...")
        data_plotted = False

        for algo_name, result in algorithms_results.items():
            if "tracking" in result and "metrics" in result["tracking"]:
                iterations = result["tracking"]["iterations"]
                metric_values = result["tracking"]["metrics"].get(metric_name, [])

                # 检查数据有效性
                valid_values = [v for v in metric_values if not np.isnan(v)]
                print(f"  {algo_name}: 共{len(metric_values)}个值，其中{len(valid_values)}个有效值")

                if iterations and valid_values:
                    # 处理NaN值 - 用前一个有效值填充
                    clean_values = []
                    last_valid = None
                    for v in metric_values:
                        if not np.isnan(v):
                            clean_values.append(v)
                            last_valid = v
                        elif last_valid is not None:
                            clean_values.append(last_valid)
                        else:
                            clean_values.append(0)  # 初始无效值填充为0

                    # 采样数据点，每sample_interval代取一个点
                    sampled_iterations = iterations[::sample_interval]
                    sampled_values = clean_values[::sample_interval]

                    # 确保包含最后一个点（如果有）
                    if iterations and iterations[-1] not in sampled_iterations:
                        sampled_iterations = np.append(sampled_iterations, iterations[-1])
                        sampled_values = np.append(sampled_values, clean_values[-1])

                    plt.plot(sampled_iterations, sampled_values, '-o', label=algo_name, markersize=4)
                    data_plotted = True
                else:
                    print(f"  {algo_name}: 无足够有效数据，跳过绘图")

        # 设置标题和标签
        metric_labels = {
            "igdf": "IGDF (Inverted Generational Distance in F-space)",
            "igdx": "IGDX (Inverted Generational Distance in X-space)",
            "rpsp": "RPSP (r-Pareto Set Proximity)",
            "hv": "HV (Hypervolume)",
            "sp": "SP (Spacing)"
        }

        metric_label = {
            "igdf": "IGDF",
            "igdx": "IGDX",
            "rpsp": "RPSP",
            "hv": "HV",
            "sp": "SP"
        }

        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(metric_labels.get(metric_name, metric_name.upper()), fontsize=12)
        plt.title(f'{metric_label.get(metric_name, metric_name.upper())} For {problem_name}', fontsize=14)

        # 设置网格和图例
        plt.grid(True, linestyle='--', alpha=0.7)

        # Y轴范围调整 - 对于不同指标采用不同策略
        if data_plotted:
            if metric_name in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                # 这些指标越小越好，从0开始显示
                ymin, ymax = plt.ylim()
                plt.ylim(0, ymax)

            plt.legend(loc='best', fontsize=10)

            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  图像已保存到: {save_path}")
        else:
            # 没有有效数据时显示提示信息
            plt.text(0.5, 0.5, f"NO {metric_name.upper()} Value",
                     ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  空图像已保存到: {save_path}")

        return plt.gcf()

    @staticmethod
    def plot_radar_chart(algorithms_results, problem_name, metrics=["igdf", "igdx", "rpsp", "hv", "sp"],
                         save_path=None):
        """
        绘制雷达图(蜘蛛网图)展示算法在多个指标上的性能对比

        algorithms_results: 字典，键为算法名称，值为算法结果
        problem_name: 问题名称
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        # 计算每个算法在每个指标上的标准化得分
        algo_names = list(algorithms_results.keys())
        n_metrics = len(metrics)

        # 提取各指标数据
        data = {}
        for metric in metrics:
            metric_values = {}
            for algo_name in algo_names:
                if f"avg_{metric}" in algorithms_results[algo_name].get(problem_name, {}).get("metrics", {}):
                    value = algorithms_results[algo_name][problem_name]["metrics"][f"avg_{metric}"]
                    if not np.isnan(value):
                        metric_values[algo_name] = value

            if metric_values:
                data[metric] = metric_values

        # 标准化分数 - 针对不同指标使用不同规则
        scores = {algo: [0] * n_metrics for algo in algo_names}

        for i, metric in enumerate(metrics):
            if metric not in data or not data[metric]:
                continue

            values = data[metric]
            # 对于IGD*和SP，较小值更好；对于HV，较大值更好
            if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                best = min(values.values())
                worst = max(values.values())
                # 标准化公式: (worst - value) / (worst - best)
                diff = worst - best
                if diff > 0:
                    for algo in values:
                        scores[algo][i] = (worst - values[algo]) / diff
            else:  # hv
                best = max(values.values())
                worst = min(values.values())
                # 标准化公式: (value - worst) / (best - worst)
                diff = best - worst
                if diff > 0:
                    for algo in values:
                        scores[algo][i] = (values[algo] - worst) / diff

        # 绘制雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 设置标签位置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.upper() for metric in metrics])

        # 绘制每个算法的得分
        colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))

        for i, algo in enumerate(algo_names):
            if algo not in scores:
                continue

            values = scores[algo]
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=algo)
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # 美化图表
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Performance Comparison on {problem_name}', fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_metrics_summary(results, problems, metrics=["igdf", "igdx", "rpsp", "hv", "sp"], save_path=None):
        """
        绘制各算法在所有问题上的指标汇总热图

        results: 结果字典
        problems: 问题列表
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        algo_names = list(results.keys())
        problem_names = [p.name for p in problems]
        n_algos = len(algo_names)
        n_problems = len(problem_names)

        # 绘制每个指标的热图
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(max(10, n_problems * 0.8), max(8, n_algos * 0.6)))

            # 准备数据
            data = np.zeros((n_algos, n_problems))
            data.fill(np.nan)  # 初始化为NaN

            # 提取数据
            for i, algo in enumerate(algo_names):
                for j, prob in enumerate(problem_names):
                    if prob in results[algo] and f"avg_{metric}" in results[algo][prob]["metrics"]:
                        data[i, j] = results[algo][prob]["metrics"][f"avg_{metric}"]

            # 处理NaN值
            mask = np.isnan(data)

            # 根据指标类型调整颜色映射
            if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                cmap = "YlOrRd_r"  # 逆序：较低的值(更好)显示为较浅的颜色
            else:  # hv
                cmap = "YlOrRd"  # 正序：较高的值(更好)显示为较深的颜色

            # 绘制热图
            im = ax.imshow(data, cmap=cmap, aspect='auto')

            # 添加数值标签
            for i in range(n_algos):
                for j in range(n_problems):
                    if not mask[i, j]:
                        # 根据值的大小调整颜色
                        val = data[i, j]
                        if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                            is_best_in_col = val == np.nanmin(data[:, j])
                        else:  # hv
                            is_best_in_col = val == np.nanmax(data[:, j])

                        # 最优值用粗体标记
                        if is_best_in_col:
                            ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                                    color="black", fontweight='bold')
                        else:
                            ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                                    color="black")

            # 设置坐标轴
            ax.set_xticks(np.arange(n_problems))
            ax.set_yticks(np.arange(n_algos))
            ax.set_xticklabels(problem_names)
            ax.set_yticklabels(algo_names)

            # 旋转x轴标签以防重叠
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # 添加标题和颜色条
            metric_titles = {
                "igdf": "IGDF (Convergence in objective space)",
                "igdx": "IGDX (Convergence in decision space)",
                "rpsp": "RPSP (r-Radial set proximity)",
                "hv": "HV (Hypervolume)",
                "sp": "SP (Distribution uniformity)"
            }

            plt.colorbar(im, ax=ax, label=f"{metric.upper()} Value")
            ax.set_title(f"{metric_titles.get(metric, metric.upper())} Comparison")

            fig.tight_layout()

            # 保存图像
            if save_path:
                metric_save_path = save_path.replace(".png", f"_{metric}.png")
                plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_performance_comparison(algorithms_results, problems, metrics=["igdf", "igdx", "rpsp", "hv", "sp"],
                                    save_path=None):
        """
        比较不同算法在多个问题上的性能

        algorithms_results: 字典，键为算法名称，值为算法结果
        problems: 问题列表或名称列表
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        n_metrics = len(metrics)
        n_problems = len(problems)

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))

        for i, metric in enumerate(metrics):
            # 提取每个算法在每个问题上的性能数据
            problem_names = [p.name if hasattr(p, 'name') else str(p) for p in problems]
            data = {algo_name: [] for algo_name in algorithms_results.keys()}

            for j, problem in enumerate(problem_names):
                for algo_name in algorithms_results.keys():
                    if problem in algorithms_results[algo_name]:
                        result = algorithms_results[algo_name][problem]
                        # 获取最后一个度量值作为最终性能
                        if "tracking" in result and "metrics" in result["tracking"]:
                            metric_values = result["tracking"]["metrics"].get(metric, [])
                            if metric_values:
                                data[algo_name].append(metric_values[-1])
                            else:
                                data[algo_name].append(float('nan'))
                        else:
                            data[algo_name].append(float('nan'))
                    else:
                        data[algo_name].append(float('nan'))

            # 绘制条形图
            bar_width = 0.8 / len(algorithms_results)
            for k, (algo_name, values) in enumerate(data.items()):
                x = np.arange(n_problems) + k * bar_width
                axes[i].bar(x, values, width=bar_width, label=algo_name, color=colors[k])

            # 设置标签和标题
            axes[i].set_xlabel('问题')
            axes[i].set_ylabel(metric.upper())

            metric_titles = {
                "igdf": "IGDF (目标空间收敛性)",
                "igdx": "IGDX (决策空间收敛性)",
                "rpsp": "RPSP (径向集逼近)",
                "hv": "HV (超体积)",
                "sp": "SP (均匀性)",
                "igd": "IGD (倒代距离)"
            }

            axes[i].set_title(f'{metric_titles.get(metric, metric.upper())} 性能比较')
            axes[i].set_xticks(np.arange(n_problems) + (len(algorithms_results) - 1) * bar_width / 2)
            axes[i].set_xticklabels(problem_names, rotation=45)
            axes[i].grid(True, linestyle='--', alpha=0.3, axis='y')

            if i == 0:
                axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(algorithms_results))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# ====================== 实验框架 ======================

class ExperimentFramework:
    """实验框架类，用于运行和比较不同算法"""

    def __init__(self, save_dir="results", export_data=True, normalize_results=True):
        """初始化实验框架"""
        self.save_dir = save_dir
        self.export_data = export_data
        self.normalize_results = normalize_results  # 新增：是否归一化结果

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(save_dir, 'experiment.log')
        )
        self.logger = logging.getLogger("ExperimentFramework")

        # 初始化数据导出器
        if export_data:
            self.data_exporter = DataExporter(os.path.join(save_dir, "exported_data"))

    def run_experiment(self, problems, algorithms, algorithm_params, problem_specific_params=None, n_runs=10,
                       verbose=True):
        """
        运行实验，并导出数据

        problems: 优化问题实例列表
        algorithms: 算法类列表
        algorithm_params: 通用算法参数字典 {算法名称: 参数字典}
        problem_specific_params: 问题特定参数字典 {算法名称: {问题名称: 参数字典}}
        n_runs: 每个问题和算法运行的次数
        verbose: 是否显示详细输出
        """
        # 创建结果字典
        results = {algo.__name__: {} for algo in algorithms}

        # 初始化问题特定参数字典（如果未提供）
        if problem_specific_params is None:
            problem_specific_params = {}

        for problem in problems:
            problem_name = problem.name

            if verbose:
                print(f"Running experiments on problem: {problem_name}")

            self.logger.info(f"Starting experiments on problem: {problem_name}")

            # 当前问题的算法结果
            problem_results = {}
            algorithm_instances = {}  # 存储算法实例以供导出使用
            all_tracking_data = {}  # 存储所有运行的跟踪数据

            for algorithm_class in algorithms:
                algo_name = algorithm_class.__name__

                if verbose:
                    print(f"  Algorithm: {algo_name}")

                self.logger.info(f"Running {algo_name} on {problem_name}")

                # 获取算法参数 - 优先使用问题特定参数
                if algo_name in problem_specific_params and problem_name in problem_specific_params[algo_name]:
                    params = problem_specific_params[algo_name][problem_name]
                else:
                    params = algorithm_params.get(algo_name, {})

                # 初始化结果
                results[algo_name][problem_name] = {
                    "pareto_fronts": [],
                    "pareto_sets": [],
                    "metrics": {
                        "igdf": [],
                        "igdx": [],
                        "rpsp": [],
                        "hv": [],
                        "sp": []
                    },
                    "runtimes": [],
                    "all_tracking": []  # 存储所有运行的跟踪数据
                }

                # 运行多次实验
                for run in range(n_runs):
                    if verbose:
                        print(f"    Run {run + 1}/{n_runs}")

                    # 创建算法实例
                    algorithm = algorithm_class(problem, **params)

                    # 保存第一次运行的算法实例，用于导出完整迭代数据
                    if run == 0:
                        algorithm_instances[algo_name] = algorithm

                    # 运行算法
                    start_time = time.time()
                    pareto_front = algorithm.optimize(tracking=True, verbose=False)
                    end_time = time.time()

                    # 收集Pareto解集
                    if hasattr(algorithm, '_get_pareto_set'):
                        pareto_set = algorithm._get_pareto_set()
                    else:
                        pareto_set = None

                    # 记录结果
                    results[algo_name][problem_name]["pareto_fronts"].append(pareto_front)
                    if pareto_set is not None:
                        results[algo_name][problem_name]["pareto_sets"].append(pareto_set)
                    results[algo_name][problem_name]["runtimes"].append(end_time - start_time)

                    # 收集跟踪数据
                    if hasattr(algorithm, 'tracking'):
                        # 深拷贝以避免引用问题
                        tracking_copy = copy.deepcopy(algorithm.tracking)
                        results[algo_name][problem_name]["all_tracking"].append(tracking_copy)

                        # 仍然保留第一次运行的跟踪数据用于兼容现有代码
                        if run == 0:
                            results[algo_name][problem_name]["tracking"] = tracking_copy

                    # 收集指标
                    if hasattr(algorithm, 'tracking') and "metrics" in algorithm.tracking:
                        for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                            values = algorithm.tracking["metrics"].get(metric_name, [])
                            if values:
                                final_value = values[-1]
                                results[algo_name][problem_name]["metrics"][metric_name].append(final_value)

                # 确定最佳解
                best_idx = self._determine_best_solution_index(results[algo_name][problem_name])

                # 保存最佳解
                if results[algo_name][problem_name]["pareto_fronts"]:
                    results[algo_name][problem_name]["pareto_front"] = \
                        results[algo_name][problem_name]["pareto_fronts"][best_idx]
                if "pareto_sets" in results[algo_name][problem_name] and results[algo_name][problem_name][
                    "pareto_sets"]:
                    results[algo_name][problem_name]["pareto_set"] = results[algo_name][problem_name]["pareto_sets"][
                        best_idx]

                # 计算平均指标
                self._calculate_average_metrics(results[algo_name][problem_name])

                # 输出结果汇总
                self._log_results_summary(algo_name, problem_name, results[algo_name][problem_name])

                # 将当前算法结果添加到问题结果集
                problem_results[algo_name] = results[algo_name][problem_name]

            # 导出数据
            if self.export_data:
                self._export_problem_data(problem_name, problem, problem_results, algorithm_instances)

            # 保存问题的比较图
            self._generate_visualizations(problem, problem_name, problem_results)

        # 保存结果汇总
        self._save_summary(results, problems)

        if verbose:
            print(f"\n实验完成! 结果已保存到 {self.save_dir} 目录")
            if self.export_data:
                print(f"详细数据已导出到 {os.path.join(self.save_dir, 'exported_data')} 目录")

        return results

    def _determine_best_solution_index(self, algorithm_result):
        """确定最佳解的索引"""
        best_idx = 0
        # 优先使用IGDF指标来确定最佳解，如果没有则尝试其他指标
        for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
            metric_values = algorithm_result["metrics"].get(metric_name, [])
            if metric_values:
                if metric_name in ["igdf", "igdx", "rpsp", "sp"]:  # 越小越好
                    best_idx = np.argmin(metric_values)
                else:  # hv，越大越好
                    best_idx = np.argmax(metric_values)
                break
        return best_idx

    def _calculate_average_metrics(self, algorithm_result):
        """计算平均指标"""
        for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
            values = algorithm_result["metrics"][metric_name]
            if values:
                # 过滤NaN值
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    algorithm_result["metrics"][f"avg_{metric_name}"] = np.mean(valid_values)
                    algorithm_result["metrics"][f"std_{metric_name}"] = np.std(valid_values)
                else:
                    algorithm_result["metrics"][f"avg_{metric_name}"] = float('nan')
                    algorithm_result["metrics"][f"std_{metric_name}"] = float('nan')
            else:
                algorithm_result["metrics"][f"avg_{metric_name}"] = float('nan')
                algorithm_result["metrics"][f"std_{metric_name}"] = float('nan')

    def _log_results_summary(self, algo_name, problem_name, algorithm_result):
        """记录结果汇总到日志"""
        self.logger.info(f"Completed {algo_name} on {problem_name}:")
        for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
            avg_key = f"avg_{metric_name}"
            if avg_key in algorithm_result["metrics"]:
                value = algorithm_result["metrics"][avg_key]
                if not np.isnan(value):
                    self.logger.info(f"  {avg_key}: {value:.6f}")
        self.logger.info(f"  Average runtime: {np.mean(algorithm_result['runtimes']):.2f} seconds")

    def _export_problem_data(self, problem_name, problem, problem_results, algorithm_instances):
        """导出问题数据 - 确保所有数据都是归一化的"""
        # 获取真实前沿作为归一化参考
        reference_front = None
        if hasattr(problem, 'pareto_front') and problem.pareto_front is not None:
            reference_front = problem.pareto_front

        for algo_name, result in problem_results.items():
            # 确保所有前沿数据已归一化
            if self.normalize_results:
                # 归一化每次运行的帕累托前沿（如果尚未归一化）
                normalized_fronts = []
                for front in result["pareto_fronts"]:
                    norm_front = PerformanceIndicators.normalize_pareto_front(front, reference_front)
                    normalized_fronts.append(norm_front)

                # 替换为归一化后的前沿
                problem_results[algo_name]["pareto_fronts"] = normalized_fronts

                # 如果有"最佳"帕累托前沿，也进行归一化
                if "pareto_front" in problem_results[algo_name]:
                    best_front = problem_results[algo_name]["pareto_front"]
                    norm_best = PerformanceIndicators.normalize_pareto_front(best_front, reference_front)
                    problem_results[algo_name]["pareto_front"] = norm_best

            # 导出所有运行的代际性能数据
            if "all_tracking" in result and result["all_tracking"]:
                self.data_exporter.export_generational_performance_all_runs(
                    problem_name,
                    algo_name,
                    result["all_tracking"],
                    algorithm_instances.get(algo_name)  # 传递算法实例以获取运算次数信息
                )

        # 导出帕累托前沿和解集
        self.data_exporter.export_pareto_front(problem_name, problem, problem_results)
        self.data_exporter.export_pareto_set(problem_name, problem, problem_results)

        # 导出汇总性能指标
        self.data_exporter.export_summary_metrics(problem_name, problem_results)

    def _generate_visualizations(self, problem, problem_name, problem_results):
        """生成问题的可视化图表"""
        # 绘制Pareto前沿比较图
        Visualizer.plot_pareto_front_comparison(
            problem,
            problem_results,
            save_path=os.path.join(self.save_dir, f"{problem_name}_pareto_front.png")
        )

        # 绘制Pareto解集比较图（如果可用）
        has_pareto_sets = any("pareto_set" in result for result in problem_results.values())
        if has_pareto_sets:
            Visualizer.plot_pareto_set_comparison(
                problem,
                problem_results,
                save_path=os.path.join(self.save_dir, f"{problem_name}_pareto_set.png")
            )

        # 绘制各指标收敛曲线
        metrics = ["igdf", "igdx", "rpsp", "hv", "sp"]
        for metric_name in metrics:
            Visualizer.plot_convergence(
                problem_results,
                metric_name=metric_name,
                problem_name=problem_name,
                save_path=os.path.join(self.save_dir, f"{problem_name}_{metric_name}_convergence.png"),
                sample_interval=10
            )

        # 绘制性能指标小提琴图/箱线图
        Visualizer.plot_algorithm_performance_boxplots(
            problem_results,
            problem_name,
            metrics=metrics,
            save_path=os.path.join(self.save_dir, f"{problem_name}_performance_boxplots.png"),
            sample_interval=10
        )

    def _save_summary(self, results, problems):
        """保存结果汇总 - 增强版支持IGDF、IGDX、RPSP、HV、SP指标"""
        summary_path = os.path.join(self.save_dir, "summary.txt")
        latex_path = os.path.join(self.save_dir, "summary_latex.tex")
        metrics_summary_path = os.path.join(self.save_dir, "metrics_summary.txt")

        # 首先创建简单的文本摘要
        with open(summary_path, "w") as f:
            f.write("====== 多目标优化实验结果汇总 ======\n\n")

            # 按问题组织结果
            for problem in problems:
                problem_name = problem.name
                f.write(f"问题: {problem_name}\n")
                f.write("-" * 60 + "\n")

                # 指标表格头
                f.write("\n性能指标:\n")
                algo_names = list(results.keys())
                header = "指标".ljust(15)
                for algo_name in algo_names:
                    header += (algo_name.ljust(20))
                f.write(header + "\n")
                f.write("-" * (15 + 20 * len(algo_names)) + "\n")

                # 填充指标值
                metrics = ["avg_igdf", "avg_igdx", "avg_rpsp", "avg_hv", "avg_sp"]
                metric_display = {
                    "avg_igdf": "IGDF (均值)",
                    "avg_igdx": "IGDX (均值)",
                    "avg_rpsp": "RPSP (均值)",
                    "avg_hv": "HV (均值)",
                    "avg_sp": "SP (均值)"
                }

                for metric in metrics:
                    line = metric_display[metric].ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name]:
                            value = results[algo_name][problem_name]["metrics"].get(metric, float('nan'))
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                # 添加标准差信息
                std_metrics = ["std_igdf", "std_igdx", "std_rpsp", "std_hv", "std_sp"]
                std_display = {
                    "std_igdf": "IGDF (标准差)",
                    "std_igdx": "IGDX (标准差)",
                    "std_rpsp": "RPSP (标准差)",
                    "std_hv": "HV (标准差)",
                    "std_sp": "SP (标准差)"
                }

                for metric in std_metrics:
                    line = std_display[metric].ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name]:
                            value = results[algo_name][problem_name]["metrics"].get(metric, float('nan'))
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                # 运行时间
                line = "运行时间".ljust(15)
                for algo_name in algo_names:
                    if problem_name in results[algo_name]:
                        value = np.mean(results[algo_name][problem_name]["runtimes"])
                        line += f"{value:.2f}s".ljust(20)
                    else:
                        line += "N/A".ljust(20)
                f.write(line + "\n\n")

                # Pareto前沿大小
                line = "前沿大小".ljust(15)
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "pareto_front" in results[algo_name][problem_name]:
                        value = len(results[algo_name][problem_name]["pareto_front"])
                        line += f"{value}".ljust(20)
                    else:
                        line += "N/A".ljust(20)
                f.write(line + "\n\n")

                f.write("=" * 60 + "\n\n")

            # 总体性能排名
            f.write("\n总体性能排名:\n")
            f.write("-" * 50 + "\n")

            # 计算每个算法在每个指标上的平均排名
            rankings = {algo_name: {"igdf": [], "igdx": [], "rpsp": [], "hv": [], "sp": []} for algo_name in algo_names}

            for problem in problems:
                problem_name = problem.name

                # 计算IGDF排名
                igdf_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_igdf" in results[algo_name][problem_name]["metrics"]:
                        igdf_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_igdf"]

                if igdf_values:
                    # 对IGDF值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(igdf_values.keys(),
                                          key=lambda x: igdf_values[x] if not np.isnan(igdf_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(igdf_values[algo_name]):
                            rankings[algo_name]["igdf"].append(rank)

                # 计算IGDX排名
                igdx_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_igdx" in results[algo_name][problem_name]["metrics"]:
                        igdx_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_igdx"]

                if igdx_values:
                    # 对IGDX值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(igdx_values.keys(),
                                          key=lambda x: igdx_values[x] if not np.isnan(igdx_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(igdx_values[algo_name]):
                            rankings[algo_name]["igdx"].append(rank)

                # 计算RPSP排名
                rpsp_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_rpsp" in results[algo_name][problem_name]["metrics"]:
                        rpsp_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_rpsp"]

                if rpsp_values:
                    # 对RPSP值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(rpsp_values.keys(),
                                          key=lambda x: rpsp_values[x] if not np.isnan(rpsp_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(rpsp_values[algo_name]):
                            rankings[algo_name]["rpsp"].append(rank)

                # 计算HV排名
                hv_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_hv" in results[algo_name][problem_name]["metrics"]:
                        hv_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_hv"]

                if hv_values:
                    # 对HV值进行排序（较大的值排名靠前）
                    sorted_algos = sorted(hv_values.keys(),
                                          key=lambda x: -hv_values[x] if not np.isnan(hv_values[x]) else float('inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(hv_values[algo_name]):
                            rankings[algo_name]["hv"].append(rank)

                # 计算SP排名
                sp_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_sp" in results[algo_name][problem_name]["metrics"]:
                        sp_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_sp"]

                if sp_values:
                    # 对SP值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(sp_values.keys(),
                                          key=lambda x: sp_values[x] if not np.isnan(sp_values[x]) else float('inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(sp_values[algo_name]):
                            rankings[algo_name]["sp"].append(rank)

            # 计算平均排名
            avg_rankings = {}
            for algo_name in algo_names:
                avg_igdf = np.mean(rankings[algo_name]["igdf"]) if rankings[algo_name]["igdf"] else float('nan')
                avg_igdx = np.mean(rankings[algo_name]["igdx"]) if rankings[algo_name]["igdx"] else float('nan')
                avg_rpsp = np.mean(rankings[algo_name]["rpsp"]) if rankings[algo_name]["rpsp"] else float('nan')
                avg_hv = np.mean(rankings[algo_name]["hv"]) if rankings[algo_name]["hv"] else float('nan')
                avg_sp = np.mean(rankings[algo_name]["sp"]) if rankings[algo_name]["sp"] else float('nan')

                # 计算综合排名（五个指标的平均值）
                valid_rankings = [r for r in [avg_igdf, avg_igdx, avg_rpsp, avg_hv, avg_sp] if not np.isnan(r)]
                avg_overall = np.mean(valid_rankings) if valid_rankings else float('nan')

                avg_rankings[algo_name] = {
                    "igdf": avg_igdf,
                    "igdx": avg_igdx,
                    "rpsp": avg_rpsp,
                    "hv": avg_hv,
                    "sp": avg_sp,
                    "overall": avg_overall
                }

            # 输出排名表格
            header = "算法".ljust(20) + "IGDF排名".ljust(15) + "IGDX排名".ljust(15) + "RPSP排名".ljust(
                15) + "HV排名".ljust(15) + "SP排名".ljust(15) + "综合排名".ljust(15)
            f.write(header + "\n")
            f.write("-" * (20 + 15 * 6) + "\n")

            # 按总体排名排序算法
            sorted_algos = sorted(avg_rankings.keys(), key=lambda x: avg_rankings[x]["overall"] if not np.isnan(
                avg_rankings[x]["overall"]) else float('inf'))

            for algo_name in sorted_algos:
                line = algo_name.ljust(20)
                line += f"{avg_rankings[algo_name]['igdf']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['igdx']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['rpsp']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['hv']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['sp']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['overall']:.2f}".ljust(15)
                f.write(line + "\n")

            f.write("\n注意: 排名值越低越好。IGDF、IGDX、RPSP、SP指标值越小越好，HV指标值越大越好。\n")

        # 创建LaTeX格式的表格 - 类似于图片中的格式
        with open(latex_path, "w") as f:
            f.write("% LaTeX表格格式 - 可直接复制到LaTeX文档中使用\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{各算法在不同测试问题上的性能指标。显著值以粗体标记。}\n")
            f.write("\\begin{tabular}{|l|l|" + "c|" * len(algo_names) + "}\n")
            f.write("\\hline\n")
            f.write("ID & & " + " & ".join(algo_names) + " \\\\ \\hline\n")

            # 按问题和指标填充表格
            for problem in problems:
                problem_name = problem.name

                # 对每个指标创建两行 (Mean和Std)
                for metric_base in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                    metric_mean = f"avg_{metric_base}"
                    metric_std = f"std_{metric_base}"

                    # 确定最优值 (IGDF,IGDX,RPSP,SP是越小越好，HV是越大越好)
                    best_values = {}
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if not np.isnan(value):
                                best_values[algo_name] = value

                    if best_values:
                        if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                            best_algo = min(best_values, key=best_values.get)
                            best_value = best_values[best_algo]
                        else:  # hv
                            best_algo = max(best_values, key=best_values.get)
                            best_value = best_values[best_algo]

                    # 均值行
                    f.write(f"{problem_name} & Mean & ")
                    mean_values = []
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if np.isnan(value):
                                mean_values.append("---")
                            else:
                                # 检查是否是最优值
                                is_best = False
                                if best_values:
                                    if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                                        is_best = np.abs(value - best_value) < 1e-6
                                    else:  # hv
                                        is_best = np.abs(value - best_value) < 1e-6

                                if is_best:
                                    mean_values.append(f"\\textbf{{{value:.4e}}}")
                                else:
                                    mean_values.append(f"{value:.4e}")
                        else:
                            mean_values.append("---")

                    f.write(" & ".join(mean_values) + " \\\\ \n")

                    # 标准差行
                    f.write(f" & Std & ")
                    std_values = []
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_std in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_std]
                            if np.isnan(value):
                                std_values.append("---")
                            else:
                                # 标准差不需要标记最优
                                std_values.append(f"{value:.4e}")
                        else:
                            std_values.append("---")

                    f.write(" & ".join(std_values) + " \\\\ \\hline\n")

            f.write("\\end{tabular}\n")
            f.write("\\label{tab:performance_metrics}\n")
            f.write("\\end{table}\n")

        # 创建每个指标的详细汇总
        with open(metrics_summary_path, "w") as f:
            f.write("====== 多目标优化实验详细指标汇总 ======\n\n")

            # 为每个指标创建独立表格
            for metric_base in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                metric_name = metric_base.upper()
                if metric_base == "igdf":
                    f.write(f"\n{metric_name} - 目标空间收敛性指标 (越小越好)\n")
                elif metric_base == "igdx":
                    f.write(f"\n{metric_name} - 决策空间收敛性指标 (越小越好)\n")
                elif metric_base == "rpsp":
                    f.write(f"\n{metric_name} - r-径向集逼近指标 (越小越好)\n")
                elif metric_base == "hv":
                    f.write(f"\n{metric_name} - 超体积指标 (越大越好)\n")
                elif metric_base == "sp":
                    f.write(f"\n{metric_name} - 解分布均匀性指标 (越小越好)\n")

                f.write("-" * 80 + "\n")

                # 表头
                header = "问题".ljust(15)
                for algo_name in algo_names:
                    header += (algo_name.ljust(20))
                f.write(header + "\n")
                f.write("-" * (15 + 20 * len(algo_names)) + "\n")

                # 填充每个问题的值
                for problem in problems:
                    problem_name = problem.name
                    metric_mean = f"avg_{metric_base}"

                    # 找出最优值
                    best_values = {}
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if not np.isnan(value):
                                best_values[algo_name] = value

                    if best_values:
                        if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                            best_algo = min(best_values, key=best_values.get)
                            best_value = best_values[best_algo]
                        else:  # hv
                            best_algo = max(best_values, key=best_values.get)
                            best_value = best_values[best_algo]

                    # 填充值
                    line = problem_name.ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                # 标记最优值
                                is_best = False
                                if best_values:
                                    if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                                        is_best = np.abs(value - best_value) < 1e-6
                                    else:  # hv
                                        is_best = np.abs(value - best_value) < 1e-6

                                if is_best:
                                    line += f"*{value:.6f}*".ljust(20)
                                else:
                                    line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                f.write("\n")

            # 添加综合排名
            f.write("\n总体排名汇总 (数值越小越好)\n")
            f.write("-" * 80 + "\n")

            header = "算法".ljust(15) + "IGDF排名".ljust(12) + "IGDX排名".ljust(12) + "RPSP排名".ljust(
                12) + "HV排名".ljust(12) + "SP排名".ljust(12) + "综合排名".ljust(12)
            f.write(header + "\n")
            f.write("-" * 80 + "\n")

            # 按综合排名排序
            for algo_name in sorted_algos:
                line = algo_name.ljust(15)
                line += f"{avg_rankings[algo_name]['igdf']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['igdx']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['rpsp']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['hv']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['sp']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['overall']:.2f}".ljust(12)
                f.write(line + "\n")


# ====================== 主函数 ======================

def main():
    """主函数，运行实验"""

    # 设置随机种子
    np.random.seed(42)
    random.seed(42)

    # 创建结果目录
    results_dir = "ZDT_results04"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 设置问题
    problems = [
        #ZDT1(n_var=30),
        #ZDT2(n_var=30),
        #ZDT3(n_var=30),
        ZDT4(n_var=10),
        #ZDT6(n_var=10),
        DTLZ1(n_obj=3),  # n_var will be calculated inside init
        #DTLZ2(n_obj=3),
        DTLZ3(n_obj=3),
        #DTLZ4(n_obj=3),
        #DTLZ5(n_obj=3),
        #DTLZ6(n_obj=3),
        DTLZ7(n_obj=3)
    ]

    # 设置算法
    algorithms = [
        CASMOPSO,
        MOPSO,
        NSGAII,
        MOEAD,
        GDE3,
    ]

    # 统一的最大迭代次数
    MAX_ITERATIONS = 300  # 所有算法的最大迭代次数

    # 开关：是否使用问题特定的CASMOPSO参数
    USE_PROBLEM_SPECIFIC_CASMOPSO = False  # True: 使用特定参数，False: 使用通用参数

    # 通用算法参数
    algorithm_params = {
        "MOPSO": {  # 使用新的动态参数接口
            "pop_size": 100,  # 种群大小
            "max_iterations": MAX_ITERATIONS,
            "w_init": 0.9,
            "w_end": 0.4,  # 动态惯性权重
            "c1_init": 2.5,
            "c1_end": 0.5,  # (等效于 c1=1.5)
            "c2_init": 0.5,
            "c2_end": 2.5,  # (等效于 c2=1.5)
            "use_archive": True,
            "archive_size": 100  # 标准存档大小
        },
        "NSGAII": {
            "pop_size": 100,  # 种群大小
            "max_generations": MAX_ITERATIONS,
            "pc": 0.8,  # 交叉概率
            "eta_c": 20,  # SBX交叉分布指数
            "pm_ratio": 1.0,  # 变异概率因子
            "eta_m": 20  # 多项式变异分布指数
        },
        "MOEAD": {
            "pop_size": 150,  # 子问题数量
            "max_generations": MAX_ITERATIONS,
            "T": 20,  # 邻域大小
            "delta": 0.9,  # 邻域选择概率
            "nr": 2  # 最大替代数量
        },
        "CASMOPSO": {  # CASMOPSO的通用参数
            "pop_size": 100,
            "max_iterations": MAX_ITERATIONS,
            "w_init": 0.9,
            "w_end": 0.4,
            "c1_init": 2.5,
            "c1_end": 0.5,
            "c2_init": 2.5,
            "c2_end": 0.5,
            "use_archive": True,
            "archive_size": 100,
            "mutation_rate": 0.1,
            "adaptive_grid_size": 25,
            "k_vmax": 0.5
        },
        "GDE3": {
            "pop_size": 100,
            "max_generations": MAX_ITERATIONS,
            "F": 0.5,  # 缩放因子 F
            "CR": 0.9  # 交叉概率 CR
        },
    }

    # 创建问题特定的CASMOPSO参数字典（如果需要）
    problem_specific_params = {}
    if USE_PROBLEM_SPECIFIC_CASMOPSO:
        problem_specific_params = {"CASMOPSO": {}}
        for problem in problems:
            problem_specific_params["CASMOPSO"][problem.name] = get_optimal_casmopso_params(
                problem.name, MAX_ITERATIONS)

    # 创建实验框架
    experiment = ExperimentFramework(save_dir=results_dir, export_data=True, normalize_results=True)

    # 运行实验，传入问题特定参数
    results = experiment.run_experiment(
        problems=problems,
        algorithms=algorithms,
        algorithm_params=algorithm_params,  # 传入通用参数
        problem_specific_params=problem_specific_params if USE_PROBLEM_SPECIFIC_CASMOPSO else None,
        # 新增：传入问题特定参数
        n_runs=1,  # 减少运行次数以节省时间
        verbose=True
    )

    print(f"实验完成! 结果已保存到 {results_dir} 目录")
    print(f"详细数据已导出到 {os.path.join(results_dir, 'exported_data')} 目录，格式如下:")
    print("1. 性能数据: exported_data/performance/<问题名称>/")
    print("   - <算法名称>_run_<运行ID>_generational.txt: 每次运行的完整代际性能数据，包含运算次数")
    print("   - <算法名称>_all_runs_summary.txt: 所有运行的汇总信息")
    print("   - summary_metrics.txt: 问题的所有算法汇总性能指标")
    print("2. 帕累托前沿数据: exported_data/pareto_fronts/<问题名称>_pareto_fronts.txt (包含所有运行的前沿)")
    print("3. 帕累托解集数据: exported_data/pareto_sets/<问题名称>_pareto_sets.txt (包含所有运行的解集)")


if __name__ == "__main__":
    main()
