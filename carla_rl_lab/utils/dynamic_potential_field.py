import math
import numpy as np

def calculate_clockwise_angle(x0, y0, ao_degrees):
    # 计算边a的逆时针角度（弧度）
    angle_a_rad = np.arctan2(y0,  x0)
    # 转换为角度制，并转为顺时针角度（0-360）
    angle_a_clockwise = (360 - np.degrees(angle_a_rad))  % 360

    # 边b的角度处理（确保在0-360范围内）
    angle_b_clockwise = ao_degrees % 360

    # 计算顺时针角度差
    angle_diff = (angle_b_clockwise - angle_a_clockwise) % 360

    return angle_diff



def old_change_new(x_old, y_old, theta_rad):
    """把 (x_old,y_old) 绕原点逆时针旋转 theta_rad"""
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_new = x_old * cos_t - y_old * sin_t
    y_new = x_old * sin_t + y_old * cos_t
    return x_new, y_new



  #τ is the critical threshold of the safe distance
#α is the undetermined parameter related to velocity
def pf_base(rel_x,rel_y,vo,θo,ao,τ=1,alpha=0.16,λ=1,β=-0.3):
    rel_xx,rel_yy=old_change_new(rel_x, rel_y, θo/180 * np.pi)
    kk=(((-rel_xx)*τ/np.exp(alpha*vo))**2)+(((-rel_yy)*τ)**2)
    k=np.sqrt(kk)
    #m_v=304.3*np.exp(-((vo-15.35)/9.135)**2)
    m_v=1.5*(np.log(vo + 1) / np.log(3))+0.3


    thr = [calculate_clockwise_angle(-rel_xx, -rel_yy, θo)]
    thr = thr
    # e_new = (m_v * λ) / k * np.exp(β * ao *np.cos(thr/180 * np.pi))
    e_new = (m_v * λ) / k
    return np.float64(np.log(np.clip(e_new - 0.3, 0, 1) + 1))


# R = np.log(pf_base(1,1,8,0,0) +1)
# print(R)
# #数据处理
# x = np.linspace(-30,30,256)
# y = np.linspace(-30,30,256)
# X,Y = np.meshgrid(x,y)
# Z = pf_base(X,Y,8,135,0)


# #画图
# import numpy as np
# import matplotlib.pyplot  as plt
# from mpl_toolkits.mplot3d  import Axes3D
# from matplotlib.colors  import BoundaryNorm

# fig = plt.figure(figsize=(5,  4))
# ax = fig.add_subplot(111)
# # 定义非均匀刻度及颜色映射规则
# ticks = [0, 1e-10, 0.2, 0.4, 0.6, 0.8, 1-(1e-10), 1]  # 用户指定的非均匀刻度
# bounds = ticks  # 边界与ticks一致
# norm = BoundaryNorm(bounds, ncolors=256)  # 关键：颜色均匀分布在bounds区间
# # 绘制热力图（注意变量名一致性，原代码中im应为heatmap）
# heatmap = ax.imshow(Z,  cmap='coolwarm', norm=norm, extent=[-30, 30, -30, 30])  # 在此处设置norm
# # 添加颜色条（不再传递norm）
# cbar = fig.colorbar(heatmap,  ticks=ticks)  # 修正变量名从im到heatmap
# cbar.set_label('Value')
# # 设置坐标轴标签
# ax.set_xlabel('X  Position')
# ax.set_ylabel('Y  Position')
# ax.set_title('v=0,a=0,Θ=0', y=-0.2, pad=10)


# plt.show()

class DynamicPF:
    def __init__(self , alpha):
        self.tau   = 1.0
        self.alpha = float(alpha)
        self.lambd = 0.3
        self.beta  = -0.3
        self.gamma_upadate = 0.99
        self.cigma_update = 0.0003


    def __call__(self,rel_x: float, rel_y: float ,vo: float, theta_o: float, ao: float):
        # 1. 坐标旋转
        rel_xx, rel_yy = old_change_new(rel_x, rel_y, theta_o/180 * np.pi)
        # 2. 计算 kk
        vo_kk_shape = vo * 0.6585 + 6.83
        kk = (((-rel_xx) * self.tau / np.exp(self.alpha * vo_kk_shape)) ** 2 +
              ((-rel_yy) * self.tau) ** 2)
        k = np.sqrt(kk)
        # 3. 计算 m_v
        vo_shape = vo * 0.8325 + 3.35
        m_v = 2.05 * (np.log(vo_shape + 1) / np.log(3)) + 0.01
        # 4. 计算 thr（顺时针角度差）
        thr = calculate_clockwise_angle(-rel_xx, -rel_yy, 0.0)
        # 5. 计算 e_new
        e_new = (m_v * self.lambd) / k * np.exp(
            self.beta * ao * np.cos(np.deg2rad(thr)))
        # 6. 返回 log(clip(e,0,1)+1)
        return np.float64(np.log(np.clip(e_new, 0.0, 1.0) + 1.0))


    def update_alpha(self, r_adv, tht_new, tht_old):
        td_error = r_adv + self.gamma_upadate*tht_new - tht_old
        self.alpha -= self.cigma_update * td_error
        self.alpha =np.clip(self.alpha, 0.13 ,0.16)

# #一次模拟的示意
# pf = DynamicPF(alpha = 0.1336)
# obs_apf = pf (1,2,10,45,30)
# print('value',obs_apf)
# #模拟更新
# pf.update_alpha(1,0.8,0.75)
# print('new_alpha',pf.alpha)
