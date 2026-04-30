import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib
import warnings
import plotly.graph_objects as go
from torch_geometric.graphgym.config import cfg
from registry import register

matplotlib.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from FeatureExtraction.CreateBox import CreateBox

"""将自定义的图转换为networkx图"""
def getNetworkx(gra):
    g = nx.Graph()
    #添加节点id和节点属性
    for id, node in gra.nodes.items():
        g.add_node(id, **node.toDict())
    #添加边两边节点的id和边的属性
    for value in gra.edges.values():
        g.add_edge(value.vertex[0], value.vertex[1], distance=value.distance)

    return g

"""建立一个可视化的图"""
def showGraph(gra):
    g = getNetworkx(gra)
    pos = {}
    for key, value in gra.nodes.items():
        pos[key] = (value.currentPosX, value.currentPosY)

    xRange, yRange = getRange(pos)
    width = xRange[1] - xRange[0]
    heigth = yRange[1] - yRange[0]
    #创建绘图，并确保图形不会太扁平
    figWidth = max(8, width)
    figHeigth = max(6, heigth)

    minRatio = 0.6
    maxRatio = 1.8

    currentAspect = figWidth / figHeigth

    if currentAspect < minRatio:
        #图形太窄，增加宽度
        figWidth = figHeigth * minRatio
    elif currentAspect > maxRatio:
        figHeigth = figWidth / maxRatio

    fig,ax = plt.subplots(figsize=(figWidth, figHeigth))

    #绘制边
    nx.draw_networkx_edges(
        g,pos,
        edgelist=g.edges,
        arrows=False,
        edge_color='gray',
        width=2,
        ax=ax
    )

    #绘制节点和方向指示器
    for key, (x, y) in pos.items():
        direction = gra.nodes[key].normalVectors
        #绘制节点
        nodeCircle = plt.Circle((x,y),0.1,color='lightblue',alpha = 0.7)
        ax.add_patch(nodeCircle)

        #绘制方向箭头
        arrowLength = 0.5
        rad = np.deg2rad(direction)
        dx = arrowLength * np.cos(rad)
        dy = arrowLength * np.sin(rad)

        arrow = FancyArrowPatch(
            (x, y),
            (x + dx, y + dy),
            arrowstyle='->',
            mutation_scale=15,
            color='red',
            linewidth=2
        )
        ax.add_patch(arrow)

        #添加节点标签
        plt.text(x, y + 0.15, gra.nodes[key].modelName,
                 ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    #设置坐标轴
    ax.set_xlim(xRange[0], xRange[1])
    ax.set_ylim(yRange[0], yRange[1])
    ax.set_aspect('equal')
    plt.title(f"{gra.name}沙盘中沙具的分布情况", fontsize=16)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


    """设置鼠标悬停事件，显示信息"""
    def hover(event):
        if event.inaxes == ax:
            # 检查是否需要移除之前的注释
            if hasattr(hover, 'annotation'):
                hover.annotation.remove()
                delattr(hover, 'annotation')
                fig.canvas.draw_idle()

            # 检查鼠标是否在节点附近
            for key, (x, y) in pos.items():
                distance = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                if distance < 0.15:
                    # 获取节点
                    node = gra.nodes[key]

                    # 显示信息
                    hover.annotation = ax.annotate(
                        f"沙具名称：{node.modelName}\n"
                        f"沙具坐标：({node.currentPosX},{node.currentPosY})\n"
                        f"沙具深度：{node.currentPosZ}",
                        xy=(x, y), xycoords='data',
                        xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
                    )
                    fig.canvas.draw_idle()
                    break

    """设置鼠标点击事件，显示边"""
    def onclick(event):
        if event.inaxes == ax:
            for key, (x, y) in pos.items():
                distance = np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)
                if distance < 0.15:
                    # 获取节点
                    node = gra.nodes[key]
                    print(f"点击了{node.modelName}")

                    #高亮显示连接的边
                    highlight = plt.Circle((x, y), 0.12, color='orange', alpha=0.5 )
                    ax.add_patch(highlight)
                    fig.canvas.draw_idle()

                    #1秒后移除高亮
                    def removeLight():
                        highlight.remove()
                        fig.canvas.draw_idle()

                    fig.canvas.start_event_loop(1)
                    removeLight()
                    break

    #绑定事件
    fig.canvas.mpl_connect('motion_notify_event', hover)
    # fig.canvas.mpl_connect('button_press_event', onclick)

"""用Plotly库实现绘制功能"""
def printGraph(gra):
    g = getNetworkx(gra)
    pos = {}
    for key, value in gra.nodes.items():
        pos[key] = (value.currentPosX, value.currentPosY)

    # 边轨迹
    edgeX, edgeY = [], []
    for edge in gra.edges.values():
        x0, y0 = pos[edge.vertex[0]]
        x1, y1 = pos[edge.vertex[1]]
        edgeX.extend([x0, x1, None])
        edgeY.extend([y0, y1, None])
    edgeTrace = go.Scatter(x=edgeX, y=edgeY, mode='lines', line=dict(width=2, color='gray'))

    # 节点轨迹
    nodeX, nodeY, nodeText = [], [], []
    for key, (x, y) in pos.items():
        nodeX.append(x)
        nodeY.append(y)
        nodeText.append(
            f"沙具名称：{gra.nodes[key].modelName}<br>"
            f"沙具坐标：({gra.nodes[key].currentPosX},{gra.nodes[key].currentPosY})<br>"
            f"沙具深度：{gra.nodes[key].currentPosZ}"
        )
    nodeTrace = go.Scatter(
        x=nodeX, y=nodeY, mode='markers+text',
        text=[node for node in g.nodes()], textposition="middle center",
        marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
        textfont=dict(size=12, color='black'),
        hovertemplate='%{hovertext}<extra></extra>', hovertext=nodeText
    )

    # 箭头轨迹（根据原始坐标计算）
    # 计算合适的箭头长度（取宽高最小值/40，）
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    width_range = max(x_vals) - min(x_vals) if x_vals else 1
    height_range = max(y_vals) - min(y_vals) if y_vals else 1
    arrow_len = min(width_range, height_range) / 40

    arrowX, arrowY = [], []
    for key, (x, y) in pos.items():
        direction = gra.nodes[key].normalVectors  # 假设是角度（度）
        rad = np.deg2rad(direction)
        dx = arrow_len * np.cos(rad)
        dy = arrow_len * np.sin(rad)
        arrowX.extend([x, x + dx, None])
        arrowY.extend([y, y + dy, None])
    arrowTrace = go.Scatter(x=arrowX, y=arrowY, mode='lines', line=dict(width=3, color='red'))

    # 创建图形
    fig = go.Figure(
        data=[edgeTrace, nodeTrace, arrowTrace],
        layout=go.Layout(
            title=dict(text=f'沙盘{gra.name}的分布情况', font=dict(size=16)),
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            # 可选：锁定坐标轴比例
            # yaxis=dict(showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1)
        )
    )
    fig.show()

"""计算节点坐标的范围"""
def getRange(pos):
    #提取二向坐标
    xCoords = [coord[0] for coord in pos.values()]
    yCoords = [coord[1] for coord in pos.values()]

    #计算最大值和最小值
    xMin, xMax = min(xCoords), max(xCoords)
    yMin, yMax = min(yCoords), max(yCoords)

    currentWidth = xMax - xMin
    currentHeigth = yMax - yMin
    targetRatio = 1.0
    if currentWidth == 0:
        currentWidth == 1
    if currentHeigth == 0:
        currentHeigth ==1
    currentRatio = currentWidth / currentHeigth

    #调整坐标范围以达到目标宽高比
    if currentRatio > targetRatio:
        #图形太宽，需要增加高度
        needHeigth = currentWidth / targetRatio
        heigthAdd = (needHeigth - currentHeigth) / 2
        yMin -= heigthAdd
        yMax += heigthAdd
    else:
        #图形太高，需要增加高度
        needWidth = currentHeigth * targetRatio
        widthAdd = (needWidth - currentWidth)/2
        xMin -= widthAdd
        xMax += widthAdd

    #增加坐标边距
    xMargin = (xMax - xMin) * 0.1
    yMargin = (yMax - yMin) * 0.1

    #记录坐标范围
    xRange = (xMin - xMargin, xMax + xMargin)
    yRange = (yMin - yMargin, yMax + yMargin)

    return xRange, yRange

@register('visu')
def shows():

    # 忽略特定警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    dataDir = cfg.structDir
    filename = f'{cfg.visual.graphId}.csv'

    cb = CreateBox()
    gra = cb.createGraph(dataDir, filename)
    """matplotlib"""
    if cfg.visual.tool == 'matplotlib':
        showGraph(gra)

    """使用Plotly创建图形"""
    if cfg.visual.tool == 'plotly':
        printGraph(gra)
