import json
import csv
import os
import re
from registry import register
from torch_geometric.graphgym.config import cfg

def parseCsv(fileContent: str, outputFile: str = 'structured_data.csv'):
    """
    将文本文件中的字符串数据解析为结构化数据并保存为CSV文件
    """
    # 清理和解析数据
    try:
        # 尝试直接解析整个文件内容
        dataList = json.loads(fileContent)
    except:
        # 如果直接解析失败，尝试手动处理
        # 移除开头和结尾的方括号，然后分割各个JSON对象
        content = fileContent.strip().strip('[]')
        # 使用正则表达式分割各个JSON对象
        pattern = r'\"({.*?})\"'
        matches = re.findall(pattern, content)
        dataList = [match for match in matches if match.strip()]

    structData = []

    for itemStr in dataList:
        try:
            # 解析单个JSON对象
            if isinstance(itemStr, str):
                itemData = json.loads(itemStr)
            else:
                itemData = itemStr

            # 基础字段
            record = {
                'ID': itemData.get('ID'),
                'handleTime': itemData.get('handleTime'),
                'useTime': itemData.get('useTime'),
                'handleType': itemData.get('handleType'),
                'handleObj': itemData.get('handleObj'),
                'currentPosX': itemData.get('currentPosX'),
                'currentPosY': itemData.get('currentPosY'),
                'currentPosZ': itemData.get('currentPosZ'),
                'currentRotaW': itemData.get('currentRotaW'),
                'currentRotaY': itemData.get('currentRotaY'),
                'currentScaleX': itemData.get('currentScaleX'),
                'currentScaleY': itemData.get('currentScaleY'),
                'currentScaleZ': itemData.get('currentScaleZ'),
                'handlePosX': itemData.get('handlePosX'),
                'handlePosY': itemData.get('handlePosY'),
                'handlePosZ': itemData.get('handlePosZ'),
                'handleScaleX': itemData.get('handleScaleX'),
                'handleScaleY': itemData.get('handleScaleY'),
                'handleScaleZ': itemData.get('handleScaleZ')
            }

            # 处理HitPoint数据
            hitPoints = itemData.get('protobufClass_HitPoint', [])
            if hitPoints:

                for i, point in enumerate(hitPoints):
                    record[f'hitPoint-{i}-x'] = point.get('x')
                    record[f'hitPoint-{i}-y'] = point.get('y')
                    record[f'hitPoint-{i}-z'] = point.get('z')

            structData.append(record)

        except json.JSONDecodeError as e:
            print(f"解析错误: {e}")
            continue
        except Exception as e:
            print(f"处理错误: {e}")
            continue

    # 写入CSV文件
    if structData:
        # 获取所有可能的字段名
        allFields = set()
        for record in structData:
            allFields.update(record.keys())

        # 排序字段名，使基础字段在前
        baseFields = ['ID', 'handleTime', 'useTime', 'handleType', 'handleObj',
                       'currentPosX', 'currentPosY', 'currentPosZ', 'currentRotaW',
                       'currentRotaY', 'currentScaleX', 'currentScaleY', 'currentScaleZ',
                       'handlePosX', 'handlePosY', 'handlePosZ', 'handleScaleX',
                       'handleScaleY', 'handleScaleZ']

        # 分离HitPoint字段并按数字排序
        hitpointFields = [f for f in allFields if f.startswith('hitPoint-')]
        hitpointFields.sort(key=lambda x: (int(x.split('-')[1]), x.split('-')[2]))

        # 合并字段顺序
        fieldnames = baseFields + hitpointFields

        with open(outputFile, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in structData:
                # 确保所有字段都存在
                for field in fieldnames:
                    if field not in record:
                        record[field] = ''
                writer.writerow(record)

        print(f"数据已成功保存到 {outputFile}")
        print(f"总共处理了 {len(structData)} 条记录")
        print(f"CSV文件包含 {len(fieldnames)} 个字段")

        # 显示一些统计信息
        handleTypes = {}
        for record in structData:
            handleType = record.get('handleType', 'Unknown')
            handleTypes[handleType] = handleTypes.get(handleType, 0) + 1

        print("\n操作类型统计:")
        for handleType, count in handleTypes.items():
            print(f"  {handleType}: {count} 次")

    else:
        print("没有找到有效数据")

"""将csv文件输出到指定目录下"""
def readFile(filePath, graphId, outDir):
    with open(filePath, 'r', encoding='utf-8') as file:
        content = file.read()

    outputFile = os.path.join(outDir, f'{graphId}.csv')
    parseCsv(content, outputFile)

@register('OperSeq')
def dealFile():
    outDir = cfg.structDir
    try:
        os.mkdir(outDir)
    except:
        print(f'{outDir}文件夹已创建')

    dealDir = cfg.dataDir
    filePath = os.path.join(dealDir, 'all_sp_operations_detail')
    #遍历sandplay下的所有文件
    for filename in os.listdir(filePath):
        sourcePath = os.path.join(filePath, filename)
        graphId = filename[0:-4]
        readFile(sourcePath, graphId, outDir)

