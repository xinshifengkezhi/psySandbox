import argparse
import logging
from torch_geometric.graphgym.config import cfg, set_cfg
import CustomConfig
import FeatureExtraction.OperationSequence
import FeatureExtraction.CreateBox
import FeatureExtraction.Visual
import FeatureExtraction.DataEmhance
import FeatureExtraction.Model.TrainEntrance
import FeatureExtraction.Model.explain
import FeatureExtraction.BaseModle.Start
import FeatureExtraction.Forecast.start

from registry import operations

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Config file path')
    args = parser.parse_args()

    set_cfg(cfg)
    cfg.merge_from_file(args.cfg)

    runModule()

"""根据配置中的workFlows来判断该执行哪个模块"""
def runModule():

    flows = cfg.workFlows
    for i in flows:
        operations[cfg.workModule[f'task{i}']]()

if __name__ == '__main__':
    main()
