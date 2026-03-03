# data_dedugging

数据的错误注入：
`data_debugging/DataDetective/fuxian/fault_gen.py`
数据标签格式转换：
`/home/mml/workspace/data_debugging/yolov7/ours/labelconver/labelconver.py`

方法：
ours:
    收集所有标注框信息：
    `/home/mml/workspace/data_debugging/yolov7/ours/collect_train_info.py/collect_gt_box`
    收集所有预测框信息：
    `/home/mml/workspace/data_debugging/yolov7/ours/collect_train_info.py/collect_predicted_box`
    标注框和预测框匹配操作：
    `/home/mml/workspace/data_debugging/yolov7/ours/match_and_collect_metrics.py`
    rank步骤：
    `/home/mml/workspace/data_debugging/yolov7/ours/rank/rank.py`
    修复步骤：
    `/home/mml/workspace/data_debugging/yolov7/ours/repair/repair_main.py`


datactive:
    训练分类模型：
    `/home/mml/workspace/data_debugging/DataDetective/fuxian/train_classmodel.py`
    推理分类模型：
    `/home/mml/workspace/data_debugging/DataDetective/fuxian/inference_classmodel.py`
    排序：
    `/home/mml/workspace/data_debugging/DataDetective/fuxian/detective.py`


