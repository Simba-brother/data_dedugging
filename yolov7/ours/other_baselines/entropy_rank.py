
import os
import math
import joblib
from ours.small_utils import read_json
from queue import PriorityQueue
from ours.data_organization_tools import get_all_gids
from ours.base_data_manager import get_collected_gt_box_json_path

def calcu_entropy(prob_list):
    entropy = 0.0
    for p in prob_list:
        entropy -= p * math.log(p)
    return entropy

def main(g_json_path,match_json_path):
    g_json = read_json(g_json_path)
    match_json = read_json(match_json_path)
    all_gids = get_all_gids(g_json)
    priority_queue = PriorityQueue() # 越小优先级越高
    matchid_gid_set = set() # 存储匹配到p_box的g_box的id
    for gid in match_json.keys():
        p_box = match_json[gid]["p_box"]
        probs = p_box["prob"]
        entropy = calcu_entropy(probs) # 熵越小说明概率分布越明确，所以熵越大的probs说明越可能有问题
        priority_queue.put((-entropy,gid))# entropy越大优先级越高
        matchid_gid_set.add(int(gid))

    for g_id in all_gids:
        if g_id not in matchid_gid_set:
            priority_queue.put((-100,g_id))

    # 获取并弹出优先级最高的元素
    gid_rank = []
    while not priority_queue.empty():
        priority, g_id = priority_queue.get()
        gid_rank.append(g_id)
    return gid_rank


if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    exp_id = "01"
    g_json_path = get_collected_gt_box_json_path(dataset_name)
    match_json_path = os.path.join(exp_root_dir, "collection_indicator_bbox_level",
                              dataset_name,model_name,"other_baselines",
                              "gp_box_match","match.json")
    gid_rank = main(g_json_path,match_json_path)

    save_dir = os.path.join(exp_root_dir,"Results","other_baselines","entropy",
                            dataset_name,model_name,f"exp_{exp_id}","rank")
    save_file_name = "rank.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(gid_rank,save_path)
    print(f"rank长度为:{len(gid_rank)}")
    print(f'rank结果保存在:{save_path}')

