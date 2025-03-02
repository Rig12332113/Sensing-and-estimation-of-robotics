
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from icp import ICP

if __name__ == "__main__":
  obj_name = 'liq_container' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)
  mean_source = np.mean(source_pc, axis=0)
  delta_source = source_pc - mean_source
                        
  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)
    # estimated_pose, you need to estimate the pose with ICP
    pose = ICP(source_pc, target_pc, 20, 0)

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)

