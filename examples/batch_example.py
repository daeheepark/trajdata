from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch


def main():
    noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        centric="agent",
        desired_dt=0.5,
        history_sec=(0.5, 2.),
        future_sec=(1.5, 6.),
        only_predict=[AgentType.VEHICLE],
        max_neighbor_num=0,
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 1,
            "map_size_px": 64,
            # "offset_frac_xy": (-0.5, 0.0),
        },
        incl_vector_map=False,
        augmentations=None,
        ego_only=True,
        num_workers=0,
        verbose=True,
        # rebuild_cache=True,
        cache_location="/ssd4tb/Datasets/unified_data_cache",
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "datasets/nuScenes",
        },
    )
    # dataset = UnifiedDataset(
    #     desired_data=["nusc_mini-mini_train"],
    #     centric="agent",
    #     desired_dt=0.1,
    #     history_sec=(3.2, 3.2),
    #     future_sec=(4.8, 4.8),
    #     only_predict=[AgentType.VEHICLE],
    #     agent_interaction_distances=defaultdict(lambda: 30.0),
    #     incl_robot_future=False,
    #     incl_raster_map=True,
    #     raster_map_params={
    #         "px_per_m": 2,
    #         "map_size_px": 224,
    #         "offset_frac_xy": (-0.5, 0.0),
    #     },
    #     augmentations=[noise_hists],
    #     num_workers=0,
    #     verbose=True,
    #     data_dirs={  # Remember to change this to match your filesystem!
    #         "nusc_mini": "~/datasets/nuScenes",
    #     },
    # )

    print(f"# Data Samples: {len(dataset):,}")

    # for data in dataset:
    #     pass

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        if batch.scene_ids[0] == 'scene-0061' and batch.scene_ts == 11:
            plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()
