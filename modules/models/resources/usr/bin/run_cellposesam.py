from pathlib import Path

import torch
import yaml
from cellpose import models
from model_utils import get_device
from utils import create_argparser_inference, get_model_name_type, load_img, save_masks


def run_cellpose(
    save_dir: Path | str,
    save_name: str,
    idxs: list[int, ...],
    config: dict,
    model_chkpt: str,
    device: torch.device,
    output_mask_type: str,
):
    # NOTE: This is just a link to the Cellpose-SAM model, but circumvents Cellpose's fixed model location
    # Initialize Cellpose-SAM model
    cp_model = models.CellposeModel(
        gpu=device.type == "cuda",
        pretrained_model=str(Path(model_chkpt).readlink()),
    )
    # Extract model config arguments
    # NOTE: rescale and channels not used anymore
    # NOTE: more than 3 channels not used
    masks, _, _ = cp_model.eval(
        img,
        batch_size=config["batch_size"],
        flow_threshold=config["flow_threshold"],
        cellprob_threshold=config["cellprob_threshold"],
        z_axis=config["z_axis"],
        channel_axis=config["channel_axis"],
        do_3D=config["do_3D"],
        anisotropy=config["anisotropy"],
        stitch_threshold=config["stitch_threshold"],
        min_size=config["min_size"],
        max_size_fraction=config["max_size_fraction"],
    )

    save_masks(Path(save_dir), save_name, masks, idxs=idxs, mask_type=output_mask_type)


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()
    with open(cli_args.model_config) as f:
        config = yaml.safe_load(f)

    # Load image and apply preprocessing if specified
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )
    # load_img() above explicitly returns CZYX
    axes = list("CZYX")
    
    # Remove singleton C/Z dimensions while keeping track of their meaning.
    # Do not infer axes from dimension lengths.
    for axis_name in ("C", "Z"):
        axis_idx = axes.index(axis_name)
    
        if img.shape[axis_idx] == 1:
            img = img.squeeze(axis=axis_idx)
            axes.pop(axis_idx)
    
    config["channel_axis"] = (
        axes.index("C")
        if "C" in axes
        else None
    )
    
    config["z_axis"] = (
        axes.index("Z")
        if "Z" in axes
        else None
    )
    
    config["do_3D"] = (
        config["z_axis"] is not None
        and img.shape[config["z_axis"]] > 1
    )

    device = get_device(model_type=get_model_name_type(cli_args.model_type))

    run_cellpose(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        config=config,
        model_chkpt=cli_args.model_chkpt,
        device=device,
        output_mask_type=cli_args.output_mask_type
        if cli_args.output_mask_type != "auto"
        else "instance",
    )
