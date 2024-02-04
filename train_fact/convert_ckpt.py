import torch 
import argparse


def add_additional_channels(state_dict, num_additional_channels):
    "state_dict should be just from unet model, not the entire SD or GLIGEN"

    if num_additional_channels != 0:
    
        new_conv_weight = torch.zeros(320, 4+num_additional_channels, 3, 3 )

        for key,value in state_dict.items():
            if key == "input_blocks.0.0.weight":
                old_conv_weight = value
                new_conv_weight[:,0:4,:,:] = old_conv_weight
                state_dict[key] = new_conv_weight






if __name__ == "__main__":
    # The following code will add additional 5 channels (for inpainting) to a GLIGEN ckpt 

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,  default=None, help="")
    parser.add_argument("--new_ckpt_path", type=str,  default=None, help="")
    args = parser.parse_args()


    new_conv_weight = torch.zeros(320, 4+4+1, 3, 3 )

    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    for key,value in ckpt["model"].items():
        if key == "input_blocks.0.0.weight":
            old_conv_weight = value
            new_conv_weight[:,0:4,:,:] = old_conv_weight
            ckpt["model"]["input_blocks.0.0.weight"] = new_conv_weight

    save = {"model":ckpt["model"]}
    torch.save(save, args.new_ckpt_path) 

