import numpy as np
import os,sys,time
import torch
import importlib
import glob
import options
from util import log

data = "./data/0142/*" ## Make sure it's 04d for now

def main():


    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):
        for path in sorted(glob.glob(data, recursive=True)):
            image_fname = path
            # print(path)
            model = importlib.import_module("model.{}".format(opt.model))
            m = model.Model(opt)

            m.load_dataset(opt, image_fname)
            m.build_networks(opt)
            m.setup_optimizer(opt)
            m.restore_checkpoint(opt)
            m.setup_visualizer(opt)

            m.train(opt, path)
    #save 'barf transfer' video output
    # save_path = f"./output/{opt.group}/{opt.name}_{opt.seed}/vis/%04d.png"
    # print(save_path)
    # os.system("ffmpeg -y -framerate 30 '-i {save_path}' -pix_fmt yuv420p -crf 10 ./output/{opt.name}_barftransfer.mp4")

if __name__=="__main__":
    main()
