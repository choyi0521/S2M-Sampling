import torch
import random
import glob
import json
from get_cls import get_cls
from misc import load_checkpoint, dict2clsattr, fix_all_seed, set_deterministic_op_trainable, set_grad
from os.path import abspath, exists, join
from sampling import Sampler
from argparse import ArgumentParser
from torch.backends import cudnn
import config


def load_gen(DATA, MODEL, MODULES, device):
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=["something"])
    Gen = module.Generator(z_dim=MODEL.z_dim,
                            g_shared_dim=MODEL.g_shared_dim,
                            img_size=DATA.img_size,
                            g_conv_dim=MODEL.g_conv_dim,
                            apply_attn=MODEL.apply_attn,
                            attn_g_loc=MODEL.attn_g_loc,
                            g_cond_mtd=MODEL.g_cond_mtd,
                            num_classes=DATA.num_classes,
                            g_init=MODEL.g_init,
                            g_depth=MODEL.g_depth,
                            mixed_precision=False,
                            MODULES=MODULES).to(device)
    return Gen


def load_gen_cls(cfgs, DATA, MODEL, MODULES, device):
    Gen = load_gen(DATA, MODEL, MODULES, device)
    Cls = get_cls(DATA.name, MODEL.d_cond_mtd).to(device)

    when = "best"
    print(cfgs.checkpoint_dir)
    if not exists(abspath(cfgs.checkpoint_dir)):
        raise NotADirectoryError
    
    g_checkpoint_dir = glob.glob(join(cfgs.checkpoint_dir,"model=G_ema-best-weights-step*.pth".format(when=when)))[0]
    Gen = load_checkpoint(Gen, g_checkpoint_dir)
    c_checkpoint_dir = glob.glob(join(cfgs.checkpoint_dir,f"model=C-best-weights-step*.pth"))[0]
    Cls = load_checkpoint(Cls, c_checkpoint_dir)

    set_grad(Gen, False)
    set_grad(Cls, False)
    Gen.eval()
    Gen.apply(set_deterministic_op_trainable)
    Cls.eval()

    return Gen, Cls


def sample(cfgs, DATA, MODEL, Gen, Cls, device):
    session_dir = join(cfgs.result_dir, cfgs.session_name)
    sampler = Sampler(
        session_dir=session_dir,
        Gen=Gen,
        Cls=Cls,
        d_cond_mtd=MODEL.d_cond_mtd,
        dataset_name=DATA.name,
        z_dim=MODEL.z_dim,
        gammas=cfgs.gammas,
        temperature_logits=cfgs.temperature_logits,
        temperature_validity=cfgs.temperature_validity,
        device=device
    )

    sampler.mh(
        batch_size=cfgs.batch_size,
        intersection_indices=cfgs.intersection_indices,
        difference_indices=cfgs.difference_indices,
        n_steps=cfgs.n_steps,
        save_img=True,
        save_every=cfgs.save_every
    )


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1, help='seed for generating random numbers')

    # sampling
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--result_dir', type=str, default='./results', help='results directory')
    parser.add_argument('--session_name', type=str, default=None, help='name of directory for saving images')
    parser.add_argument('--n_steps', type=int, default=200, help='MCMC steps for Metropolis-hastings sampling')
    parser.add_argument('--intersection_indices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--difference_indices', type=int, nargs='+', default=[])
    parser.add_argument('--temperature_logits', type=float, default=1.0)
    parser.add_argument('--temperature_validity', type=float, default=1.0)
    parser.add_argument('--gammas', type=float, nargs='+', default=[1.0, 1.0, 1.0])
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()


    run_cfgs = vars(args)
    cfgs = config.Configurations(args.config_path)
    cfgs.update_cfgs(run_cfgs, super="RUN")

    if cfgs.RUN.seed == -1:
        cfgs.RUN.seed = random.randint(1,4096)
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True

    fix_all_seed(cfgs.RUN.seed)
    device = torch.cuda.current_device()

    Gen, Cls = load_gen_cls(cfgs.RUN, cfgs.DATA, cfgs.MODEL, cfgs.MODULES, device)
    sample(cfgs.RUN, cfgs.DATA, cfgs.MODEL, Gen, Cls, device)


if __name__ == '__main__':
    main()
