import argparse
from src.DistDataModel import DistDataModel
from src.Compressor import *

compressor_map = {'none': NoneCompressor(),
				   'topk30': TopKCompressor(0.3),
				   'topk40': TopKCompressor(0.4),
				   'topk50': TopKCompressor(0.5),
                   'topk55': TopKCompressor(0.55),
				   'topk60': TopKCompressor(0.6),
				   'qsgd': QSGDCompressor(2)}


def parse_args():
    p = argparse.ArgumentParser(description="Run a distributed-data experiment")
    p.add_argument("--dataset",     type=str,   required=True,    choices=["FashionMNIST","CIFAR10","Shakespeare"])
    p.add_argument("--compress",    type=str,   default="none",   choices=compressor_map)
    p.add_argument("--optimizer",   type=str,   required=True,    choices=["DADAM","DAMSCo","DaSHCo","CDProxSGD","DAdaGrad","ESMuon","NSMuon","ESMuonpnorm","ESMuonqnorm"])
    p.add_argument("--comm-set",    nargs="+",  default=['x'],    help="communication variables")
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--lr-decay",    type=str,   default="none",   choices=["none","cosine"])
    p.add_argument("--variety",     type=str,   default="index")
    p.add_argument("--topology",    type=str,   default="ring")
    p.add_argument("--model",       type=str,   default="LeNet5", choices=["LeNet5","fixup_resnet20","nanoGPT"])
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--rank",        type=int,   default=4)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--resume",      action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    # If using top-k, change comm_set to bar-variants
    if args.compress.startswith("topk"):
        args.comm_set = [c + "_bar" for c in args.comm_set]

    model = DistDataModel(
        model=args.model,
        dataset=args.dataset,
        topology=args.topology,
        optimizer=args.optimizer,
        comm_set=args.comm_set,
        batch_size=args.batch_size,
        device=args.device,
        track=True,
        seed=1337,
        compressor=compressor_map[args.compress],
        lr_decay=args.lr_decay,
        variety=args.variety,
        lr=args.lr,
        resume= args.resume
    )

    model.epochs = args.epochs
    name = (f"{args.dataset}-{args.optimizer}-"
            f"{args.compress}-Rank-{args.rank}-"
            f"{args.variety}-lrtype-{args.lr_decay}-"
            f"{args.epochs}")

    print(f"[INFO] {name} -> Initializing model…", flush=True)
    model.train(verbose=True, output_file=name)
    print("[INFO] Training finished.", flush=True)

if __name__ == "__main__":
    main()
