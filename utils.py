from tqdm import tqdm

def optional_tqdm(iter, args):
    return tqdm(iter) if args.tqdm else iter
