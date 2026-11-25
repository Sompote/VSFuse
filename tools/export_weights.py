"""Export a trained estimator (model.pt) to the flat binary the Rust app reads.
usage: python3 export_weights.py model_dir out.vsf"""
import sys, json, struct, array, torch
ck = torch.load(f"{sys.argv[1]}/model.pt", map_location="cpu", weights_only=False)
meta = dict(G=int(ck["G"]), K=int(ck["K"]), nin=int(ck["nin"]), nout=int(ck["nout"]), vmu=float(ck["vmu"]), vsd=float(ck["vsd"]),
            lam=[float(x) for x in ck["lam"]], thick=[float(x) for x in ck["thick"]], kz=[float(x) for x in ck["kz"]])
with open(sys.argv[2], "wb") as f:
    mj = json.dumps(meta).encode(); f.write(b"VSF1"); f.write(struct.pack("<I", len(mj))); f.write(mj)
    f.write(struct.pack("<I", len(ck["state"])))
    for k, v in ck["state"].items():
        kb = k.encode(); f.write(struct.pack("<I", len(kb))); f.write(kb)
        f.write(struct.pack("<I", v.dim()));
        for d in v.shape: f.write(struct.pack("<I", d))
        array.array("f", v.flatten().tolist()).tofile(f)
        print(k, tuple(v.shape))
