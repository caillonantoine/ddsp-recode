import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange


class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    EPOCHS = 1000000
    BATCH = 16
    LR = 1e-3


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DDSP(**config["model"]).to(device)

dataset = Dataset(config["preprocess"]["out_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.LR)
step = 0
for e in tqdm(range(args.EPOCHS)):
    for s, p, l in dataloader:
        s = s.to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)

        l = (l - mean_loudness) / std_loudness

        y = model(p, l).squeeze(-1)

        ori_stft = multiscale_fft(
            s,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)
        step += 1

    if not e % 100:
        torch.save(
            model.state_dict(),
            path.join(args.ROOT, args.NAME, "state.pth"),
        )

        audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

        sf.write(
            path.join(args.ROOT, args.NAME, f"eval_{e:06d}.wav"),
            audio,
            config["preprocess"]["sampling_rate"],
        )
