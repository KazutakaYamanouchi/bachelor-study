# 標準モジュール
import argparse
import csv
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random
import sys
from time import perf_counter

# 追加モジュール
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# 訓練に関する引数
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=0, metavar='B'
)
parser.add_argument(
    '-e', '--num-epochs', help='学習エポック数を指定します。',
    type=int, default=0, metavar='E'
)
parser.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='mnist',
    choices=['mnist', 'fashion_mnist', 'cifar10', 'stl10', 'imagenet2012']
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default='~/.datasets/vision'
)

parser.add_argument(
    '--nz', help='潜在空間の次元を指定します。',
    tupe=int, default=512

)

parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)
# コマンドライン引数をパースする
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
LAUNCH_DATETIME = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# 名前を指定してロガーを取得する
logger = getLogger('main')

batch_size = args.batch_size
num_epochs = args.num_epochs
lr_scale = args.lr_scale
nc = args.dataset
nz = args.nz
lr_g = 0.001
ngf = 64
lr_d = 0.001
ndf = 64

# 出力に関する定数
if args.dir_name is None:
    OUTPUT_DIR = Path(
        LAUNCH_DATETIME.strftime(
            f'./outputs/{args.dataset}/%Y%m%d%H%M%S'))
else:
    OUTPUT_DIR = Path(f'./outputs/{args.dataset}/{args.dir_name}')
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
f_outputs = open(
    OUTPUT_DIR.joinpath('outputs.txt'), mode='w', encoding='utf-8')
f_outputs.write(' '.join(sys.argv) + '\n')
OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
OUTPUT_SAMPLE_DIR.mkdir(parents=True)
logger.info(f'画像用のディレクトリ({OUTPUT_SAMPLE_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

# 乱数生成器のシード値の設定
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# TODO: 完成したらコメントを外す
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
logger.info('乱数生成器のシード値を設定しました。')

device = 'cuda'
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')

logger.info('画像に適用する変換のリストを定義します。')
data_transforms = []

to_tensor = transforms.ToTensor()
data_transforms.append(to_tensor)

normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
data_transforms.append(normalize)
logger.info('変換リストに正規化を追加しました。')

dataset = dset.STL10(
            root=args.data_path, split='train',
            transform=transforms.Compose(data_transforms), download=True)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
logger.info('データローダを生成しました。')


# =========================================================================== #
# モデルの定義
# =========================================================================== #
# TODO: モデルを定義する
model_g = None
model_d = None
# model_g = Generator(
#     nz=nz, nc=nc, ngf=ngf,
#     num_classes=num_classes
# ).to(device)
# model_d = Discriminator(
#     nz=nz, nc=nc, ndf=ndf,
#     num_classes=num_classes
# ).to(device)

# =========================================================================== #
# オプティマイザの定義
# =========================================================================== #
# TODO: オプティマイザを定義してモデルのパラメータを渡す
optim_g = None
optim_d = None
# optim_g = torch.optim.Adam(
#     model_g.parameters(),
#     lr=lr_g * lr_scale / batch_size,
#     betas=[0.5, 0.999])
# optim_d = torch.optim.Adam(
#     model_d.parameters(),
#     lr=lr_d * lr_scale / batch_size,
#     betas=[0.5, 0.999])

sample_z = torch.randn(64, nz, device=device)

f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
csv_writer = csv.writer(f_results, lineterminator='\n')
result_items = [
    'Epoch',
    'Generator Loss Mean', 'Discriminator Loss Mean',
    'Train Elapsed Time'
]
csv_writer.writerow(result_items)
csv_idx = {item: i for i, item in enumerate(result_items)}


# =========================================================================== #
# 訓練
# =========================================================================== #
fig, ax = plt.subplots(1, 1)
for epoch in range(num_epochs):
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    log_loss_g, log_loss_d = [], []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f'[{epoch+1}/{num_epochs}] 訓練開始',
        total=len(dataset)//batch_size,
        leave=False)
    model_g.train()  # Generatorを訓練モードに切り替える
    model_d.train()  # Discriminatorを訓練モードに切り替える
    begin_time = perf_counter()  # 時間計測開始
    for i, (real_images, _) in pbar:
        real_images = real_images.to(device)

        z = torch.randn(batch_size, nz, device=device)
        fake_images = model_g(z)

        #######################################################################
        # Discriminatorの訓練
        #######################################################################
        model_d.zero_grad()

        # Real画像についてDを訓練
        pred_d_real = model_d(real_images)
        # TODO 損失の計算式
        loss_d_real = F.relu(1 - pred_d_real).mean()
        loss_d_real.backward()

        # Fake画像についてDを訓練
        pred_d_fake = model_d(fake_images.detach())
        loss_d_fake = F.relu(pred_d_fake).mean()
        loss_d_fake.backward()

        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())
        optim_d.step()

        #######################################################################
        # Generatorの訓練
        #######################################################################
        model_g.zero_grad()
        pred_g = model_d(fake_images)
        loss_g = -pred_g.mean()
        loss_g.backward()
        log_loss_g.append(loss_g.item())
        optim_g.step()

        # プログレスバーの情報を更新
        pbar.set_description_str(
            f'[{epoch+1}/{num_epochs}] 訓練中... '
            f'<損失: (G={loss_g.item():.016f}, D={loss_d.item():.016f})>')
    end_time = perf_counter()  # 時間計測終了
    pbar.close()

    loss_g_mean = np.mean(log_loss_g)
    loss_d_mean = np.mean(log_loss_d)
    results[csv_idx['Generator Loss Mean']] = f'{loss_g_mean:.016f}'
    results[csv_idx['Discriminator Loss Mean']] = f'{loss_d_mean:.016f}'

    train_elapsed_time = end_time - begin_time
    results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    print(
        f'[{epoch+1}/{num_epochs}] 訓練完了. '
        f'<エポック処理時間: {train_elapsed_time:.07f}[s/epoch]'
        f', 平均損失: (G={loss_g_mean:.016f}, D={loss_d_mean:.016f})>')

    model_g.eval()
    model_d.eval()

    if (
        epoch == 0
        or (epoch + 1) % args.sample_interval == 0
        or epoch == num_epochs - 1
    ):
        sample_dir = OUTPUT_SAMPLE_DIR.joinpath(f'{epoch + 1}')
        sample_dir.mkdir()
        for i in range(num_classes):
            with torch.no_grad():
                sample_images = model_g(sample_z).cpu()
                vutils.save_image(
                    sample_images,
                    sample_dir.joinpath(f'{i}_{class_names[i]}.png'),
                    nrow=int(np.sqrt(args.num_samples)),
                    range=(-1.0, 1.0))

    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch == num_epochs - 1):
        model_g_fname = f'generator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
        model_d_fname = f'discriminator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
    csv_writer.writerow(results)
    f_results.flush()
f_results.close()
f_outputs.close()
