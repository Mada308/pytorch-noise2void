from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv
import time
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def infer_full_image(net, x):
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        x_pad = x
    out = net(x_pad)
    return out[:, :, :h, :w]


def save_preview_png(noisy_np, pred_np, gt_np, wb, save_path):
    img_noisy = raw4_to_srgb(noisy_np, wb)
    img_pred = raw4_to_srgb(pred_np, wb)
    img_gt = raw4_to_srgb(gt_np, wb)
    combined = np.concatenate([img_noisy, img_pred, img_gt], axis=1)
    pil_img = Image.fromarray(combined)
    w, h = pil_img.size
    pil_img = pil_img.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
    pil_img.save(save_path)

##
class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.norm = args.norm

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.dir_list = args.dir_list
        self.network = args.network
        self.max_train_images = args.max_train_images
        self.val_max_images = args.val_max_images
        self.num_workers = args.num_workers

        self.gpu_ids = args.gpu_ids

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")
            self.gpu_ids = []

    def save(self, dir_chck, netG, optimG, epoch, best_psnr=-1.0, is_best=False):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        state = {
            'netG': netG.state_dict(),
            'optimG': optimG.state_dict(),
            'epoch': epoch,
            'best_psnr': best_psnr
        }
        torch.save(state, '%s/model_epoch%04d.pth' % (dir_chck, epoch))
        torch.save(state, '%s/model_latest.pth' % dir_chck)
        if is_best:
            torch.save(state, '%s/model_best.pth' % dir_chck)

    def load(self, dir_chck, netG, optimG=None, epoch=None, mode='train'):
        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            if mode == 'train':
                return netG, optimG, 0, -1.0
            return netG, 0

        if mode == 'test' and not epoch and os.path.exists(os.path.join(dir_chck, 'model_best.pth')):
            ckpt_path = os.path.join(dir_chck, 'model_best.pth')
        elif not epoch and os.path.exists(os.path.join(dir_chck, 'model_latest.pth')):
            ckpt_path = os.path.join(dir_chck, 'model_latest.pth')
        else:
            if not epoch:
                ckpt = [f for f in os.listdir(dir_chck) if f.startswith('model_epoch') and f.endswith('.pth')]
                if not ckpt:
                    if mode == 'train':
                        return netG, optimG, 0, -1.0
                    return netG, 0
                ckpt.sort()
                epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])
            ckpt_path = '%s/model_epoch%04d.pth' % (dir_chck, epoch)

        dict_net = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        loaded_epoch = dict_net.get('epoch', epoch if epoch else 0)
        best_psnr = dict_net.get('best_psnr', -1.0)

        print('Loaded checkpoint: %s (epoch %d)' % (ckpt_path, loaded_epoch))

        netG.load_state_dict(dict_net['netG'])
        if mode == 'train':
            if optimG is not None and 'optimG' in dict_net:
                optimG.load_state_dict(dict_net['optimG'])
            return netG, optimG, loaded_epoch, best_psnr
        return netG, loaded_epoch

    def train(self):
        mode = self.mode
        train_continue = self.train_continue
        num_epoch = self.num_epoch
        lr_G = self.lr_G
        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker
        size_window = (5, 5)

        norm = self.norm
        name_data = self.name_data
        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup directories
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)
        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')
        dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val')
        preview_dir = os.path.join(dir_result_val, 'previews')

        os.makedirs(dir_chck, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)

        ## setup dataset RAW Sony SID
        train_list = os.path.join(self.dir_list, 'Sony_train_list.txt')
        val_list = os.path.join(self.dir_list, 'Sony_val_list.txt')

        transform_train = transforms.Compose([RandomFlip(), ToTensor()])
        transform_val = transforms.Compose([ToTensor()])

        dataset_train = Dataset(
            self.dir_data, train_list, mode='train', transform=transform_train,
            patch_size=self.ny_load, max_images=self.max_train_images,
            ratio=0.985, size_window=size_window
        )
        dataset_val = Dataset(
            self.dir_data, val_list, mode='val', transform=transform_val,
            max_images=self.val_max_images
        )

        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=(device.type == 'cuda'),
            worker_init_fn=seed_worker
        )
        loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

        num_batch_train = len(loader_train)
        print(f'Device: {device} | Network: {self.network} | Train: {len(dataset_train)} | Val: {len(dataset_val)}')

        ## setup network
        if self.network == 'resnet':
            netG = ResNet(nch_in, nch_out, nch_ker, norm)
        else:
            netG = UNet(nch_in, nch_out, nch_ker, norm)

        netG = init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        netG.to(device)

        ## setup optimization & mixed precision
        optimG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(self.beta1, 0.999))
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        ## load from checkpoints
        st_epoch = 0
        best_psnr = -1.0
        if train_continue == 'on':
            netG, optimG, st_epoch, best_psnr = self.load(dir_chck, netG, optimG, mode=mode)

        ## setup tensorboard & csv log
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        csv_path = os.path.join(self.dir_result, self.scope, name_data, 'metrics_log.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'epoch', 'train_loss', 'val_psnr', 'val_ssim',
                    'noisy_psnr', 'noisy_ssim', 'sec'
                ])

        for epoch in range(st_epoch + 1, num_epoch + 1):
            t0 = time.time()

            ## training phase
            netG.train()
            loss_G_train = []

            for batch, data in enumerate(loader_train, 1):
                label = data['label'].to(device, non_blocking=True)
                inp = data['input'].to(device, non_blocking=True)
                mask = data['mask'].to(device, non_blocking=True)

                optimG.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    output = netG(inp)
                    blind = 1.0 - mask
                    loss_G = (((output - label) ** 2) * blind).sum() / (blind.sum() + 1e-8)

                scaler.scale(loss_G).backward()
                scaler.step(optimG)
                scaler.update()

                loss_G_train.append(loss_G.item())

                if batch % 20 == 0 or batch == num_batch_train:
                    print('TRAIN: EPOCH %04d/%04d: BATCH %04d/%04d: LOSS: %.6f'
                          % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_train)))

            avg_train_loss = float(np.mean(loss_G_train))
            writer_train.add_scalar('loss_G', avg_train_loss, epoch)

            ## validation phase
            val_psnr_str = val_ssim_str = noisy_psnr_str = noisy_ssim_str = ''
            is_best = False

            if (epoch % num_freq_disp == 0) or (epoch == num_epoch):
                netG.eval()
                psnr_list, ssim_list = [], []
                noisy_psnr_list, noisy_ssim_list = [], []

                with torch.no_grad():
                    for i, data in enumerate(loader_val):
                        inp = data['input'].to(device)
                        gt_np = data['label'][0].numpy()
                        wb_np = data['wb'][0].numpy()
                        fname = data['name'][0]

                        output = infer_full_image(netG, inp)
                        output = torch.clamp(output, 0.0, 1.0)

                        pred_np = output[0].cpu().numpy()
                        noisy_np = np.clip(data['input'][0].numpy(), 0.0, 1.0)

                        psnr_list.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
                        ssim_list.append(structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=0))
                        noisy_psnr_list.append(peak_signal_noise_ratio(gt_np, noisy_np, data_range=1.0))
                        noisy_ssim_list.append(structural_similarity(gt_np, noisy_np, data_range=1.0, channel_axis=0))

                        if i < 3:
                            clean_name = fname.replace('.ARW', '')
                            out_path = os.path.join(preview_dir, f'ep{epoch:04d}_{i:02d}_{clean_name}.png')
                            save_preview_png(noisy_np, pred_np, gt_np, wb_np, out_path)

                val_psnr = float(np.mean(psnr_list))
                val_ssim = float(np.mean(ssim_list))
                noisy_psnr = float(np.mean(noisy_psnr_list))
                noisy_ssim = float(np.mean(noisy_ssim_list))

                writer_val.add_scalar('psnr', val_psnr, epoch)
                writer_val.add_scalar('ssim', val_ssim, epoch)

                val_psnr_str = f'{val_psnr:.4f}'
                val_ssim_str = f'{val_ssim:.6f}'
                noisy_psnr_str = f'{noisy_psnr:.4f}'
                noisy_ssim_str = f'{noisy_ssim:.6f}'

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    is_best = True

                print(f'VALID: EPOCH {epoch:04d} | PSNR: {val_psnr:.2f} dB (Noisy: {noisy_psnr:.2f}, '
                      f'+{val_psnr - noisy_psnr:.2f} dB) | SSIM: {val_ssim:.4f} (Noisy: {noisy_ssim:.4f}) '
                      f'| Best: {best_psnr:.2f} dB')

            elapsed = time.time() - t0
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, f'{avg_train_loss:.6f}', val_psnr_str, val_ssim_str,
                    noisy_psnr_str, noisy_ssim_str, f'{elapsed:.1f}'
                ])

            ## save
            if (epoch % num_freq_save == 0) or is_best or (epoch == num_epoch):
                self.save(dir_chck, netG, optimG, epoch, best_psnr=best_psnr, is_best=is_best)
            else:
                torch.save({
                    'netG': netG.state_dict(),
                    'optimG': optimG.state_dict(),
                    'epoch': epoch,
                    'best_psnr': best_psnr
                }, os.path.join(dir_chck, 'model_latest.pth'))

        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker
        norm = self.norm
        name_data = self.name_data

        ## setup directories & dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)
        dir_result_test = os.path.join(self.dir_result, self.scope, name_data, 'test')
        preview_dir = os.path.join(dir_result_test, 'previews')
        os.makedirs(preview_dir, exist_ok=True)

        test_list = os.path.join(self.dir_list, 'Sony_test_list.txt')
        transform_test = transforms.Compose([ToTensor()])
        dataset_test = Dataset(self.dir_data, test_list, mode='test', transform=transform_test)
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

        ## setup network
        if self.network == 'resnet':
            netG = ResNet(nch_in, nch_out, nch_ker, norm)
        else:
            netG = UNet(nch_in, nch_out, nch_ker, norm)

        netG = init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        netG.to(device)

        ## load best checkpoint
        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        netG.eval()
        psnr_list, ssim_list = [], []
        noisy_psnr_list, noisy_ssim_list = [], []

        print(f'TEST: Valutazione su {len(dataset_test)} immagini (checkpoint epoca {st_epoch})...')
        with torch.no_grad():
            for i, data in enumerate(loader_test):
                inp = data['input'].to(device)
                gt_np = data['label'][0].numpy()
                wb_np = data['wb'][0].numpy()
                fname = data['name'][0]

                output = infer_full_image(netG, inp)
                output = torch.clamp(output, 0.0, 1.0)

                pred_np = output[0].cpu().numpy()
                noisy_np = np.clip(data['input'][0].numpy(), 0.0, 1.0)

                psnr_list.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
                ssim_list.append(structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=0))
                noisy_psnr_list.append(peak_signal_noise_ratio(gt_np, noisy_np, data_range=1.0))
                noisy_ssim_list.append(structural_similarity(gt_np, noisy_np, data_range=1.0, channel_axis=0))

                if i < 10:
                    clean_name = fname.replace('.ARW', '')
                    out_path = os.path.join(preview_dir, f'TEST_{i:02d}_{clean_name}.png')
                    save_preview_png(noisy_np, pred_np, gt_np, wb_np, out_path)

                if (i + 1) % 50 == 0 or (i + 1) == len(dataset_test):
                    print(f'TEST: {i + 1}/{len(dataset_test)} | PSNR parziale: {np.mean(psnr_list):.2f} dB')

        test_psnr = float(np.mean(psnr_list))
        test_ssim = float(np.mean(ssim_list))
        noisy_psnr = float(np.mean(noisy_psnr_list))
        noisy_ssim = float(np.mean(noisy_ssim_list))

        print(f'TEST BASELINE NOISY -> PSNR: {noisy_psnr:.2f} dB | SSIM: {noisy_ssim:.4f}')
        print(f'TEST DENOISED N2V   -> PSNR: {test_psnr:.2f} dB (+{test_psnr - noisy_psnr:.2f} dB) '
              f'| SSIM: {test_ssim:.4f} (+{test_ssim - noisy_ssim:.4f})')

        with open(os.path.join(dir_result_test, 'test_results.txt'), 'w') as f:
            f.write(f'Checkpoint epoch: {st_epoch}\n')
            f.write(f'Network: {self.network}\n')
            f.write(f'Test images: {len(dataset_test)}\n')
            f.write(f'Noisy Baseline PSNR: {noisy_psnr:.4f} dB | SSIM: {noisy_ssim:.6f}\n')
            f.write(f'Denoised N2V   PSNR: {test_psnr:.4f} dB | SSIM: {test_ssim:.6f}\n')
            f.write(f'Gain           PSNR: +{test_psnr - noisy_psnr:.4f} dB | SSIM: +{test_ssim - noisy_ssim:.6f}\n')


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
