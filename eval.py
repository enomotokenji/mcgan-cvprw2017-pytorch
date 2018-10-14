import numpy as np
from skimage.measure import compare_ssim as SSIM

from torch.autograd import Variable

from utils import save_image


def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    for i, batch in enumerate(test_data_loader):
        x, t = Variable(batch[0]), Variable(batch[1])
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        out = gen(x)

        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 4
            c = 3
            p = config.size

            allim = np.zeros((h, w, c, p, p))
            x_ = x.cpu().numpy()[0]
            rgb = x_[:3]
            nir = x_[3]
            allim[0, 0, :] = np.repeat(nir[None, :, :], repeats=3, axis=0) * 127.5 + 127.5
            allim[0, 1, :] = rgb * 127.5 + 127.5
            allim[0, 2, :] = t.cpu().numpy()[0] * 127.5 + 127.5
            allim[0, 3, :] = np.clip(out.cpu().numpy()[0], -1, 1) * 127.5 + 127.5
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*p, c))

            save_image(config, allim, i, epoch)

        mse = criterionMSE(out, t)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = np.tensordot(out.cpu().numpy()[0].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(t.cpu().numpy()[0].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        
        ssim = SSIM(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim
    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test
