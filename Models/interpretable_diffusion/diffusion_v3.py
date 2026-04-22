import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.TransGCN_with_features import TransformerWithGCN as Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TSGCN(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            external_feat_dim=2,  # 外部特征的维度
            **kwargs
    ):
        super(Diffusion_TSGCN, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        # 修改 n_feat 的设置
        self.model = Transformer(
            n_feat1=feature_size,  # 每个节点的总特征数
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs
        )

        # 外部特征处理层：将外部特征映射到与时序特征相同的嵌入维度
        self.external_feat_fc = nn.Linear(external_feat_dim, d_model)
        self.output_fc = nn.Linear(48, 282)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 初始化Beta和其他相关的参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def forward(self, x, external_feats, **kwargs):
        adj = kwargs.pop('adj', None)
        adj_new = kwargs.pop('adj_new', None)
        b, t, n, device = *x.shape, x.device  # b: batch size, t: timesteps, n: num nodes

        t = kwargs.pop('t', None)
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 处理 x1 的维度，确保其形状在预测时一致
        x1 = x
        # print(x1.shape)
        '''if x1.dim() == 1:
            x1 = x1.unsqueeze(0)  # 调整 x1 为 2D 或 3D 形状
        if x1.size(1) != self.seq_length or x1.size(2) != n:
            x1 = x1.view(b, self.seq_length, n)  # 根据需要调整 x1 形状

        # 确保 x 和 external_feats 具有相同的维度
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # 将 x 扩展为 4D (batch_size, timesteps, num_nodes, 1)'''
        x = external_feats
        '''# 扩展 external_feats 到与 x 匹配的维度
        external_feats = external_feats.unsqueeze(2).expand(-1, -1, n,
                                                            -1)  # (batch_size, timesteps, num_nodes, num_external_features)

        # 在特征维度拼接 (station features + external features)
        x = torch.cat([x, external_feats], dim=-1)  # (batch_size, timesteps, num_nodes, 1 + num_external_features)

        # 调整形状以适应 conv1d 的输入格式 (batch_size, num_nodes, timesteps, features)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_nodes, timesteps, features)

        # 合并特征维度，将其变为 (batch_size, num_nodes, timesteps * (1 + num_external_features))
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, timesteps * features)
        x = x.permute(0, 2, 1)'''

        # 调用 _train_loss 方法
        return self._train_loss(x_start=x, x_original=x1, t=t, adj=adj, adj_new=adj_new, **kwargs)

    def predict_noise_from_start(self, x_t, x_t1, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t1.shape) * x_t1 - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t1.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, x1, t, adj, adj_new, padding_masks=None):
        # print(x)
        trend, season, contrastive_loss = self.model(x, x1, t=t, adj=adj, adj_new=adj_new, padding_masks=padding_masks)
        model_output = trend + season
        self.contrastive_loss_value = contrastive_loss
        return model_output

    def model_predictions(self, x, x1, t, adj, adj_new, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        # print('x:', x.shape)
        # print('t:', t.shape)
        x_start = self.output(x, x1, t, adj, adj_new, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, x1, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        img1 = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, img1, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, x_original, t, adj, adj_new, target=None, noise=None, padding_masks=None):
        noise1 = default(noise, lambda: torch.randn_like(x_original))
        #noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_original.permute(0, 2, 1)

        # print(x_original.shape)
        # print(x_start.shape)
        
        x1 = self.q_sample(x_start=x_original, t=t, noise=noise1)
        #x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        x = x_start

        '''if adj is not None:
            print('adj:', adj.shape)
            x = torch.matmul(adj, x) #'''

        model_out = self.output(x, x1, t, adj, adj_new, padding_masks)  # Ensure t is passed correctly
        # print('model_out:', model_out.shape)
        # model_out = self.output_fc(model_out)
        train_loss = self.loss_fn(model_out, target, reduction='none')
        contrastive_loss = self.contrastive_loss_value

        # train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        contrastive_loss_weight = 0.1  # Set the weight for contrastive loss (can be adjusted)
        # Combine the losses (primary loss + weighted contrastive loss)
        total_loss = train_loss + contrastive_loss_weight * contrastive_loss
        return train_loss.mean()

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, external_feats, adj, adj_new, sampling_timesteps, partial_mask=None,
                           clip_denoised=True,
                           model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img1 = torch.randn(shape, device=device)
        img = external_feats.to(device)
        if adj is not None:
            adj = adj.to(device)
        if adj_new is not None:
            adj_new = adj_new.to(device)
        target = target.to(device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, img1, time_cond, adj, adj_new,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img1 = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img1)

            img1 = pred_mean + sigma * noise
            img1 = self.langevin_fn(sample=img, sample1=img1, mean=pred_mean, sigma=sigma, t=time_cond, adj=adj,
                                   adj_new=adj_new,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img1[partial_mask] = target_t[partial_mask]

        img1[partial_mask] = target[partial_mask]

        return img1

    def sample_infill(
            self,
            shape,
            target,
            external_feats,
            partial_mask=None,
            clip_denoised=True,
            model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img1 = torch.randn(shape, device=device)
        img = external_feats

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, x1=img1, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)

        img[partial_mask] = target[partial_mask]
        return img

    def p_sample_infill(
            self,
            x,
            x1,
            target,
            t: int,
            partial_mask=None,
            clip_denoised=True,
            model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
            self,
            coef,
            partial_mask,
            tgt_embs,
            learning_rate,
            sample,
            sample1,
            mean,
            sigma,
            adj,
            adj_new,
            t,
            coef_=0.
    ):

        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)
        input_embs_param1 = torch.nn.Parameter(sample1)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param1], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, x1=input_embs_param1, t=t, adj=adj, adj_new=adj_new)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param1) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param1) ** 2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param1.data)
                input_embs_param1 = torch.nn.Parameter(
                    (input_embs_param1.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample1[~partial_mask] = input_embs_param1.data[~partial_mask]
        return sample1


if __name__ == '__main__':
    pass
