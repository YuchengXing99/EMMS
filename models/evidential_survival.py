import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from pycox.evaluation import EvalSurv
CORR_TARGETS = ['RNA', 'WSI']
L2_ONLY_TARGETS = []
L2_TARGETS = CORR_TARGETS + L2_ONLY_TARGETS


def l2_normalize(X, eps=1e-8):
    if isinstance(X, torch.Tensor):
        norm = torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True)) + eps
        return X / norm
    else:
        norm = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + eps
        return X / norm


def correlation_normalize(X, eps=1e-8):
    if isinstance(X, torch.Tensor):
        # Step 1: mean-centering (per sample)
        X_centered = X - X.mean(dim=1, keepdim=True)
        # Step 2: L2 normalize
        norm = torch.sqrt(torch.sum(X_centered ** 2, dim=1, keepdim=True)) + eps
        return X_centered / norm
    else:
        # Step 1: mean-centering (per sample)
        X_centered = X - X.mean(axis=1, keepdims=True)
        # Step 2: L2 normalize
        norm = np.sqrt(np.sum(X_centered ** 2, axis=1, keepdims=True)) + eps
        return X_centered / norm


def ENNreg_init(X_dict, y, K_dict, mask_dict=None, nstart=100, c=1, rna_gamma_scale=0.3):
    if isinstance(K_dict, dict):
        K_dict = K_dict.copy()

    prototypes = []
    for key in X_dict:
        X = X_dict[key]
        p = X.shape[1]
        K = K_dict[key]

        # Whether this modality needs L2 normalization
        use_l2 = (key in L2_TARGETS)

        # === 1. Mask handling ===
        if mask_dict is not None and key in mask_dict:
            mask = mask_dict[key]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            valid_idx = np.where(mask == 1)[0]

            # All samples missing: use dummy prototypes (never actually used)
            if len(valid_idx) == 0:
                print(f"  WARNING: All samples missing for modality '{key}', using dummy prototypes")
                Beta = torch.zeros(K, p, dtype=torch.float64)
                alpha = torch.zeros(K, dtype=torch.float64)
                sig = torch.ones(K, dtype=torch.float64)
                W = torch.zeros(K, p, dtype=torch.float64)
                gam = torch.ones(K, dtype=torch.float64)
                eta = torch.ones(K, dtype=torch.float64) * 2
                init = {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}
                prototypes.append(init)
                continue

            if isinstance(X, torch.Tensor):
                X_valid = X[valid_idx].cpu().numpy()
                y_valid = y[valid_idx] if isinstance(y, torch.Tensor) else y[valid_idx]
            else:
                X_valid = X[valid_idx]
                y_valid = y[valid_idx]
        else:
            X_valid = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
            y_valid = y
            valid_idx = np.arange(len(y))

        # === 2. Ensure enough samples ===
        if len(valid_idx) < K:
            raise ValueError(
                f"Modality '{key}' has only {len(valid_idx)} valid samples but K={K}."
            )

        # === 3. Conditional normalization ===
        if key in CORR_TARGETS:
            # RNA: Correlation distance (mean-centering + L2)
            X_proc = correlation_normalize(X_valid)
            print(f"  INFO: [Corr] Applied correlation normalization to '{key}'")
        elif key in L2_ONLY_TARGETS:
            # L2 normalize only
            X_proc = l2_normalize(X_valid)
            print(f"  INFO: [L2] Applied L2 normalization to '{key}'")
        else:
            X_proc = X_valid
            print(f"  INFO: [Std] Kept raw scale for '{key}'")

        # === 4. Check number of unique data points ===
        n_unique = len(np.unique(X_proc, axis=0))
        if n_unique < K:
            raise ValueError(
                f"Modality '{key}' has only {n_unique} unique points but K={K}."
            )

        K_dict[key] = K

        # === 5. KMeans clustering ===
        clus = KMeans(n_clusters=K, max_iter=5000, n_init=nstart, random_state=0).fit(X_proc)

        Beta = torch.zeros(K, p, dtype=torch.float64)
        alpha = torch.zeros(K, dtype=torch.float64)
        sig = torch.ones(K, dtype=torch.float64)

        # === 6. Center handling (conditional) ===
        if key in CORR_TARGETS:
            # Correlation-distance mode: centers also get mean-centering + L2
            centers = correlation_normalize(clus.cluster_centers_)
            assigned_centers = centers[clus.labels_]
            global_inertia = np.sum((X_proc - assigned_centers) ** 2)
            W = torch.tensor(centers, dtype=torch.float64)
        elif key in L2_ONLY_TARGETS:
            # L2 mode: centers only get L2 normalization
            centers = l2_normalize(clus.cluster_centers_)
            assigned_centers = centers[clus.labels_]
            global_inertia = np.sum((X_proc - assigned_centers) ** 2)
            W = torch.tensor(centers, dtype=torch.float64)
        else:
            # Plain mode: use KMeans centers directly
            W = torch.tensor(clus.cluster_centers_, dtype=torch.float64)

        gam = torch.ones(K, dtype=torch.float64)

        for k in range(K):
            if use_l2:
                # L2 mode: use numpy indexing
                mask_k = clus.labels_ == k
                ii_np = np.where(mask_k)[0]
                nk = len(ii_np)

                if isinstance(y_valid, torch.Tensor):
                    alpha[k] = torch.mean(y_valid[ii_np])
                else:
                    alpha[k] = torch.mean(torch.tensor(y_valid[ii_np], dtype=torch.float64))

                if nk > 1:
                    gam[k] = 1 / np.sqrt(global_inertia / nk + 1e-8)
                    if isinstance(y_valid, torch.Tensor):
                        sig[k] = torch.std(y_valid[ii_np])
                    else:
                        sig[k] = torch.std(torch.tensor(y_valid[ii_np], dtype=torch.float64))
            else:
                mask_k = torch.eq(torch.tensor(clus.labels_), k)
                ii = torch.nonzero(mask_k, as_tuple=True)[0]
                nk = len(ii)

                if isinstance(y_valid, torch.Tensor):
                    alpha[k] = torch.mean(y_valid[ii])
                else:
                    alpha[k] = torch.mean(torch.tensor(y_valid[ii], dtype=torch.float64))

                if nk > 1:
                    gam[k] = 1 / torch.sqrt(torch.tensor(clus.inertia_) / nk)
                    if isinstance(y_valid, torch.Tensor):
                        sig[k] = torch.std(y_valid[ii])
                    else:
                        sig[k] = torch.std(torch.tensor(y_valid[ii], dtype=torch.float64))

        # === 7. Gamma scaling ===
        if use_l2:
            gam *= c * rna_gamma_scale  # L2 space has a small distance range (0~4), so shrink gamma
            print(f"  INFO: [L2] Gamma scaled by c*{rna_gamma_scale}={c*rna_gamma_scale:.2f} for '{key}', range: [{gam.min():.4f}, {gam.max():.4f}]")
        else:
            gam *= c  # Plain modality keeps the original logic
            print(f"  INFO: [Std] Gamma scaled by c={c:.2f} for '{key}', range: [{gam.min():.4f}, {gam.max():.4f}]")

        eta = torch.ones(K) * 2
        init = {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}
        prototypes.append(init)

    return prototypes, K_dict


class ENN_survival_prediction(nn.Module):
    def __init__(self, X_dict, K_dict, prototypes, trainable_lambda=True,
                 discount_init=None):
        super(ENN_survival_prediction, self).__init__()
        self.X_dict = X_dict
        self.K_dict = K_dict
        self.prototypes = prototypes

        # Keep modality order for discount initialization
        self.modality_order = list(X_dict.keys())

        self.input_dim_list = [X.shape[1] for X in X_dict.values()]
        # Must follow X_dict key order to fetch K, otherwise an order mismatch causes IndexError
        self.prototype_dim_list = [K_dict[key] for key in X_dict.keys()]
        self.num_sources = len(self.input_dim_list)

        self.alphas = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.betas = nn.ParameterList([Parameter(torch.Tensor(k, p)) for k, p in zip(self.prototype_dim_list, self.input_dim_list)])
        self.sigs = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.etas = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.gammas = nn.ParameterList([Parameter(torch.Tensor(k, 1)) for k in self.prototype_dim_list])
        self.ws = nn.ParameterList([Parameter(torch.Tensor(k, p)) for k, p in zip(self.prototype_dim_list, self.input_dim_list)])

        # Learnable per-dimension scaling (diagonal Mahalanobis)
        self.dim_scales = nn.ParameterList([
            Parameter(torch.ones(input_dim, dtype=torch.float64))
            for input_dim in self.input_dim_list
        ])
        print(f"  INFO: Added learnable dim_scales for {self.num_sources} modalities: {self.input_dim_list}")

        # log1p hx compression: discount starts at 0.0 (neutral) and is learned
        if discount_init is None:
            discount_init = {}
        default_discount = 0.0

        discount_values = []
        for key in self.modality_order:
            if key in discount_init:
                val = discount_init[key]
            else:
                val = default_discount
            discount_values.append(val)
            print(f"  INFO: Discount init for '{key}': {val:.1f} (sigmoid={torch.sigmoid(torch.tensor(val)).item():.4f})")

        self.discounts = nn.ParameterList([
            Parameter(torch.tensor([val])) for val in discount_values
        ])

        if trainable_lambda:
            self.z = nn.ParameterList([Parameter(torch.tensor([0.0])) for i in range(self.num_sources)])

        self.reset_parameters(prototypes)

    def reset_parameters(self, prototypes):
        for i, prototype in enumerate(prototypes):
            self.alphas[i] = Parameter(prototype['alpha'])
            self.betas[i] = Parameter(prototype['Beta'])
            self.sigs[i] = Parameter(prototype['sig'])
            self.etas[i] = Parameter(prototype['eta'])
            self.gammas[i] = Parameter(prototype['gam'])
            self.ws[i] = Parameter(prototype['W'])

    def forward(self, inputs, masks=None):
        assert isinstance(inputs, dict) and all(torch.is_tensor(input) for input in inputs.values())
        nt = next(iter(inputs.values())).size(0)

        mux_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        sig2x_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        hx_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        hx_dc = torch.zeros((self.num_sources, nt), dtype=torch.float64)

        penalty1 = torch.zeros((self.num_sources + 1), dtype=torch.float64)
        penalty2 = torch.zeros((self.num_sources + 1), dtype=torch.float64)

        eps = 1e-8
        modality_order = []

        for i, (key, input) in enumerate(inputs.items()):
            modality_order.append(key)
            h = self.etas[i] ** 2

            # Conditional normalization + diagonal scaling
            a = torch.zeros(nt, self.prototype_dim_list[i])

            # Step 1: normalize
            if key in CORR_TARGETS:
                input_proc = correlation_normalize(input, eps=eps)
            elif key in L2_ONLY_TARGETS:
                input_proc = l2_normalize(input, eps=eps)
            else:
                input_proc = input

            # Step 2: diagonal scaling
            scale = self.dim_scales[i]
            input_scaled = input_proc * scale

            for k in range(self.prototype_dim_list[i]):
                if key in CORR_TARGETS:
                    w_k = correlation_normalize(self.ws[i][k, :].unsqueeze(0), eps=eps)
                elif key in L2_ONLY_TARGETS:
                    w_k = l2_normalize(self.ws[i][k, :].unsqueeze(0), eps=eps)
                else:
                    w_k = self.ws[i][k, :].unsqueeze(0)
                w_scaled = w_k * scale
                a[:, k] = torch.exp(
                    -self.gammas[i][k] ** 2 * torch.sum(
                        (input_scaled - w_scaled.expand(nt, -1)) ** 2, dim=1
                    )
                )

            H = h.expand(nt, -1)
            hx = torch.sum(a * H, dim=1)
            hx_safe = hx + eps

            # Compute mu (using the processed input)
            if key in CORR_TARGETS or key in L2_ONLY_TARGETS:
                mu = torch.mm(input_proc, self.betas[i].T) + self.alphas[i].expand(nt, -1)
            else:
                mu = torch.mm(input, self.betas[i].T) + self.alphas[i].expand(nt, -1)
            mux = torch.sum(mu * a * H, dim=1) / hx_safe
            sig2x = torch.sum((self.sigs[i] ** 2).expand(nt, -1) * (a ** 2) * (H ** 2), dim=1) / (hx_safe ** 2)
            sig2x = torch.clamp(sig2x, min=eps)

            # Mask handling
            if masks is not None and key in masks:
                mask = masks[key].to(dtype=torch.float64)
            else:
                mask = torch.ones(nt, dtype=torch.float64)

            mux_total[i] = mux * mask
            sig2x_total[i] = sig2x * mask
            hx_total[i] = hx * mask
            # Temporarily store the raw hx (compression is done after the loop)
            hx_dc[i] = hx * mask

            penalty1[i] = torch.mean(h)
            penalty2[i] = torch.mean(self.gammas[i]**2)

        # log1p-compress + discount hx before fusion.
        # Use a new tensor to avoid in-place ops (keeps autograd happy).
        hx_dc_new = torch.zeros_like(hx_dc)
        for i in range(self.num_sources):
            hx_compressed = torch.log1p(hx_dc[i])
            hx_dc_new[i] = hx_compressed * torch.sigmoid(self.discounts[i])
        hx_dc = hx_dc_new

        # Fusion
        den = torch.sum(hx_dc, dim=0) + eps
        mux_comb = torch.sum(mux_total * hx_dc, dim=0) / den
        sig2x_comb = torch.sum(sig2x_total * hx_dc**2, dim=0) / (den ** 2)
        sig2x_comb = torch.clamp(sig2x_comb, min=eps)
        hx_comb = torch.sum(hx_dc, dim=0)

        mux_final = torch.cat((mux_total, mux_comb.unsqueeze(0)), dim=0)
        sig2x_final = torch.cat((sig2x_total, sig2x_comb.unsqueeze(0)), dim=0)
        hx_final = torch.cat((hx_total, hx_comb.unsqueeze(0)), dim=0)

        penalty1[i+1] = 0
        penalty2[i+1] = 0

        return {
            "mux": mux_final,
            "sig2x": sig2x_final,
            "hx": hx_final,
            'penalty1': penalty1,
            'penalty2': penalty2,
            'masks': masks,
            'modality_order': modality_order
        }


# ============================================================
# Complete-case Gaussian KL alignment loss
# ============================================================
def sym_kl_1d_gaussian(mu1, var1, mu2, var2, eps=1e-8):
    """
    Symmetric KL divergence (closed form for 1D Gaussians).
    SKL(q1, q2) = KL(q1||q2) + KL(q2||q1)

    Args:
        mu1, var1: mean and variance of the first Gaussian
        mu2, var2: mean and variance of the second Gaussian
        eps: numerical-stability constant

    Returns:
        Symmetric KL divergence [N]
    """
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    kl12 = 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0)
    kl21 = 0.5 * (torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / var1 - 1.0)
    return kl12 + kl21  # [N]


def gaussian_kl_alignment_loss(pred, masks, modality_order, eps=1e-8, detach_var=True):

    if masks is None or modality_order is None:
        return torch.tensor(0.0, dtype=torch.float64)

    # Find the index of RNA/WSI in pred['mux']
    if 'RNA' not in modality_order or 'WSI' not in modality_order:
        return torch.tensor(0.0, dtype=torch.float64)

    idx_rna = modality_order.index('RNA')
    idx_wsi = modality_order.index('WSI')

    # Get masks and convert to bool
    mask_r = masks.get('RNA', None)
    mask_w = masks.get('WSI', None)

    if mask_r is None or mask_w is None:
        return torch.tensor(0.0, dtype=torch.float64)

    if not isinstance(mask_r, torch.Tensor):
        mask_r = torch.tensor(mask_r)
    if not isinstance(mask_w, torch.Tensor):
        mask_w = torch.tensor(mask_w)

    mask_r = mask_r.to(torch.bool)
    mask_w = mask_w.to(torch.bool)

    # Complete-case: samples that have both modalities
    complete_mask = mask_r & mask_w

    if complete_mask.sum() == 0:
        return torch.tensor(0.0, dtype=pred['mux'].dtype, device=pred['mux'].device)

    # Extract predictions for the complete-case samples
    mu_r = pred['mux'][idx_rna][complete_mask]
    var_r = pred['sig2x'][idx_rna][complete_mask]
    mu_w = pred['mux'][idx_wsi][complete_mask]
    var_w = pred['sig2x'][idx_wsi][complete_mask]

    # Stop-gradient on variance so alignment only pulls the means together
    if detach_var:
        var_r = var_r.detach()
        var_w = var_w.detach()

    # Compute symmetric KL
    skl = sym_kl_1d_gaussian(mu_r, var_r, mu_w, var_w, eps=eps)
    return skl.mean()


class Loss_function(nn.Module):

    def __init__(self, fusion_weight=0.01, align_weight=0.01, detach_var_in_align=True):
        super(Loss_function, self).__init__()
        self.fusion_weight = fusion_weight
        self.align_weight = align_weight
        self.detach_var_in_align = detach_var_in_align

    def forward(self, y, nu, events, xi, rho, pred, sigma, lambd=None, c=1):
        if lambd is None:
            lambd = 0.5

        n_total = len(y)
        num_sources = len(pred['mux'])
        device = y.device
        total_loss = torch.zeros((num_sources), dtype=torch.float64, device=device)

        masks = pred.get('masks', None)
        modality_order = pred.get('modality_order', None)

        def _to_bool_mask(m):
            if m is None:
                return None
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(m)
            m = m.to(device)
            return (m > 0.5)

        def _get_mask_for_source(i):
            if masks is None or modality_order is None:
                return None, None

            if i < len(modality_order):
                name = modality_order[i]
                return _to_bool_mask(masks.get(name, None)), name

            # fusion (last) -- union mask (any modality present)
            if i == num_sources - 1:
                any_mask = None
                for name in modality_order:
                    if name in masks:
                        m = _to_bool_mask(masks[name])
                        any_mask = m if any_mask is None else (any_mask | m)
                return any_mask, 'fusion'

            return None, None

        for i in range(num_sources):
            mask_bool, name = _get_mask_for_source(i)

            # Take the current source's predictions
            mux_all = pred['mux'][i]
            sig2x_all = pred['sig2x'][i]
            hx_all = pred['hx'][i]
            penalty1 = pred['penalty1'][i]
            penalty2 = pred['penalty2'][i]

            # penalty may be on CPU; make sure the device matches
            if isinstance(penalty1, torch.Tensor):
                penalty1 = penalty1.to(device)
            else:
                penalty1 = torch.tensor(penalty1, dtype=torch.float64, device=device)
            if isinstance(penalty2, torch.Tensor):
                penalty2 = penalty2.to(device)
            else:
                penalty2 = torch.tensor(penalty2, dtype=torch.float64, device=device)

            # === Sample-level mask: compute loss only on valid samples ===
            if mask_bool is not None:
                w = mask_bool.to(torch.float64).mean()  # ratio of valid samples
                if w < 1e-6:
                    total_loss[i] = torch.tensor(0.0, dtype=torch.float64, device=device)
                    continue
                idx_valid = torch.nonzero(mask_bool, as_tuple=True)[0]
                y_i = y[idx_valid]
                events_i = events[idx_valid]
                mux = mux_all[idx_valid]
                sig2x = sig2x_all[idx_valid]
                hx = hx_all[idx_valid]
            else:
                w = torch.tensor(1.0, dtype=torch.float64, device=device)
                y_i = y
                events_i = events
                mux = mux_all
                sig2x = sig2x_all
                hx = hx_all

            # ---------- loss1 (likelihood-like) ----------
            sig2x = torch.clamp(sig2x, min=1e-8)
            sigx = torch.sqrt(sig2x)
            Z2 = hx * sig2x + 1
            Z = torch.sqrt(Z2)
            sig1 = sigx * Z

            pl = 1 / Z * torch.exp(-0.5 * hx * (y_i - mux) ** 2 / Z2)

            eps = 1e-4 * torch.std(y_i)
            if torch.isnan(eps) or eps == 0:
                eps = torch.tensor(1e-4, dtype=torch.float64, device=device)

            norm_dist = Normal(mux, sigx)
            Sy1 = 1 - norm_dist.cdf(y_i) - pl + pl * Normal(mux, sig1).cdf(y_i)

            pl1 = 1 / Z * torch.exp(-0.5 * hx * (y_i - eps - mux) ** 2 / Z2)
            pl2 = 1 / Z * torch.exp(-0.5 * hx * (y_i + eps - mux) ** 2 / Z2)

            Sy2 = 1 - norm_dist.cdf(y_i) + pl * Normal(mux, sig1).cdf(y_i)
            Fy2_1 = norm_dist.cdf(y_i + eps) + pl1 * Normal(mux, sig1).cdf(y_i - eps)
            Fy2_2 = norm_dist.cdf(y_i - eps) - pl2 * (1 - Normal(mux, sig1).cdf(y_i + eps))

            fy2 = Fy2_1 - Fy2_2
            fy1 = fy2 - pl1 * Normal(mux, sig1).cdf(y_i) - pl2 * (1 - Normal(mux, sig1).cdf(y_i))

            Sy1 = torch.clamp(Sy1, min=0.0)
            Sy2 = torch.clamp(Sy2, min=0.0)
            fy1 = torch.clamp(fy1, min=0.0)
            fy2 = torch.clamp(fy2, min=0.0)

            # Use masked events_i so tensor shapes stay aligned under missing-modality training.
            loss1 = -lambd * torch.mean(torch.log(fy1 + nu) * events_i + torch.log(Sy2 + nu) * (1 - events_i)) \
                - (1 - lambd) * torch.mean(torch.log(fy2 + nu) * events_i + torch.log(Sy1 + nu) * (1 - events_i)) \
                + xi * penalty1 + rho * penalty2

            # ---------- loss2 (ranking-like) ----------
            if len(y_i) > 1:
                idx = torch.argsort(y_i)

                # Recompute for surv_df
                sig2x_sort = torch.clamp(sig2x, min=1e-8)
                sigx_sort = torch.sqrt(sig2x_sort)
                Z2_sort = hx * sig2x_sort + 1
                Z_sort = torch.sqrt(Z2_sort)

                D, M = torch.meshgrid(y_i, mux, indexing='ij')
                diff = (D - M)
                exp_term = torch.exp(-0.5 * hx * diff ** 2 / Z2_sort)
                cumulative_density = torch.distributions.Normal(mux, sigx_sort).cdf(D)

                Fy1_mat = cumulative_density - 1 / Z_sort * exp_term * cumulative_density
                Fy2_mat = Fy1_mat + 1 / Z_sort * exp_term

                surv_df = 1 - (lambd * Fy1_mat + (1 - lambd) * Fy2_mat)
                surv_df = surv_df[:, idx]
                surv_df = surv_df[idx, :]

                n = surv_df.shape[0]
                matrix = torch.triu(torch.ones((n, n), dtype=torch.float64, device=device), diagonal=1)
                temp = torch.exp((torch.diagonal(surv_df).unsqueeze(1).expand(-1, n) - surv_df) / sigma)
                final = temp * matrix

                events_sorted = events_i[idx]
                final_upper = events_sorted.reshape(n, 1).expand(-1, n) * final

                n_nonzero = int((final_upper != 0).sum().item())
                if n_nonzero > 0:
                    loss2 = torch.sum(final_upper) / n_nonzero
                else:
                    loss2 = torch.tensor(0.0, dtype=torch.float64, device=device)
            else:
                loss2 = torch.tensor(0.0, dtype=torch.float64, device=device)

            loss = c * loss1 + (1 - c) * loss2

            total_loss[i] = loss

        # === Per-modality weights (bound to modality_order to avoid hard-coded indices) ===
        if modality_order is not None:
            for i, name in enumerate(modality_order):
                if name == 'RNA':
                    total_loss[i] *= 1.0
                elif name == 'WSI':
                    total_loss[i] *= 1.0
        else:
            total_loss[0] *= 1.0
            if len(total_loss) > 2:
                total_loss[1] *= 1.0

        # Fusion-result weight
        total_loss[-1] *= self.fusion_weight

        # === Complete-case Gaussian KL alignment loss ===
        if self.align_weight > 0 and masks is not None and modality_order is not None:
            align_loss = gaussian_kl_alignment_loss(
                pred, masks, modality_order,
                eps=1e-8, detach_var=self.detach_var_in_align
            )
            return torch.mean(total_loss) + self.align_weight * align_loss.to(device)

        return torch.mean(total_loss)


class Eval_Loss_function(nn.Module):
    def __init__(self):
        super(Eval_Loss_function, self).__init__()

    def forward(self, y, nu, events, xi, rho, pred, sigma, lambd=None, c=1):
        n = len(y)
        i = -1  # use the fused prediction

        mux = pred["mux"][i]
        sig2x = pred["sig2x"][i]
        hx = pred["hx"][i]
        penalty1 = pred["penalty1"][i]
        penalty2 = pred["penalty2"][i]

        sig2x = torch.clamp(sig2x, min=1e-8)
        sigx = torch.sqrt(sig2x)
        Z2 = hx * sig2x + 1
        Z = torch.sqrt(Z2)
        sig1 = sigx * Z

        pl = 1 / Z * torch.exp(-0.5 * hx * (y - mux) ** 2 / Z2)

        eps = 1e-4 * torch.std(y)
        norm_dist = Normal(mux, sigx)
        Fy1 = norm_dist.cdf(y) - pl * Normal(mux, sig1).cdf(y)
        Sy1 = 1 - Fy1

        pl1 = 1 / Z * torch.exp(-0.5 * hx * (y - eps - mux) ** 2 / Z2)
        pl2 = 1 / Z * torch.exp(-0.5 * hx * (y + eps - mux) ** 2 / Z2)

        Fy2 = Fy1 + pl
        Sy2 = 1 - Fy2
        Fy2_1 = norm_dist.cdf(y + eps) + pl1 * Normal(mux, sig1).cdf(y - eps)
        Fy2_2 = norm_dist.cdf(y - eps) - pl2 * (1 - Normal(mux, sig1).cdf(y + eps))

        fy2 = Fy2_1 - Fy2_2
        fy1 = fy2 - pl1 * Normal(mux, sig1).cdf(y) - pl2 * (1 - Normal(mux, sig1).cdf(y))

        Sy1 = torch.clamp(Sy1, min=0.0)
        Sy2 = torch.clamp(Sy2, min=0.0)
        fy1 = torch.clamp(fy1, min=0.0)
        fy2 = torch.clamp(fy2, min=0.0)

        loss1 = -lambd * torch.mean(torch.log(fy1 + nu) * events + torch.log(Sy1 + nu) * (1 - events)) \
                - (1 - lambd) * torch.mean(torch.log(fy2 + nu) * events + torch.log(Sy2 + nu) * (1 - events)) \
                + xi * penalty1 + rho * penalty2

        idx = np.argsort(y)
        mux = pred['mux'][i]
        sig2x_sort = torch.clamp(pred['sig2x'][i], min=1e-8)
        sigx = torch.sqrt(sig2x_sort)
        hx = pred['hx'][i]
        Z2 = hx * sig2x_sort + 1
        Z = torch.sqrt(Z2)

        D, M = torch.meshgrid(y, mux, indexing='ij')  # explicitly specify indexing
        diff = (D - M)
        exp_term = torch.exp(-0.5 * hx * diff ** 2 / Z2)
        cumulative_density = torch.distributions.Normal(mux, sigx).cdf(D)

        Fy1 = cumulative_density - 1 / Z * exp_term * cumulative_density
        Fy2 = Fy1 + 1 / Z * exp_term

        surv_df = 1 - (lambd * Fy1 + (1 - lambd) * Fy2)
        surv_df = surv_df[:, idx]
        surv_df = surv_df[idx, :]

        matrix = torch.triu(torch.ones((n, n), dtype=int), diagonal=1)
        temp = torch.exp((torch.diagonal(surv_df).unsqueeze(1).expand(-1, n) - surv_df) / sigma)
        final = temp * matrix
        final_upper = events[idx].reshape(n, 1).expand(-1, n) * final

        n_nonzero = torch.nonzero(final_upper).size(0)
        if n_nonzero > 0:
            loss2 = torch.sum(final_upper) / n_nonzero
        else:
            loss2 = torch.tensor(0.0, dtype=torch.float64)

        loss = c * loss1 + (1 - c) * loss2

        return loss


def _as_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def _build_dispro_bins(train_times, train_events, n_bins=4, eps=1e-8):
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=bool)
    uncensored = train_times[train_events]

    if uncensored.size > 0:
        try:
            _, q_bins = pd.qcut(uncensored, q=n_bins, retbins=True, labels=False, duplicates='drop')
            q_bins = np.asarray(q_bins, dtype=float)
        except Exception:
            q_bins = np.linspace(train_times.min() - eps, train_times.max() + eps, n_bins + 1)

        if q_bins.size >= 2:
            q_bins[0] = train_times.min() - eps
            q_bins[-1] = train_times.max() + eps
        q_bins = np.unique(q_bins)
    else:
        q_bins = np.linspace(train_times.min() - eps, train_times.max() + eps, n_bins + 1)

    if q_bins.size < 3:
        q_bins = np.linspace(train_times.min() - eps, train_times.max() + eps, max(3, n_bins + 1))

    return q_bins


def _predict_survival_at_times(mux, sigx, sig1, hx, Z, Z2, weight, times, pt=None, YJ=False):
    times_t = torch.as_tensor(times, dtype=torch.float64, device=mux.device)

    if YJ:
        D_input = YJtransform(times_t.detach().cpu(), pt).to(mux.device)
    else:
        D_input = torch.log(times_t)

    D, M = torch.meshgrid(D_input, mux, indexing='ij')
    diff = D - M
    pl = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
    Fy1 = torch.distributions.Normal(mux, sigx).cdf(D) - pl * torch.distributions.Normal(mux, sig1).cdf(D)
    Fy2 = Fy1 + pl
    surv = 1 - (weight * Fy1 + (1 - weight) * Fy2)
    return torch.clamp(surv, min=1e-12, max=1.0)


def _compute_dispro_style_ibs(mux, sigx, sig1, hx, Z, Z2, weight,
                              durations_train, events_train,
                              durations_test, events_test,
                              pt=None, YJ=False, n_bins=4):
    from sksurv.metrics import integrated_brier_score

    train_times = _as_numpy_1d(durations_train).astype(float)
    train_events = _as_numpy_1d(events_train).astype(bool)
    test_times = _as_numpy_1d(durations_test).astype(float)
    test_events = _as_numpy_1d(events_test).astype(bool)

    bins = _build_dispro_bins(train_times, train_events, n_bins=n_bins)
    times = bins[1:].copy()
    if times.size < 2:
        return float('nan')

    surv_probs = _predict_survival_at_times(
        mux, sigx, sig1, hx, Z, Z2, weight, times, pt=pt, YJ=YJ
    ).detach().cpu().numpy().T

    # Same clipping strategy as DisPro: keep only the common observable range.
    t_min = max(np.min(train_times), np.min(test_times))
    t_max = min(np.max(train_times), np.max(test_times))
    col_mask = (times > t_min) & (times < t_max)
    if col_mask.sum() < 2:
        return float('nan')
    times = times[col_mask]
    surv_probs = surv_probs[:, col_mask]

    dt = np.dtype([('event', bool), ('time', float)])
    y_train = np.array(list(zip(train_events, train_times)), dtype=dt)
    y_test = np.array(list(zip(test_events, test_times)), dtype=dt)

    try:
        return integrated_brier_score(y_train, y_test, surv_probs, times)
    except Exception:
        return float('nan')


def evreg_evaluation(pred, durations_test, events_test, weight, pt=None, YJ=False,
                     durations_train=None, events_train=None, ibs_bins=4):
    mux = pred['mux'][-1]
    sig2x = torch.clamp(pred['sig2x'][-1], min=1e-8)
    sigx = torch.sqrt(sig2x)
    hx = pred['hx'][-1]
    Z2 = hx * sig2x + 1
    Z = torch.sqrt(Z2)
    sig1 = sigx * Z

    durations_test_np = _as_numpy_1d(durations_test).astype(float)
    time_grid = np.linspace(durations_test_np.min(), durations_test_np.max(), 100)

    surv_eval = _predict_survival_at_times(
        mux, sigx, sig1, hx, Z, Z2, weight, durations_test_np, pt=pt, YJ=YJ
    )
    surv_df = pd.DataFrame(surv_eval.detach().cpu().numpy(), index=durations_test_np)

    events_np = _as_numpy_1d(events_test)
    ev = EvalSurv(surv_df, durations_test_np, events_np, censor_surv='km')

    c_index = ev.concordance_td('adj_antolini')
    NBLL = ev.integrated_nbll(time_grid)

    if durations_train is not None and events_train is not None:
        IBS = _compute_dispro_style_ibs(
            mux, sigx, sig1, hx, Z, Z2, weight,
            durations_train=durations_train,
            events_train=events_train,
            durations_test=durations_test_np,
            events_test=events_test,
            pt=pt, YJ=YJ, n_bins=ibs_bins
        )
    else:
        # Backward-compatible fallback when train split stats are not provided.
        IBS = ev.integrated_brier_score(time_grid)

    return c_index, IBS, NBLL


def YJtransform(x, pt):
    return torch.tensor(pt.transform(torch.log(x.reshape(-1, 1))).squeeze(), dtype=torch.float64)
