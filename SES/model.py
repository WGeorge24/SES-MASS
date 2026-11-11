"""SES model: MASS backbone with an equal-interval epoch timetable."""

from __future__ import annotations

from . import _env  # noqa: F401

from typing import Dict, Optional

import torch

from . import paths as _paths  # noqa: F401
from structure.mass import MASS  # noqa: E402

from .config import ModelHParams
from .sampling import ObservationPolicy, PolicyOutput


class SES(MASS):
    """MASS three-stage encoder. The observation policy decides which tokens exist.

    ``equal`` keeps whole epochs and 1 s patches on fixed equal-interval grids.
    Eval uses phase 0. Train may shift a grid inside one step and, for patches
    only, mix a fraction of random subsets.
    """

    def __init__(
        self,
        hparams: ModelHParams,
        policy: ObservationPolicy,
        mean,
        std,
        seed: int = 2025,
    ):
        intra_mask = getattr(policy, "intra_mask", 0.0)
        inter_mask = getattr(policy, "inter_mask", 0.0)
        super().__init__(
            embed_dim=hparams.embed_dim,
            num_heads=hparams.num_heads,
            mlp_ratio=hparams.mlp_ratio,
            dropout=hparams.dropout,
            num_patches=hparams.num_patches,
            num_epoches=hparams.num_epoches,
            intra_mask_ratio=intra_mask,
            inter_mask_ratio=inter_mask,
            global_encoder_depth=hparams.global_encoder_depth,
            intra_encoder_depth=hparams.intra_encoder_depth,
            inter_encoder_depth=hparams.inter_encoder_depth,
            inter_hidden_dim=hparams.inter_hidden_dim,
            mean=mean,
            std=std,
            seed=seed,
        )
        self.hparams = hparams
        self.policy = policy
        self.last_policy: Optional[PolicyOutput] = None

    def generate_global_prompt(self, eeg, policy_out: PolicyOutput):
        b, num_epoches, num_patches, _ = eeg.shape
        unmasked_epoches_index_batch = policy_out.epoch_visible
        masked_epoches_index_batch = policy_out.epoch_masked
        unmasked_patches_idx_batch = policy_out.patch_visible
        num_visible_epoches = unmasked_epoches_index_batch.shape[1]

        batch_indices = torch.arange(b, device=eeg.device).view(-1, 1)
        visible_epoches_data = eeg[batch_indices, unmasked_epoches_index_batch]

        batch_idx_3 = torch.arange(b, device=eeg.device).view(b, 1, 1)
        epoch_idx_3 = torch.arange(num_visible_epoches, device=eeg.device).view(1, num_visible_epoches, 1)
        all_patches = visible_epoches_data[batch_idx_3, epoch_idx_3, unmasked_patches_idx_batch]
        all_patches = all_patches.view(b, -1, all_patches.shape[-1])

        patch_tokens = self.patch_embedding(all_patches)
        global_cls_tokens = self.global_cls_token.expand(b, -1, -1)
        global_input = torch.cat([global_cls_tokens, patch_tokens], dim=1)

        absolute_positions = unmasked_epoches_index_batch.unsqueeze(2) * num_patches + unmasked_patches_idx_batch
        absolute_positions = absolute_positions.view(b, -1)
        max_position = num_epoches * num_patches
        full_pos_encoding = self.create_sinusoidal_encoding(max_position + 1, self.embed_dim).to(eeg.device)
        cls_pos_encoding = full_pos_encoding[0:1].expand(b, -1, -1)
        patch_pos_encoding = full_pos_encoding[absolute_positions + 1]
        pos_encoding = torch.cat([cls_pos_encoding, patch_pos_encoding], dim=1)
        global_input = global_input + pos_encoding

        global_hidden = self.global_encoder_norm(self.global_encoder(global_input))
        prompt_token = global_hidden[:, 0, :]

        patch_masks_dict = {}
        for i in range(b):
            sample_masks = {}
            for j, epoch_idx in enumerate(unmasked_epoches_index_batch[i]):
                sample_masks[int(epoch_idx.item())] = unmasked_patches_idx_batch[i, j]
            patch_masks_dict[i] = sample_masks

        return prompt_token, unmasked_epoches_index_batch, masked_epoches_index_batch, patch_masks_dict

    def encode(self, eeg: torch.Tensor) -> Dict:
        b, num_epoches, _, _ = eeg.shape
        self.policy.train(self.training)
        policy_out = self.policy.sample(b, eeg.device, eeg=eeg)
        self.last_policy = policy_out
        eeg_obs = self.policy.apply_frequency(eeg, policy_out.freq_mask)

        prompt_token, vis_e, msk_e, patch_masks = self.generate_global_prompt(eeg_obs, policy_out)
        epoch_cls = self.intra_epoch_encoding(eeg_obs, prompt_token, vis_e, patch_masks)
        features = self.inter_epoch_encoding(epoch_cls, prompt_token, vis_e, msk_e)

        flat = features.contiguous().view(b * num_epoches, -1)
        stage_logits = self.sleep_stage_classifier(flat)

        num_masked = msk_e.shape[1]
        if num_masked > 0:
            batch_indices = torch.arange(b, device=eeg.device).view(-1, 1)
            masked_features = features[batch_indices, msk_e].view(b * num_masked, -1)
            transition_logits = self.transition_classifier(masked_features)
        else:
            transition_logits = self.transition_classifier(flat.new_zeros(0, flat.shape[-1]))

        return {
            "stage": stage_logits,
            "transition": transition_logits,
            "features": features,
            "policy": policy_out,
            "masked_epochs": msk_e,
        }

    def forward(self, eeg, temperature=None, transition=None, return_aux: bool = False):
        b = eeg.shape[0]
        out = self.encode(eeg)
        msk_e = out["masked_epochs"]
        if transition is not None:
            if msk_e.shape[1] > 0:
                batch_indices = torch.arange(b, device=eeg.device).view(-1, 1)
                transition_label = transition[batch_indices, msk_e].reshape(-1)
            else:
                transition_label = transition.new_zeros(0, dtype=torch.long)
        else:
            transition_label = None

        if return_aux:
            out["transition_label"] = transition_label
            return out
        return out["stage"], out["transition"], transition_label
