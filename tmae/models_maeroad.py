# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import math
import torch
import torch.nn as nn

from ViT_cross import Block, CrossAttentionBlock, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
from road import RoadNet 


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,self_attn=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) 
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.self_attn=self_attn
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # Dealing with positional embedding, patch sampling 
        # encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # decoder 
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # --------------------------------------------------------------------------
        self.road_net=RoadNet(patch_size=patch_size,base_channels=embed_dim)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def attention_masking(self,x,attention,road,mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        keep_n = int(L * (1-mask_ratio)) #遮蔽
        ids_shuffle = torch.argsort(attention, dim=1,descending=True)  # 降序
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 还原下标
        ids_keep=ids_shuffle[:,:keep_n]
        ids_drop=ids_shuffle[:,keep_n:]
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_(1,ids_keep,0)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) 
        road_vis=torch.gather(road,dim=1,index=ids_drop.unsqueeze(-1).repeat(1, 1, D))
        x_ = torch.cat([x_masked, road_vis], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x_,mask,ids_restore
    def grid_patchify(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        return x

    def forward_encoder(self, x,attention,road, mask_ratio):
        x, mask, ids_restore = self.attention_masking(x, attention,road,mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :] 
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
    def forward_encoder_test(self,x):
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for i,blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def mask_tokens_grid(self, mask, ids_restore):
        N, L = ids_restore.shape
        x = self.decoder_pos_embed[:, 1:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.shape[-1])
        x = x + self.mask_token
        return x

    def forward_decoder(self, y, mask, ids_restore, mask_ratio):
        N, L = ids_restore.shape
        x_=self.decoder_embed(y)
        pos=self.decoder_pos_embed[:,1:,:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1,  self.decoder_embed_dim)
        x=x_[:,1:,:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1,  self.decoder_embed_dim)
        x=x+pos
        mask_=1-mask
        y=y[:,1:,:].masked_select(mask_.bool().unsqueeze(-1)).reshape(N, -1, self.embed_dim)

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, y)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x) 
        return x, None
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.shape[0], -1, target.shape[-1])

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean()

        assert np.isnan(loss.cpu().detach().numpy())==False
        return loss
    
    def forward_metric(self, imgs, pred, mask):
        target = self.patchify(imgs)
        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.shape[0], -1, target.shape[-1])
        mse = (pred - target) ** 2
        mse = mse.mean()
        mae=(pred - target).abs().mean().cpu().detach().numpy()
        mapes = []
        for i in range(len(pred)):
            gt_sum = target[i].abs().sum()
            er_sum = (target[i] - pred[i]).abs().sum()
            mape=(er_sum / gt_sum).cpu().detach().numpy()
            mapes.append(mape)
        mapes=np.mean(mapes)
        assert np.isnan(mse.cpu().detach().numpy())==False
        return mse,mae,mape

    def forward(self, imgs,road_map,ent,i,mode='att',mask_ratio=0.75):
        with torch.cuda.amp.autocast():
            x = self.grid_patchify(imgs)
            N,L,_=x.shape
            road=self.road_net(road_map).expand(16,-1,-1)
            if i%2==0:
                entropy=ent.unsqueeze(0).repeat(imgs.shape[0],1)
                latent, mask, ids_restore= self.forward_encoder(x,entropy,road,mask_ratio)
                pred, combined = self.forward_decoder(latent, mask, ids_restore, mask_ratio)  # [N, L, p*p*3] [16, 768, 16]
            else:
                if mode=='att':
                    _,attention_flow=self.forward_encoder_test(x)#[16, 8, 1025, 1025]
                    attention_flow=attention_flow[:,:,0,1:].mean(1).detach().clone()
                elif mode=='random':
                    attention_flow=torch.rand(N, L, device=x.device)
                latent, mask, ids_restore= self.forward_encoder(x,attention_flow,road,mask_ratio)
                pred, combined = self.forward_decoder(latent, mask, ids_restore,  mask_ratio)  # [N, L, p*p*3] [16, 768, 16]
            loss = self.forward_loss(imgs, pred, mask)

            return loss

    def forward_test(self, imgs,entropy,road_map,mask_ratio=0.75):
        with torch.cuda.amp.autocast():
            x = self.grid_patchify(imgs)
            road=self.road_net(road_map).expand(16,-1,-1)
            entropy=entropy.unsqueeze(0).repeat(imgs.shape[0],1)
            latent, mask, ids_restore= self.forward_encoder(x,entropy,road,mask_ratio)
            pred, combined = self.forward_decoder(latent, mask, ids_restore, mask_ratio)  # [N, L, p*p*3] [16, 768, 16]
            mse,mae,mape = self.forward_metric(imgs,pred, mask)

            return mse,mae,mape

        # return loss

def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128,patch_size=4,in_chans=1, embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def mae_vit_base_patch2_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=64,patch_size=2,in_chans=2, embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b
mae_vit_base_patch2 = mae_vit_base_patch2_dec512d8b
model = MaskedAutoencoderViT()
model.forward_encoder