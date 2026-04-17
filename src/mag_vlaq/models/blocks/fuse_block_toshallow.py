import torch
import torch.nn.functional as F
from torch import nn

from mag_vlaq.models.blocks.diff_block import DiffBlock
from mag_vlaq.models.layers.sparse_utils import sparse_global_avg_pool


def _has_diff_type(args, name):
    return any(item.split("@", 1)[0] == name for item in args.diff_type)


def _stage_indexes(length, direction):
    if direction == "forward":
        return range(length)
    if direction == "backward":
        return range(length - 1, -1, -1)
    raise NotImplementedError


class FuseBlockToShallow(nn.Module):
    def __init__(self, dims=None, img_dims=None, vox_dims=None, bev_dims=None, args=None):
        if bev_dims is None:
            bev_dims = [64, 128, 256]
        if vox_dims is None:
            vox_dims = [64, 128, 256]
        if img_dims is None:
            img_dims = [64, 128, 256]
        if dims is None:
            dims = [256, 256, 256]
        super().__init__()
        if args is None:
            raise ValueError("FuseBlockToShallow requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args

        self.dims = dims
        self.img_dims = img_dims
        self.vox_dims = vox_dims
        self.bev_dims = bev_dims
        self.blocks = nn.ModuleList()
        self.updimsbev = nn.ModuleList()
        self.updimsimg = nn.ModuleList()
        self.updimsvox = nn.ModuleList()
        for i in range(len(dims)):
            diffblock = DiffBlock(dim=dims[-1], ode_dim=dims[-1], args=self.args)
            self.blocks.append(diffblock)
            if self.img_dims[i] == dims[-1]:
                self.updimsimg.append(nn.Identity())
            else:
                self.updimsimg.append(nn.Linear(self.img_dims[i], dims[-1]))

            if self.vox_dims[i] != dims[-1]:
                raise ValueError(
                    "Voxel features must be projected to the fusion dimension in UtoniaFE; "
                    f"got vox_dims[{i}]={self.vox_dims[i]} and fusion dim={dims[-1]}"
                )
            self.updimsvox.append(nn.Identity())

        # self.cde = DiffBlock(dim=dims[-1], ode_dim=dims[-1])

    def forward_imgbev(self, imagemaplist, bevmaplist=None, voxmaplist=None):
        assert len(imagemaplist) == len(self.dims)

        imageveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in imagemaplist]
        bevveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in bevmaplist]

        if _has_diff_type(self.args, "cde"):
            # ==== cde
            if self.args.diff_direction == "forward":
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist))]
                bevveclist = [self.updimsbev[i](bevveclist[i]) for i in range(len(bevveclist))]
            elif self.args.diff_direction == "backward":
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist) - 1, -1, -1)]
                bevveclist = [self.updimsbev[i](bevveclist[i]) for i in range(len(bevveclist) - 1, -1, -1)]
            imageveclist = torch.stack(imageveclist, dim=1)  # [b,seq,c]
            bevveclist = torch.stack(bevveclist, dim=1)
            fuseveclist = imageveclist + bevveclist
            fusevec = self.cde(fuseveclist, z0=fuseveclist[:, 0])
        else:
            # ==== deep to shallow
            fusevec = 0
            for i in _stage_indexes(len(self.dims), self.args.diff_direction):
                imagevec = imageveclist[i]
                bevvec = bevveclist[i]
                block = self.blocks[i]
                updimimage = self.updimsimg[i]
                updimbev = self.updimsbev[i]

                imagevec = updimimage(imagevec)
                bevvec = updimbev(bevvec)

                fusevec = fusevec + imagevec + bevvec
                fusevec = block(fusevec)

        return fusevec

    def forward_imgvox(self, imagemaplist, bevmaplist=None, voxmaplist=None):
        assert len(imagemaplist) == len(self.dims)

        imageveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in imagemaplist]
        voxveclist = [sparse_global_avg_pool(e) for e in voxmaplist]

        if _has_diff_type(self.args, "cde"):
            # ==== cde
            if self.args.diff_direction == "forward":
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist))]
                voxveclist = [self.updimsvox[i](voxveclist[i]) for i in range(len(voxveclist))]
            elif self.args.diff_direction == "backward":
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist) - 1, -1, -1)]
                voxveclist = [self.updimsvox[i](voxveclist[i]) for i in range(len(voxveclist) - 1, -1, -1)]
            imageveclist = torch.stack(imageveclist, dim=1)  # [b,seq,c]
            voxveclist = torch.stack(voxveclist, dim=1)
            fuseveclist = imageveclist + voxveclist
            fusevec = self.cde(fuseveclist, z0=fuseveclist[:, 0])

        else:
            # ==== deep to shallow
            fusevec = 0
            for i in _stage_indexes(len(self.dims), self.args.diff_direction):
                imagevec = imageveclist[i]
                voxvec = voxveclist[i]
                block = self.blocks[i]
                updimimage = self.updimsimg[i]
                updimvox = self.updimsvox[i]

                imagevec = updimimage(imagevec)
                voxvec = updimvox(voxvec)

                fusevec = fusevec + imagevec + voxvec
                fusevec = block(fusevec)

        return fusevec

    def forward(self, imagefeatmaplist, bevfeatmaplist, voxfeatmaplist, type=None):
        if type == "bev":
            output = self.forward_imgbev(imagefeatmaplist, bevfeatmaplist, voxfeatmaplist)
        elif type == "vox":
            output = self.forward_imgvox(imagefeatmaplist, bevfeatmaplist, voxfeatmaplist)
        else:
            raise NotImplementedError
        return output
