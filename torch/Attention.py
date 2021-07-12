import torch
import torch.nn as nn
import math

# class sSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.pa = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 8, 1, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, U):
#         q = self.pa(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
#         return U * q  # 广播机制

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q # 广播机制



class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        # self.ca = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // 8, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels // 8, in_channels, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )
    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)
        # z = self.ca(z)
        # return U * z






class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class scSE_8(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = deepWiseChannelAttention(in_channels)
        self.sSE =sSE(in_channels)
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        res = self.conv(torch.cat([U_sse, U_cse], dim = 1))
        # print(res.shape)
        res += U
        return res

def get_ld_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)

    # split channel for multi-spectral attention
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                val = get_ld_dct(t_x, u_x, width) * get_ld_dct(t_y, v_y, height)
                dct_weights[:, i * c_part: (i+1) * c_part, t_x, t_y] = val

    return dct_weights


class FcaLayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(FcaLayer, self).__init__()
        self.register_buffer("pre_computed_dct_weights", get_dct_weights(64,64,32,[0,1,2,3,4,5,6,0,2,6,0,0,0,0,2,5],[0,0,0,0,0,0,0,1,1,1,3,4,5,6,3,4]))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        n,c,_,_ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        y = self.fc(y).view(n,c,1,1)
        return x * y.expand_as(x)

# class sSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.pa = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 8, 1, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, U):
#         q = self.pa(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
#         return U * q  # 广播机制
class deepWiseChannelAttention(nn.Module):
    def __init__(self,in_channels):
        super(deepWiseChannelAttention,self).__init__()
        self.Deepwise = nn.Sequential(
            nn.Conv2d(in_channels, 1,1,padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        q = self.Deepwise(x)
        return x *q


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

if __name__ == "__main__":
    bs, c, h, w = 4, 16, 256, 256
    in_tensor = torch.ones(bs, c, h, w)

    sc_se = scSE_8(c)
    print("in shape:",in_tensor.shape)
    out_tensor= sc_se(in_tensor)
    print("out shape:", out_tensor.shape)
    # fca = FcaLayer(32)
    # out = fca(in_tensor)
    # print("out shape:", out.shape)
