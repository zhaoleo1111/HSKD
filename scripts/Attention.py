import torch
from torch import nn
import torch.nn.functional as F
    
def get_attention(preds,temp=0.5):
    """ preds: Bs*C*W*H """
    N, C, H, W= preds.shape
    value = torch.abs(preds)
    # Bs*W*H
    fea_map = value.mean(axis=1, keepdim=True)
    S_attention = (F.sigmoid((fea_map/temp).view(N,-1))).view(N,-1, H, W)
    # Bs*C
    channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
    C_attention = F.sigmoid(channel_map/temp).view(N,C,1,1)

    return S_attention, C_attention

def mask_method(feat,r=0.15):
    N, C, H, W = feat.shape
    device = feat.device
    masked_feat = feat.clone()
    mat = torch.rand((N,C,1,1)).to(device)
    mat = torch.where(mat< r,0,1)  
    masked_feat = torch.mul(masked_feat, mat)
    return masked_feat


def mask_import(feat,feat3,r=0.65):
    N, C, H, W = feat.shape
    device = feat.device
    sorted_s, indices_s = torch.sort(torch.nn.functional.normalize(feat3, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    N, C, H, W = feat.shape  #1,256,8,8
    length = int(len(indices_s)*r)
    p_indices = indices_s[0:length]
    mat = torch.ones((N,C,1,1)).to(device)
    mat = mat.index_fill(1, p_indices, 0)
    masked_feat = torch.mul(feat, mat)
    return masked_feat

def mask_noimport(feat,r=0.15):
    N, C, H, W = feat.shape
    device = feat.device
    sorted_s, indices_s = torch.sort(torch.nn.functional.normalize(feat, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    N, C, H, W = feat.shape  #1,256,8,8
    length = int(len(indices_s)*r)
    p_indices = indices_s[0:length]
    mat = torch.zeros((N,C,1,1)).to(device)
    mat = mat.index_fill(1, p_indices, 1)
    masked_feat = torch.mul(feat, mat)
    return masked_feat

def spatial_mask(feat,att,r=0.55):
    att = torch.where(att >=r, 0, 1)  
    masked_feat = torch.mul(feat, att)
    return masked_feat

def channel_mask(feat,att,r=0.65):
    att = torch.where(att >=r, 0, 1)  
    masked_feat = torch.mul(feat, att)
    return masked_feat


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    # student_matrix = student_matrix / torch.norm(student_matrix, dim=-1, keepdim=True)
    # teacher_matrix = teacher_matrix / torch.norm(teacher_matrix, dim=-1, keepdim=True)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss