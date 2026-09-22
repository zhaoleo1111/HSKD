import torch
from torch import nn
import math
import torch.nn.functional as F

class SampleSimilarities(nn.Module):

    def __init__(self, feats_dim, queueSize, T):
        super(SampleSimilarities, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q, update=True):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).cuda()
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, q)
                self.index = (self.index + batchSize) % self.queueSize
        return out

class SampleSimilarities5(nn.Module):

    def __init__(self, feats_dim, queueSize, T):
        super(SampleSimilarities5, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q, update=True):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        out1 = torch.mm(queue.detach(), q[:batchSize//2].transpose(1, 0))
        out1 = out1.transpose(0, 1)
        out1 = torch.div(out1, self.T)
        out1 = out1.squeeze().contiguous()

        out2 = torch.mm(queue.detach(), q[batchSize//2:].transpose(1, 0))
        out2 = out2.transpose(0, 1)
        out2 = torch.div(out2, self.T)
        out2 = out2.squeeze().contiguous()

        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).cuda()
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, q)
                self.index = (self.index + batchSize) % self.queueSize
        return out,out1,out2
    
class CompReSS5(nn.Module):

    def __init__(self , teacher_feats_dim, student_feats_dim, queue_size=128000, T=0.04):
        super(CompReSS5, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.student_sample_similarities = SampleSimilarities5(student_feats_dim , queue_size , T).cuda()
        self.teacher_sample_similarities = SampleSimilarities5(teacher_feats_dim , queue_size , T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student,out_s1,out_s2 = self.student_sample_similarities(student_feats)
        similarities_teacher,out_t1,out_t2 = self.teacher_sample_similarities(teacher_feats)
        inter_loss = (self.criterion(out_s1, out_s2) + self.criterion(out_t1 , out_t2)).mean()
        st_loss = self.criterion(similarities_teacher, similarities_student)
        loss = inter_loss + st_loss
        # print(inter_loss,st_loss)
        return loss

class CompReSSA(nn.Module):

    def __init__(self, teacher_feats_dim, queue_size=128000, T=0.04):
        super(CompReSSA, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim, queue_size, T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student = self.teacher_sample_similarities(student_feats, update=False)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher, similarities_student)
        return loss
    
class CompReSS(nn.Module):

    def __init__(self , teacher_feats_dim, student_feats_dim, queue_size=128000, T=0.04):
        super(CompReSS, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.student_sample_similarities = SampleSimilarities(student_feats_dim , queue_size , T).cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim , queue_size , T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student = self.student_sample_similarities(student_feats)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher , similarities_student)
        return loss

class SampleSimilaritiesMomentum(nn.Module):

    def __init__(self, feats_dim, queueSize, T):
        super(SampleSimilaritiesMomentum, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q , q_key):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        # update memory bank
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, q_key)
            self.index = (self.index + batchSize) % self.queueSize
        return out
    
class CompReSSMomentum(nn.Module):
    def __init__(self , teacher_feats_dim, student_feats_dim, queue_size=128000, T=0.04):
        super(CompReSSMomentum, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.student_sample_similarities = SampleSimilaritiesMomentum(student_feats_dim, queue_size, T).cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim, queue_size, T).cuda()

    def forward(self, teacher_feats, student_feats, student_feats_key):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)
        student_feats_key = self.l2norm(student_feats_key)

        similarities_student = self.student_sample_similarities(student_feats, student_feats_key)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher, similarities_student)
        return loss
       
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class KLD(nn.Module):

    def forward(self, targets, inputs):
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')
