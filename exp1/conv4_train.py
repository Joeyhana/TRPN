from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import TRPN
from backbone.conv4 import EmbeddingImagenet
import shutil
import os
import random
#import seaborn as sns
import time
from IPython import embed as e

class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 gcn_module,
                 data_loader):
        # set encoder and gnn
        self.enc_module = enc_module.to(tt.arg.device)
        self.gcn_module = gcn_module.to(tt.arg.device)


        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1, 2, 3], dim=0)
            self.gcn_module = nn.DataParallel(self.gcn_module, device_ids=[0, 1, 2, 3], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = list(self.enc_module.parameters()) + list(self.gcn_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.bce_loss = nn.BCELoss() #reduction='mean')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        val_acc = self.val_acc

        num_supports = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_queries = tt.arg.num_ways_train * 1
        num_samples = num_supports + num_queries
        num_tasks = tt.arg.meta_batch_size
        num_ways = tt.arg.num_ways_train
        num_shots = tt.arg.num_shots_train

        # batch_size x num_samples x num_samples
        support_edge_mask = torch.zeros(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        # batch_size x num_samples x num_samples
        evaluation_mask = torch.ones(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)

        # for semi-supervised setting, ignore unlabeled support sets for evaluation
        for c in range(tt.arg.num_ways_train):
            evaluation_mask[:,
            ((c + 1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_train,
            :num_supports] = 0
            evaluation_mask[:, :num_supports,
            ((c + 1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_train] = 0

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            # support_data: batch_size x num_supports x 3 x 84 x 84
            # support_label: batch_size x num_supports
            # query_data: batch_size x num_queries x 3 x 84 x 84
            # query_label: batch_size x num_queries
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways_train,
                                                                     num_shots=tt.arg.num_shots_train,
                                                                     seed=iter + tt.arg.seed)
            # set as single data
            # batch_size x num_samples x 3 x 84 x 84
            full_data = torch.cat([support_data, query_data], 1)
            # batch_size x num_samples
            full_label = torch.cat([support_label, query_label], 1)

            # batch_size x 2 x num_samples x num_samples
            full_edge = self.label2edge(full_label)

            # set init edge
            # batch_size x 2 x num_samples x num_samples
            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # for semi-supervised setting,
            for c in range(tt.arg.num_ways_train):
                init_edge[:, :, ((c+1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_train, :num_supports] = 0.5
                init_edge[:, :, :num_supports, ((c+1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_train] = 0.5

            # set as train mode
            self.enc_module.train()
            self.gcn_module.train()
            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1) # batch_size x num_samples x featdim

            # num_tasks x num_quries x num_supports, num_tasks x num_samples x num_samples
            query_score_list, learned_score_list = self.gcn_module(node_feat=full_data, adj=init_edge[:, 0, :num_supports, :num_supports])
            # print(query_score_list.size(), learned_score_list.size())
            # (4) compute loss
            loss1_pos = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * full_edge[:,0,:,:] * evaluation_mask).sum() / ((evaluation_mask * full_edge[:,0,:,:]).sum())
            loss1_neg = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * (1 -  full_edge[:,0,:,:]) * evaluation_mask).sum() / ( (evaluation_mask * (1. - full_edge[:,0,:,:])).sum())

            loss2_pos =( self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) * full_edge[:, 0, num_supports:, :] * evaluation_mask[:,num_supports:, :]).sum() / ((evaluation_mask[:,num_supports:, :]*full_edge[:, 0, num_supports:, :]).sum())
            loss2_neg =( self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) *(1. -  full_edge[:, 0, num_supports:, :]) * evaluation_mask[:,num_supports:, :]).sum() / ((evaluation_mask[:,num_supports:, :]*(1.-full_edge[:, 0, num_supports:, :])).sum())



            # compute node accuracy: num_tasks x num_quries x num_ways == {num_tasks x num_quries x num_supports} * {num_tasks x num_supports x num_ways}
            query_node_pred1 = torch.bmm(query_score_list[:, :, :num_supports], self.one_hot_encode(tt.arg.num_ways_train, support_label.long()))
            query_node_accr1 = torch.eq(torch.max(query_node_pred1, -1)[1], query_label.long()).float().mean()
            total_loss = (loss1_pos + loss1_neg + loss2_pos + loss2_neg) / 4
            e()
            total_loss.backward()
            self.optimizer.step()
            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],lr=tt.arg.lr,iter=self.global_step)

            # logging
            tt.log_scalar('train/loss', total_loss, self.global_step)
            tt.log_scalar('train/accr1', query_node_accr1, self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gcn_module_state_dict': self.gcn_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                    }, is_best)

            tt.log_step(global_step=self.global_step)

    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
        num_queries = tt.arg.num_ways_test * 1
        num_samples = num_supports + num_queries
        num_tasks = tt.arg.test_batch_size
        num_ways = tt.arg.num_ways_test
        num_shots = tt.arg.num_shots_test

        support_edge_mask = torch.zeros(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
        # for semi-supervised setting, ignore unlabeled support sets for evaluation
        for c in range(tt.arg.num_ways_test):
            evaluation_mask[:,
            ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test,
            :num_supports] = 0
            evaluation_mask[:, :num_supports,
            ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test] = 0

        query_edge_losses = []
        query_node_accrs1 = []

        # for each iteration
        for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                       num_ways=tt.arg.num_ways_test,
                                                                       num_shots=tt.arg.num_shots_test,
                                                                       seed=iter)
            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # for semi-supervised setting,
            for c in range(tt.arg.num_ways_test):
                init_edge[:, :, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test, :num_supports] = 0.5
                init_edge[:, :, :num_supports, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test] = 0.5
            
            # set as train mode
            self.enc_module.eval()
            self.gcn_module.eval()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)
            
            # num_tasks x num_quries x num_ways
            query_score_list, learned_score_list = self.gcn_module(node_feat=full_data, adj=init_edge[:, 0, :num_supports, :num_supports])
 
            # (4) compute loss
            loss1_pos = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * full_edge[:,0,:,:] * evaluation_mask).sum() / ((evaluation_mask * full_edge[:,0,:,:]).sum())
            loss1_neg = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * (1 -  full_edge[:,0,:,:]) * evaluation_mask).sum() / ( (evaluation_mask * (1. - full_edge[:,0,:,:])).sum())

            loss2_pos =( self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) * full_edge[:, 0, num_supports:, :] * evaluation_mask[:,num_supports:, :]).sum() / ((evaluation_mask[:,num_supports:, :]*full_edge[:, 0, num_supports:, :]).sum())
            loss2_neg =( self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) *(1. -  full_edge[:, 0, num_supports:, :]) * evaluation_mask[:,num_supports:, :]).sum() / ((evaluation_mask[:,num_supports:, :]*(1.-full_edge[:, 0, num_supports:, :])).sum())
            # compute node accuracy: num_tasks x num_quries x num_ways == {num_tasks x num_quries x num_supports} * {num_tasks x num_supports x num_ways}
            #query_score_list = query_score_list * evaluation_mask[:,num_supports:]
            query_node_pred1 = torch.bmm(query_score_list[:, :, :num_supports], self.one_hot_encode(tt.arg.num_ways_test, support_label.long()))
            query_node_accr1 = torch.eq(torch.max(query_node_pred1, -1)[1], query_label.long()).float().mean()

            total_loss = (loss1_pos + loss1_neg + loss2_pos + loss2_neg) / 4
            # print(total_loss)
            query_edge_losses += [total_loss.item()]
            query_node_accrs1 += [query_node_accr1.item()]

        # logging
        if log_flag:
            tt.log('---------------------------')
            tt.log_scalar('{}/edge_loss'.format(partition), np.array(query_edge_losses).mean(), self.global_step)
            tt.log_scalar('{}/node_accr1'.format(partition), np.array(query_node_accrs1).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy1: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs1).mean() * 100,
                    np.array(query_node_accrs1).std() * 100,
                    1.96 * np.array(query_node_accrs1).std() / np.sqrt(float(len(np.array(query_node_accrs1)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs1).mean()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, './model_checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth')
        if is_best:
            shutil.copyfile('./model_checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth',
                            './model_checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth')


if __name__ == '__main__':

    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = '../dataset/' 
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.meta_batch_size = 20 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.transductive = True if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    # model parameter related
    tt.arg.emb_size = 128

    # train, test parameters
    tt.arg.train_iteration = 150000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000 if tt.arg.test_interval is None else tt.arg.test_interval
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = 'num_ways_' + str(tt.arg.num_ways) + '_num_shots_' + str(tt.arg.num_shots) + '_model/' if tt.arg.experiment is None else tt.arg.experiment

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('./model_checkpoints'):
        os.makedirs('./model_checkpoints')
    if not os.path.exists('./model_checkpoints/' + tt.arg.experiment):
        os.makedirs('./model_checkpoints/' + tt.arg.experiment)


    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    gcn_module = TRPN(n_feat=tt.arg.emb_size, n_queries=tt.arg.num_ways_train * 1)

    if tt.arg.dataset == 'mini':
        train_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'tiered':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           gcn_module=gcn_module,
                           data_loader=data_loader)

    trainer.train()
