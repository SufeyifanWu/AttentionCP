import abc
import numpy as np  
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn
from modified_bert import ModifiedBert,load_custom_model
from data_process import AmazonReviewsDatasetSplit
# loss function
# vector CI vs point CI
# the best method to find true "v"?


# firstly, we need to define the function to get the true "v"

class Find_v(object):
    def __init__(self):
        super(Find_v, self).__init__()
        
    def find_v(
        self,
        model,
        input_ids,
        attention_mask,
        token_type_ids,
        init_v,  
        y,       
        criterion=nn.CrossEntropyLoss(),
        lr=1e-1,
        steps=200,
    ):
        # 拷贝 + set requires_grad
        v = init_v.detach().clone()
        v.requires_grad_()
        optimizer = torch.optim.Adam([v], lr=lr)
        
        for step in range(steps):
            logits = model.g(input_ids, attention_mask, token_type_ids, v)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step} | loss = {loss.item()}")
        # print init_v和最终v的距离的范数
        print(f"Distance between init_v and v: {torch.norm(init_v - v)}")
        return v
    
    
# Load fine-tuned model and tokenizer
model_path = ".\\model_hub\\bert-base"
checkpoint_path = ".\\trained_model\\model_step_43000.pth"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model = ModifiedBert(model_path)
model.load_state_dict(state_dict)
model.eval()
model.to(device)



# Get attention weights from test set
tokenizer = model.get_tokenizer()
max_len = 512   

data = AmazonReviewsDatasetSplit(".\\data\\Reviews.csv", tokenizer, max_len)
_, _, test_dataset = data.get_data()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
# get the first batch of data
find_v = Find_v()
for i, data in enumerate(test_loader):
    if i < 1000:
        
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = data['label']
        
        if labels != 0:
            continue
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        
        # get the true "v"
        init_v,_ = model.f(input_ids, attention_mask, token_type_ids)
        print(init_v)
        true_v = find_v.find_v(model, input_ids, attention_mask, token_type_ids, init_v, labels)
        print(true_v)
        
        break


        
# class RegressionErrFunc(object):
#     __metaclass__ = abc.ABCMeta

#     def __init__(self):
#         super(RegressionErrFunc, self).__init__()

#     @abc.abstractmethod
#     def apply(self, prediction, y):
#         pass

#     @abc.abstractmethod
#     def apply_inverse(self, nc, significance):
#         pass


# class AbsErrorErrFunc(RegressionErrFunc):
#     def __init__(self):
#         super(AbsErrorErrFunc, self).__init__()

#     def apply(self, prediction, y):
#         err = np.abs(prediction - y)
#         if err.ndim > 1:
#             err = np.linalg.norm(err, ord=np.inf, axis=1)
#         return err

#     def apply_inverse(self, nc, significance):
#         nc = np.sort(nc)[::-1]
#         border = int(np.floor(significance * (nc.size + 1))) - 1
#         border = min(max(border, 0), nc.size - 1)
#         return np.vstack([nc[border], nc[border]])

# class BaseModelNc(object): 
#     __metaclass__ = abc.ABCMeta

#     def __init__(self, model, err_func, normalizer, beta):
#         super(BaseModelNc, self).__init__()
#         self.model = model
#         self.err_func = err_func
        
#     def predict(self, x, nc, significance=None):
#         pass


# class RegressorNc(BaseModelNc):
#     def __init__(self, model, err_func=AbsErrorErrFunc()):
#         super(RegressorNc, self).__init__(model, err_func)

#     def predict(self, x, nc, significance=None):
#         n_test = x.shape[0]
#         prediction = self.model.predict(x)
#         norm = np.ones(n_test)
#         if significance:
#             intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
#             err_dist = self.err_func.apply_inverse(nc, significance)  # (2, y_dim)
#             err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
#             if prediction.ndim > 1:  # CQR
#                 intervals[..., 0] = prediction - err_dist[:, 0]
#                 intervals[..., 1] = prediction + err_dist[:, 1]
#             else:  # regular conformal prediction
#                 err_dist *= norm[:, None, None]
#                 intervals[..., 0] = prediction[:, None] - err_dist[:, 0]
#                 intervals[..., 1] = prediction[:, None] + err_dist[:, 1]

#             return intervals
#         else:  # Not tested for CQR
#             significance = np.arange(0.01, 1.0, 0.01)
#             intervals = np.zeros((x.shape[0], 2, significance.size))

#             for i, s in enumerate(significance):
#                 err_dist = self.err_func.apply_inverse(nc, s)
#                 err_dist = np.hstack([err_dist] * n_test)
#                 err_dist *= norm

#                 intervals[:, 0, i] = prediction - err_dist[0, :]
#                 intervals[:, 1, i] = prediction + err_dist[0, :]

#             return intervals


# class FeatRegressorNc(BaseModelNc):
#     def __init__(self, model,
#                  # err_func=FeatErrorErrFunc(),
#                  inv_lr, inv_step, criterion=default_loss, feat_norm=np.inf, certification_method=0, cert_optimizer='sgd',
#                  normalizer=None, beta=1e-6, g_out_process=None):

#         err_func = AbsErrorErrFunc()
#         super(FeatRegressorNc, self).__init__(model, err_func, normalizer, beta)
#         self.criterion = criterion
#         self.inv_lr = inv_lr
#         self.inv_step = inv_step
#         self.certification_method = certification_method
#         self.cmethod = ['IBP', 'IBP+backward', 'backward', 'CROWN-Optimized'][self.certification_method]
#         print(f"Use {self.cmethod} method for certification")

#         self.cert_optimizer = cert_optimizer
#         # the function to post process the output of g, because FCN needs interpolate and reshape
#         self.g_out_process = g_out_process

#     def inv_g(self, z0, y, step=None, record_each_step=False):
#         # z0 is hidden layer output, y is the label
#         z = z0.detach().clone()
#         z = z.detach()
#         z.requires_grad_()
#         # if self.cert_optimizer == "sgd":
#         #     optimizer = torch.optim.SGD([z], lr=self.inv_lr)
#         # elif self.cert_optimizer == "adam":
#         optimizer = torch.optim.Adam([z], lr=self.inv_lr)

#         self.model.eval()
#         each_step_z = []
#         for _ in range(step):
#             pred = self.model.g(z)
#             # if self.g_out_process is not None:
#             #     pred = self.g_out_process(pred)
#             # 暂时使用0-1loss
#             loss = self.criterion(pred.squeeze(), y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if record_each_step:
#                 each_step_z.append(z.detach().cpu().clone())

#         if record_each_step:
#             return each_step_z
#         else:
#             return z.detach().cpu()

#     def get_each_step_err_dist(self, x, y, z_pred, steps):
#         each_step_z_true = self.inv_g(z_pred, y, step=steps, record_each_step=True)

#         if self.normalizer is not None:
#             raise NotImplementedError
#         else:
#             norm = np.ones(len(x))

#         err_dist_list = []
#         for i, step_z_true in enumerate(each_step_z_true):
#             err_dist = self.err_func.apply(z_pred.detach().cpu(), step_z_true.detach().cpu()).numpy() / norm
#             err_dist_list.append(err_dist)
#         return err_dist_list

#     def coverage_loose(self, x, y, z_pred, steps, val_significance):
#         z_pred_detach = z_pred.detach().clone()

#         idx = torch.randperm(len(z_pred_detach))
#         n_val = int(np.floor(len(z_pred_detach) / 5))
#         val_idx, cal_idx = idx[:n_val], idx[n_val:]

#         cal_x, val_x = x[cal_idx], x[val_idx]
#         cal_y, val_y = y[cal_idx], y[val_idx]
#         cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

#         cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)

#         val_coverage_list = []
#         for i, step_cal_score in enumerate(cal_score_list):
#             val_predictions = self.predict(x=val_x.detach().cpu().numpy(), nc=step_cal_score,
#                                            significance=val_significance)
#             val_y_lower, val_y_upper = val_predictions[..., 0], val_predictions[..., 1]
#             val_coverage, _ = compute_coverage(val_y.detach().cpu().numpy(), val_y_lower, val_y_upper, val_significance,
#                                                name="{}-th step's validation".format(i), verbose=False)
#             val_coverage_list.append(val_coverage)
#         return val_coverage_list, len(val_x)

#     def coverage_tight(self, x, y, z_pred,  steps, val_significance):
#         z_pred_detach = z_pred.detach().clone()

#         idx = torch.randperm(len(z_pred_detach))
#         n_val = int(np.floor(len(z_pred_detach) / 5))
#         val_idx, cal_idx = idx[:n_val], idx[n_val:]

#         cal_x, val_x = x[cal_idx], x[val_idx]
#         cal_y, val_y = y[cal_idx], y[val_idx]
#         cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

#         cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)
#         val_score_list = self.get_each_step_err_dist(val_x, val_y, val_z_pred, steps=steps)

#         val_coverage_list = []
#         for i, (cal_score, val_score) in enumerate(zip(cal_score_list, val_score_list)):
#             err_dist_threshold = self.err_func.apply_inverse(nc=cal_score, significance=val_significance)[0][0]
#             val_coverage = np.sum(val_score < err_dist_threshold) * 100 / len(val_score)
#             val_coverage_list.append(val_coverage)
#         return val_coverage_list, len(val_x)

#     def find_best_step_num(self, x, y, z_pred):
#         max_inv_steps = 200
#         val_significance = 0.1

#         each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)
#         # each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)

#         tolerance = 1
#         count = 0
#         final_coverage, best_step = None, None
#         for i, val_coverage in enumerate(each_step_val_coverage):
#             # print("{}-th step's validation coverage is {}".format(i, val_coverage))
#             if val_coverage > (1 - val_significance) * 100 and final_coverage is None:
#                 count += 1
#                 if count == tolerance:
#                     final_coverage = val_coverage
#                     best_step = i
#             elif val_coverage <= (1 - val_significance) * 100 and count > 0:
#                 count = 0

#         if final_coverage is None or best_step is None:
#             raise ValueError(
#                 "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
#         print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
#         return best_step + 1

#     def find_best_step_num_batch(self, dataloader):
#         max_inv_steps = 200
#         val_significance = 0.1

#         accumulate_val_coverage = np.zeros(max_inv_steps)
#         accumulate_val_num = 0
#         print("begin to find the best step number")
#         for x, _, y in tqdm(dataloader):
#             x, y = x.to(self.model.device), y.to(self.model.device)
#             z_pred = self.model.model.encoder(x)
#             # batch_each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
#             batch_each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
#             accumulate_val_coverage += np.array(batch_each_step_val_coverage) * val_num
#             accumulate_val_num += val_num

#         each_step_val_coverage = accumulate_val_coverage / accumulate_val_num

#         tolerance = 3
#         count = 0
#         final_coverage, best_step = None, None
#         for i, val_coverage in enumerate(each_step_val_coverage):
#             # print("{}-th step's validation tight coverage is {}".format(i, val_coverage))
#             if val_coverage > (1 - val_significance) * 100 and final_coverage is None:
#                 count += 1
#                 if count == tolerance:
#                     final_coverage = val_coverage
#                     best_step = i
#             elif val_coverage <= (1 - val_significance) * 100 and count > 0:
#                 count = 0

#         if final_coverage is None or best_step is None:
#             raise ValueError(
#                 "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
#         print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
#         return best_step + 1

#     def score(self, x, y=None):  # overwrite BaseModelNc.score()
#         self.model.model.eval()
#         n_test = x.shape[0]
#         x, y = torch.from_numpy(x).to(self.model.device), torch.from_numpy(y).to(self.model.device)
#         z_pred = self.model.model.encoder(x)

#         if self.inv_step is None:
#             self.inv_step = self.find_best_step_num(x, y, z_pred)

#         z_true = self.inv_g(z_pred, y, step=self.inv_step)

#         if self.normalizer is not None:
#             raise NotImplementedError
#         else:
#             norm = np.ones(n_test)

#         ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())  # || z_pred - z_true ||
#         ret_val = ret_val.numpy() / norm
#         return ret_val

#     def score_batch(self, dataloader):
#         self.model.model.eval()
#         if self.inv_step is None:
#             self.inv_step = self.find_best_step_num_batch(dataloader)

#         print('calculating score:')
#         ret_val = []
#         for x, _, y in tqdm(dataloader):
#             x, y = x.to(self.model.device), y.to(self.model.device)

#             if self.normalizer is not None:
#                 raise NotImplementedError
#             else:
#                 norm = np.ones(len(x))

#             z_pred = self.model.model.encoder(x)
#             z_true = self.inv_g(z_pred, y, step=self.inv_step)
#             batch_ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())
#             batch_ret_val = batch_ret_val.detach().cpu().numpy() / norm
#             ret_val.append(batch_ret_val)
#         ret_val = np.concatenate(ret_val, axis=0)
#         return ret_val

#     def predict(self, x, nc, significance=None):
#         n_test = x.shape[0]
#         prediction = self.model.predict(x)

#         if self.normalizer is not None:
#             norm = self.normalizer.score(x) + self.beta
#         else:
#             norm = np.ones(n_test)

#         if significance:
#             intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
#             feat_err_dist = self.err_func.apply_inverse(nc, significance)

#             if prediction.ndim > 1:
#                 if isinstance(x, torch.Tensor):
#                     x = x.to(self.model.device)
#                 else:
#                     x = torch.from_numpy(x).to(self.model.device)
#                 z = self.model.model.encoder(x).detach()

#                 lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
#                 ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])
#                 my_input = BoundedTensor(z, ptb)

#                 if 'Optimized' in self.cmethod:
#                     lirpa_model.set_bound_opts(
#                         {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
#                 lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)
#                 if self.g_out_process is not None:
#                     lb = self.g_out_process(lb)
#                     ub = self.g_out_process(ub)
#                 lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

#                 intervals[..., 0] = lb
#                 intervals[..., 1] = ub
#             else:
#                 if not isinstance(x, torch.Tensor):
#                     x = torch.from_numpy(x).to(self.model.device)
#                 z = self.model.model.encoder(x).detach()

#                 lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
#                 ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])  # feat_err_dist=[[0.122, 0.122]]
#                 my_input = BoundedTensor(z, ptb)

#                 if 'Optimized' in self.cmethod:
#                     lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
#                 lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
#                 if self.g_out_process is not None:
#                     lb = self.g_out_process(lb)
#                     ub = self.g_out_process(ub)
#                 lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

#                 intervals[..., 0] = lb
#                 intervals[..., 1] = ub

#             return intervals

#         else:
#             raise NotImplementedError

