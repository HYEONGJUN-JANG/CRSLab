# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt
from sklearn.metrics import f1_score, precision_score, recall_score
from platform import system as sysChecker
class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']
        self.item_idset=set(self.item_ids) # HJ : F1_REC
        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        self.f1rec_true=[]
        self.f1rec_pred=[]
        if sysChecker() == 'Linux':  # HJ KT-Server
            self.conv_epoch=90
        elif sysChecker() == "Windows":  # HJ Local
            self.conv_epoch = 1
        else:
            print("Check Your Platform")
            exit()
    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'pretrain':
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == 'train':
                    self.backward(gen_loss.sum())
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
            return pred
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.module.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())
        templist=[] #HJ
        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train') #HJ
            self.evaluator.report(epoch=epoch, mode='train')

            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()

            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                preds = self.step(batch, stage='conv', mode='test')
                # templist.append(self.printConv(batch,preds)) #HJ : Print Conv
                rec_f1=self.ConvRecF1(batch, preds)
                self.f1rec_true.extend(rec_f1[0])
                self.f1rec_pred.extend(rec_f1[1])
            self.evaluator.report(mode='test')
            logger.info(f'\nF1_Rec F1-Score For Test : {round(f1_score(self.f1rec_true, self.f1rec_pred, pos_label=1),2)}\n')
            logger.info(f'F1_Rec Precision-Score For Test : {round(precision_score(self.f1rec_true, self.f1rec_pred, pos_label=1),2)}\n')
            logger.info(f'F1_Rec Recall-Score For Test : {round(recall_score(self.f1rec_true, self.f1rec_pred, pos_label=1),2)}\n')
            logger.info(f'F1_Rec Recommend True Counter For Test : {len(list(filter(lambda x : x==1, self.f1rec_true)))}\n')
            logger.info(f'F1_Rec Pred True Counter For Test : {len(list(filter(lambda x : x==1, self.f1rec_pred)))}\n')
        # self.saveConv(templist)


    def fit(self):
        if sysChecker() == 'Linux':  # HJ KT-Server
            self.pretrain()
            self.train_recommender()
        elif sysChecker() == "Windows":  # HJ Local
            pass
        # self.pretrain()
        # self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass

    def ConvRecF1(self,batch,preds_tok):
        '''
        Args:
            self.item_idset : index 중 item 인 것을 모아놓은 set
            batch: batch (dict) 를 그대로 받는것
            preds_tok: preds로 step의 결과인 token index list (2-dim)
        Returns:
            true (list), pred (list) : 0, 1 로 이루어진 list
        '''
        true=[]
        pred=[]
        responses=batch[3].tolist()
        for bt in range(len(batch[0])):
            response_tok = list(filter(lambda x: x in self.item_idset, responses[bt]))  # response
            preds = list(filter(lambda x: x in self.item_idset, preds_tok[bt].tolist()))
            if response_tok : true.append(1)
            else: true.append(0)
            if preds: pred.append(1)
            else: pred.append(0)
        return true, pred


    def printConv(self,batch,preds):
        '''
        :param batch: batch (dict) 를 그대로 받는것
        :param preds: preds로 step의 결과인 token index list (2-dim)
        :return: 3-dim array [B,[context(1d list),response(text),pred(text)]]
        '''
        uttrlist = []
        # 원본문장?
        batchlen=len(batch[0])
        contexts=batch[0].tolist()
        # entities=batch[1].tolist()
        # words=batch[2].tolist()
        responses=batch[3].tolist()
        for bt in range(batchlen):
            context = self.list2Text([self.ind2tok.get(i) for i in list(filter(lambda x : x, contexts[bt]))])
            # ents=self.list2Text([self.ind2tok.get(i) for i in list(filter(lambda x : x, entities[bt]))])
            response=self.list2Text([self.ind2tok.get(i) for i in list(filter(lambda x : x, responses[bt]))]) # response
            pred=self.list2Text([self.ind2tok.get(i) for i in preds[bt].tolist()])

            uttrlist.append([context,response,pred])
        return uttrlist

    def list2Text(self,wlist,totok=None):
        txt=''
        for i in wlist:
            if i!=totok: txt+=f'{i} '
            else: txt+=f'<TOK> '
        return txt.rstrip()

    def saveConv(self,convlist):
        path=f"./convlog_{str(self.model.module).split('(')[0]}_withcopying.txt"
        with open(path,'w',encoding='UTF-8') as f:
            for i in convlist:
                for k in i:
                    context, response, pred = k
                    f.write("<< Context >>\n")
                    # for cont in context:
                    f.write(f"{context}\n")
                    f.write(f"\n<<Real Response>> : {response}\n")
                    f.write(f"<<Created Response>>: {pred}\n")
                    f.write("\n================< NEW LINE >================\n\n")
        print(f"Conversation Saved in {path}")